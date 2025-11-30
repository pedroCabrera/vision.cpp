#include "testing.h"
#include "visp/ml.h"
#include "visp/nn.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

namespace visp {
#ifdef VISP_VULKAN
VISP_TEST(backend_available_vulkan) {
    CHECK(backend_is_available(backend_type::cpu));
    if (backend_is_available(backend_type::gpu)) {
        CHECK(backend_is_available(backend_type::vulkan));
    }
}
#endif
#ifdef VISP_CUDA
VISP_TEST(backend_available_cuda) {
    CHECK(backend_is_available(backend_type::cpu));
    if (backend_is_available(backend_type::gpu)) {
        CHECK(backend_is_available(backend_type::cuda));
    }
}
#endif


VISP_TEST(model_transfer_type_conversion) {
    model_weights src = model_init(2);

    tensor i = ggml_new_tensor_1d(src, GGML_TYPE_I32, 2);
    ggml_set_name(i, "i32_tensor");
    auto i32_data = std::array{4, -1};
    i->data = i32_data.data();

    tensor f = ggml_new_tensor_1d(src, GGML_TYPE_F16, 2);
    ggml_set_name(f, "f16_tensor");
    auto f16_data = std::array{ggml_fp32_to_fp16(2.5f), ggml_fp32_to_fp16(-0.5f)};
    f->data = f16_data.data();

    backend_device dev = backend_init(backend_type::cpu);
    model_weights dst = model_init(2);
    model_transfer(src, dst, dev, GGML_TYPE_F32); // f16 -> f32 conversion

    int32_t const* i32_result = (int32_t const*)ggml_get_tensor(dst, "i32_tensor")->data;
    CHECK_EQUAL(i32_result[0], 4);
    CHECK_EQUAL(i32_result[1], -1);

    tensor f_result = ggml_get_tensor(dst, "f16_tensor");
    CHECK(f_result->type == GGML_TYPE_F32);
    float const* f32_result = (float const*)f_result->data;
    CHECK_EQUAL(f32_result[0], 2.5f);
    CHECK_EQUAL(f32_result[1], -0.5f);
}

VISP_TEST(model_transfer_layout_conversion) {
    model_weights src = model_init(3);

    tensor conv_dw = ggml_new_tensor_4d(src, GGML_TYPE_F32, 2, 2, 1, 3); // wh1c
    ggml_set_name(conv_dw, "conv_dw");
    auto conv_dw_data = std::array<float, 2 * 2 * 1 * 3>{};
    std::iota(conv_dw_data.begin(), conv_dw_data.end(), 1.0f);
    conv_dw->data = conv_dw_data.data();

    tensor conv = ggml_new_tensor_4d(src, GGML_TYPE_F32, 2, 2, 4, 3); // whco
    ggml_set_name(conv, "conv");
    auto conv_data = std::array<float, 2 * 2 * 3 * 4>{};
    std::iota(conv_data.begin(), conv_data.end(), 1.0f);
    conv->data = conv_data.data();

    tensor no_conv = ggml_new_tensor_1d(src, GGML_TYPE_F32, 2);
    ggml_set_name(no_conv, "no_conv");
    auto no_conv_data = std::array<float, 2>{1.0f, 2.0f};
    no_conv->data = no_conv_data.data();

    auto conv_weights = std::array{0, 1};
    auto src_layout = tensor_data_layout::whcn;
    auto dst_layout = tensor_data_layout::cwhn;

    backend_device dev = backend_init(backend_type::cpu);
    model_weights dst = model_init(3);
    model_transfer(src, dst, dev, GGML_TYPE_COUNT, src_layout, dst_layout, conv_weights);

    auto conv_dw_expected = std::array{
        1.0f, 5.0f, 9.0f,  //
        2.0f, 6.0f, 10.0f, //
        3.0f, 7.0f, 11.0f, //
        4.0f, 8.0f, 12.0f  //
    };
    float const* conv_dw_result = (float const*)ggml_get_tensor(dst, "conv_dw")->data;
    for (int i = 0; i < int(conv_dw_expected.size()); ++i) {
        CHECK_EQUAL(conv_dw_result[i], conv_dw_expected[i]);
    }

    auto conv_expected = std::array{
        1.0f,  5.0f,  9.0f,  13.0f, 2.0f, 6.0f, 10.0f, 14.0f, //
        3.0f,  7.0f,  11.0f, 15.0f, 4.0f, 8.0f, 12.0f, 16.0f, //

        17.0f,  21.0f,  25.0f, 29.0f, 18.0f, 22.0f, 26.0f, 30.0f, //
        19.0f, 23.0f, 27.0f, 31.0f, 20.0f, 24.0f, 28.0f, 32.0f, //

        33.0f, 37.0f, 41.0f, 45.0f, 34.0f, 38.0f, 42.0f, 46.0f, //
        35.0f, 39.0f, 43.0f, 47.0f, 36.0f, 40.0f, 44.0f, 48.0f  //
    };
    float const* conv_result = (float const*)ggml_get_tensor(dst, "conv")->data;
    for (int i = 0; i < int(conv_expected.size()); ++i) {
        CHECK_EQUAL(conv_result[i], conv_expected[i]);
    }

    float const* no_conv_result = (float const*)ggml_get_tensor(dst, "no_conv")->data;
    CHECK_EQUAL(no_conv_result[0], 1.0f);
    CHECK_EQUAL(no_conv_result[1], 2.0f);
}

#ifdef VISP_CUDA
VISP_TEST(conv2d_deform_cuda_matches_cpu) {
    if (!backend_is_available(backend_type::cuda)) {
        throw test_skip{"CUDA backend not available"};
    }

    struct test_case {
        int src_w;
        int src_h;
        int c_in;
        int c_out;
        int kw;
        int kh;
        int stride;
        int pad;
        int batch;
    };

    const std::array test_cases{
        test_case{8, 6, 4, 3, 3, 3, 1, 1, 1},
        test_case{32, 32, 64, 256, 1, 1, 1, 0, 1},
    };

    auto fill_with = [](std::vector<float>& data, float scale, float step) {
        for (size_t i = 0; i < data.size(); ++i) {
            float angle = step * float(i);
            data[i] = static_cast<float>(scale * std::sin(angle));
        }
    };

    auto run_backend = [&](backend_type backend, test_case const& cfg) {
        const int dst_w = (cfg.src_w + 2 * cfg.pad - cfg.kw) / cfg.stride + 1;
        const int dst_h = (cfg.src_h + 2 * cfg.pad - cfg.kh) / cfg.stride + 1;

        const size_t input_elems = size_t(cfg.src_w) * cfg.src_h * cfg.c_in * cfg.batch;
        const size_t weight_elems = size_t(cfg.kw) * cfg.kh * cfg.c_in * cfg.c_out;
        const size_t offset_elems = size_t(dst_w) * dst_h * 2 * cfg.kw * cfg.kh * cfg.batch;
        const size_t mask_elems = size_t(dst_w) * dst_h * cfg.kw * cfg.kh * cfg.batch;

        std::vector<float> input_data(input_elems);
        std::vector<float> weight_data(weight_elems);
        std::vector<float> offset_data(offset_elems);
        std::vector<float> mask_data(mask_elems);

        fill_with(input_data, 0.7f, 0.01f);
        fill_with(weight_data, 0.5f, 0.02f);
        fill_with(offset_data, 0.25f, 0.005f);
        fill_with(mask_data, 0.5f, 0.015f);
        for (float& v : mask_data) {
            v = 0.5f + v; // keep weights positive to avoid degenerate zero masks
        }

        backend_device dev = backend_init(backend);
        model_weights weights = model_init(4);
        weights.buffer_type = backend;
        compute_graph graph = compute_graph_init();
        model_ref m(weights, graph);
        m.backend = backend;

        tensor weight = ggml_new_tensor_4d(
            m.weights_context, GGML_TYPE_F32, cfg.kw, cfg.kh, cfg.c_in, cfg.c_out);
        ggml_set_name(weight, "conv.weight");

        tensor offset = ggml_new_tensor_4d(
            m.weights_context, GGML_TYPE_F32, dst_w, dst_h, 2 * cfg.kw * cfg.kh, cfg.batch);
        ggml_set_name(offset, "offset");

        tensor mask = ggml_new_tensor_4d(
            m.weights_context, GGML_TYPE_F32, dst_w, dst_h, cfg.kw * cfg.kh, cfg.batch);
        ggml_set_name(mask, "mask");

        model_allocate(weights, dev);
        transfer_to_backend(weight, span<const float>(weight_data));
        transfer_to_backend(offset, span<const float>(offset_data));
        transfer_to_backend(mask, span<const float>(mask_data));

        tensor input = compute_graph_input(
            m, GGML_TYPE_F32, i64x4{cfg.src_w, cfg.src_h, cfg.c_in, cfg.batch}, "input");

        tensor output = conv_2d_deform(
            m, input, weight, offset, mask, cfg.stride, cfg.pad);
        output = compute_graph_output(m, output, "output");

        compute_graph_allocate(graph, dev);
        transfer_to_backend(input, span<const float>(input_data));
        compute(graph, dev);

        tensor_data result = transfer_from_backend(output);
        auto span_f32 = result.as_f32();
        return std::vector<float>(span_f32.begin(), span_f32.end());
    };

    for (test_case const& cfg : test_cases) {
        auto cpu = run_backend(backend_type::cpu, cfg);
        auto cuda = run_backend(backend_type::cuda, cfg);

        ASSERT(cpu.size() == cuda.size(), "Mismatched output sizes");
        float max_abs = 0.0f;
        double sum_sq = 0.0;
        for (size_t i = 0; i < cpu.size(); ++i) {
            float diff = cpu[i] - cuda[i];
            max_abs = std::max(max_abs, std::abs(diff));
            sum_sq += diff * diff;
        }
    float rmse = static_cast<float>(std::sqrt(sum_sq / double(cpu.size())));
        test_set_info(format(
            "{}x{}x{} -> {} rmse={} max_abs={}",
            cfg.src_w, cfg.src_h, cfg.c_in, cfg.c_out, rmse, max_abs));
        CHECK(max_abs < 1e-4f);
    }
}
#endif // VISP_CUDA

} // namespace visp