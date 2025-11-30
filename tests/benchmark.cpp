#include "testing.h"
#include "visp/image.h"
#include "visp/ml.h"
#include "visp/util.h"
#include "visp/vision.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace visp;

struct bench_timings {
    double mean = 0.0;
    double stdev = 0.0;
};

struct bench_run {
    bench_timings time;
    ggml_type weight_type = GGML_TYPE_COUNT;
};

// Controls whether to include H2D transfers inside the timed loop
static bool g_include_transfer_each_iter = false;
// Print per-iteration timings
static bool g_trace = false;
// Override total iterations (before GPU multiplier)
static bool g_override_iters = false;
static int g_iters = 0;
// Disable GPU iteration multiplier
static bool g_no_gpu_mul = false;

enum class esrgan_input_choice {
    def, // keep current benchmark default (F32)
    f32  // explicitly F32
};
static esrgan_input_choice g_esrgan_input = esrgan_input_choice::def;

// Optional override for model weight float type
static ggml_type g_weight_type = GGML_TYPE_COUNT; // default: backend/model preferred

struct input_transfer {
    tensor x;
    span<byte const> data;

    input_transfer(tensor x, span<byte const> data) : x(x), data(data) {}
    input_transfer(tensor x, image_view img) : x(x), data((byte const*)img.data, n_bytes(img)) {}
};

bench_timings run_benchmark(
    compute_graph& graph,
    backend_device& backend,
    int iterations,
    std::vector<input_transfer> const& transfers = {},
    bool include_transfer_each_iter = false) {

    if (g_override_iters) {
        iterations = g_iters;
    }

    if ((backend.type() & backend_type::gpu) && !g_no_gpu_mul) {
        iterations *= 4;
    }

    std::vector<double> timings;
    timings.reserve(iterations);

    // Warm-up: pre-transfer once (if excluding transfer from timing), then compute
    if (!transfers.empty() && !include_transfer_each_iter) {
        for (const auto& transfer : transfers) {
            transfer_to_backend(transfer.x, transfer.data);
        }
    }
    compute(graph, backend);

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        if (include_transfer_each_iter) {
            for (const auto& transfer : transfers) {
                transfer_to_backend(transfer.x, transfer.data);
            }
        }
        compute(graph, backend);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        double ms = elapsed.count();
        timings.push_back(ms);
        if (g_trace) {
            printf("  iter %3d: %.3f ms%s\n", i + 1, ms, include_transfer_each_iter ? " (xfer+compute)" : "");
        }
    }

    double mean = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
    double sq_sum = std::inner_product(timings.begin(), timings.end(), timings.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / timings.size() - mean * mean);
    return {mean, stdev};
}

bench_run benchmark_sam(path model_path, backend_device& backend) {
    path input_path = test_dir().input / "cat-and-hat.jpg";

    sam_model model = sam_load_model(model_path.string().c_str(), backend, g_weight_type);
    image_data input = image_load(input_path.string().c_str());
    image_data input_data = sam_process_input(input, model.params);

    sam_encode(model, image_view(input));
    bench_timings encoder_timings = run_benchmark(
        model.encoder, backend, 16, {{model.input_image, input_data}}, g_include_transfer_each_iter);

    sam_compute(model, i32x2{200, 300});
    bench_timings decoder_timings = run_benchmark(model.decoder, backend, 50, {}, g_include_transfer_each_iter);

    bench_run r;
    r.time = {
        encoder_timings.mean + decoder_timings.mean,
        std::sqrt(
            encoder_timings.stdev * encoder_timings.stdev +
            decoder_timings.stdev * decoder_timings.stdev)};
    r.weight_type = model.weights.float_type();
    return r;
}

bench_run benchmark_birefnet(path model_path, backend_device& backend) {
    path input_path = test_dir().input / "wardrobe.jpg";

    birefnet_model model = birefnet_load_model(model_path.string().c_str(), backend, g_weight_type);
    image_data input = image_load(input_path.string().c_str());
    image_data input_data = birefnet_process_input(input, model.params);

    birefnet_compute(model, input);
    bench_run r;
    r.time = run_benchmark(model.graph, backend, 8, {{model.input, input_data}}, g_include_transfer_each_iter);
    r.weight_type = model.weights.float_type();
    return r;
}

bench_run benchmark_depth_anything(path model_path, backend_device& backend) {
    path input_path = test_dir().input / "wardrobe.jpg";

    depthany_model model = depthany_load_model(model_path.string().c_str(), backend, g_weight_type);
    image_data input = image_load(input_path.string().c_str());
    depthany_compute(model, input);

    image_data input_data = depthany_process_input(input, model.params);
    bench_run r;
    r.time = run_benchmark(model.graph, backend, 12, {{model.input, input_data}}, g_include_transfer_each_iter);
    r.weight_type = model.weights.float_type();
    return r;
}

bench_run benchmark_migan(path model_path, backend_device& backend) {
    path image_path = test_dir().input / "bench-image.jpg";
    path mask_path = test_dir().input / "bench-mask.png";

    migan_model model = migan_load_model(model_path.string().c_str(), backend, g_weight_type);
    image_data image = image_load(image_path.string().c_str());
    image_data mask = image_load(mask_path.string().c_str());
    image_data input_data = migan_process_input(image, mask, model.params);

    migan_compute(model, image, mask);
    bench_run r;
    r.time = run_benchmark(model.graph, backend, 32, {{model.input, input_data}}, g_include_transfer_each_iter);
    r.weight_type = model.weights.float_type();
    return r;
}

bench_run benchmark_esrgan(path model_path, backend_device& backend) {
    path input_path = test_dir().input / "vase-and-bowl.jpg";

    esrgan_model model = esrgan_load_model(model_path.string().c_str(), backend, g_weight_type);
    image_data input = image_load(input_path.string().c_str());
    image_data input_data = image_u8_to_f32(input, image_format::rgb_f32);

    // Build the graph similarly to the model path but choose input precision per flag
    compute_graph graph = compute_graph_init(esrgan_estimate_graph_size(model.params));
    model_ref m(model.weights, graph);
    i64x4 input_shape = {3, input.extent[0], input.extent[1], 1};
    ggml_type in_type = GGML_TYPE_F32;
    if (g_esrgan_input == esrgan_input_choice::f32) {
        in_type = GGML_TYPE_F32;
        input_data = image_u8_to_f32(input, image_format::rgb_f32);
    } else {
        // keep current behavior (F32)
        in_type = GGML_TYPE_F32;
        input_data = image_u8_to_f32(input, image_format::rgb_f32);
    }
    model.input = compute_graph_input(m, in_type, input_shape);
    model.output = esrgan_generate(m, model.input, model.params);

    compute_graph_allocate(graph, backend);
    bench_run r;
    r.time = run_benchmark(graph, backend, 8, {{model.input, input_data}}, g_include_transfer_each_iter);
    r.weight_type = model.weights.float_type();
    return r;
}

backend_device initialize_backend(std::string_view backend_type) {
    if (backend_type == "cpu") {
        backend_device cpu = backend_init(backend_type::cpu);
        backend_set_n_threads(cpu, (int)std::thread::hardware_concurrency());
        return cpu;
    } else if (backend_type == "vulkan") {
        return backend_init(backend_type::vulkan);
    } else if (backend_type == "cuda") {
        return backend_init(backend_type::cuda);
    } else if (backend_type == "gpu") {
        return backend_init(backend_type::gpu);
    } else {
        throw std::invalid_argument("Invalid backend type. Use 'cpu', 'gpu', 'vulkan' or 'cuda'.");
    }
}

struct bench_result {
    std::string_view arch;
    std::string_view model;
    std::string_view backend;
    bench_timings time;
    ggml_type weight_type = GGML_TYPE_COUNT;
};

bench_result benchmark_model(
    std::string_view arch, std::string_view model, backend_device& backend) {

    bench_result result;
    result.arch = arch;
    result.model = model;
    result.backend = to_string(backend.type());

    auto select_model = [&](std::string_view model, std::string_view fallback) {
        if (model.empty()) {
            result.model = fallback;
            return test_dir().models / fallback;
        }
        path p = path(model);
        if (!exists(p)) {
            fprintf(stderr, "Model file not found: %s\n", p.string().c_str());
            result.model = fallback;
            return test_dir().models / fallback;
        }
        return p;
    };

    if (arch == "sam") {
        path model_path = select_model(model, "MobileSAM-F16.gguf");
        bench_run r = benchmark_sam(model_path, backend);
        result.time = r.time;
        result.weight_type = r.weight_type;

    } else if (arch == "birefnet") {
        path model_path = select_model(model, "BiRefNet-lite-F16.gguf");
        bench_run r = benchmark_birefnet(model_path, backend);
        result.time = r.time;
        result.weight_type = r.weight_type;

    } else if (arch == "depthany") {
        path model_path = select_model(model, "Depth-Anything-V2-Small-F16.gguf");
        bench_run r = benchmark_depth_anything(model_path, backend);
        result.time = r.time;
        result.weight_type = r.weight_type;

    } else if (arch == "migan") {
        path model_path = select_model(model, "MIGAN-512-places2-F16.gguf");
        bench_run r = benchmark_migan(model_path, backend);
        result.time = r.time;
        result.weight_type = r.weight_type;

    } else if (arch == "esrgan") {
        path model_path = select_model(model, "RealESRGAN-x4plus_anime-6B-F16.gguf");
        bench_run r = benchmark_esrgan(model_path, backend);
        result.time = r.time;
        result.weight_type = r.weight_type;

    } else {
        fprintf(stderr, "Unknown model architecture: %s\n", arch.data());
    }
    return result;
}

char const* next_arg(int argc, char** argv, int& i) {
    if (++i < argc) {
        return argv[i];
    } else {
        throw except("Missing argument after {}", argv[i - 1]);
    }
}

void print(fixed_string<128> const& str) {
    printf("%s", str.c_str());
}

int main(int argc, char** argv) {
    std::vector<std::pair<std::string_view, std::string_view>> models;
    std::vector<std::string_view> backends;
    bool include_transfer_each_iter = false;
    // Runtime controls (replaces env vars)
    enum class trinary { unset, off, on };
    trinary opt_cuda_graphs = trinary::unset;
    trinary opt_cuda_allow_f16_interp = trinary::unset;
    trinary opt_flash_attention = trinary::unset;
    trinary opt_cuda_mmq = trinary::unset;
    trinary opt_cuda_cublas = trinary::unset;

    try {

        for (int i = 1; i < argc; ++i) {
            std::string_view arg(argv[i]);
            if (arg == "-m" || arg == "--model") {
                std::string_view text = next_arg(argc, argv, i);
                auto p = text.find(':');
                if (p == std::string_view::npos) {
                    models.push_back({text, ""});
                } else {
                    std::string_view arch = text.substr(0, p);
                    std::string_view model = text.substr(p + 1);
                    models.emplace_back(arch, model);
                }
            } else if (arg == "-b" || arg == "--backend") {
                backends.push_back(next_arg(argc, argv, i));
            } else if (arg == "--include-transfer") {
                include_transfer_each_iter = true;
            } else if (arg == "--trace") {
                g_trace = true;
            } else if (arg == "--iters") {
                g_override_iters = true;
                g_iters = std::stoi(std::string(next_arg(argc, argv, i)));
            } else if (arg == "--no-gpu-mul") {
                g_no_gpu_mul = true;
            } else if (arg == "--esrgan-input") {
                std::string choice = std::string(next_arg(argc, argv, i));
                if (choice == "f32") g_esrgan_input = esrgan_input_choice::f32;
                else g_esrgan_input = esrgan_input_choice::def;
            } else if (arg == "--weights") {
                std::string w = std::string(next_arg(argc, argv, i));
                if (w == "f16") g_weight_type = GGML_TYPE_F16;
                else if (w == "f32") g_weight_type = GGML_TYPE_F32;
                else g_weight_type = GGML_TYPE_COUNT;
            } else if (arg == "--cuda-graphs") {
                std::string v = std::string(next_arg(argc, argv, i));
                opt_cuda_graphs = (v == "on") ? trinary::on : (v == "off" ? trinary::off : trinary::unset);
            } else if (arg == "--cuda-allow-f16-interp") {
                std::string v = std::string(next_arg(argc, argv, i));
                opt_cuda_allow_f16_interp = (v == "on") ? trinary::on : (v == "off" ? trinary::off : trinary::unset);
            } else if (arg == "--flash-attention") {
                std::string v = std::string(next_arg(argc, argv, i));
                opt_flash_attention = (v == "on") ? trinary::on : (v == "off" ? trinary::off : trinary::unset);
            } else if (arg == "--cuda-mmq") {
                std::string v = std::string(next_arg(argc, argv, i));
                opt_cuda_mmq = (v == "on") ? trinary::on : (v == "off" ? trinary::off : trinary::unset);
            } else if (arg == "--cuda-cublas") {
                std::string v = std::string(next_arg(argc, argv, i));
                opt_cuda_cublas = (v == "on") ? trinary::on : (v == "off" ? trinary::off : trinary::unset);
            } else {
                throw std::invalid_argument("Unknown argument: " + std::string(arg));
            }
        }

    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    if (models.empty()) {
        models = {{"sam", ""}, {"birefnet", ""}, {"migan", ""}, {"esrgan", ""}};
    }

    if (backends.empty()) {
        backends = {"cpu", "gpu"};
    }

    try {
        fixed_string<128> line;
        size_t n_tests = models.size() * backends.size();
        std::vector<bench_result> results;
        results.reserve(n_tests);

    // propagate flag to benchmarks
    g_include_transfer_each_iter = include_transfer_each_iter;

    // Apply runtime options BEFORE initializing any backend
    if (opt_cuda_graphs != trinary::unset) {
        visp_set_cuda_graphs_enabled(opt_cuda_graphs == trinary::on);
    }
    if (opt_cuda_allow_f16_interp != trinary::unset) {
        visp_set_cuda_allow_f16_interpolation(opt_cuda_allow_f16_interp == trinary::on);
    }
    if (opt_flash_attention != trinary::unset) {
        visp_set_flash_attention_enabled(opt_flash_attention == trinary::on);
    }
    // For MMQ/CUBLAS, ggml-cuda expects env vars. Set them programmatically for this process.
    #if defined(_WIN32)
        if (opt_cuda_mmq != trinary::unset) {
            _putenv_s("GGML_CUDA_FORCE_MMQ", opt_cuda_mmq == trinary::on ? "1" : "0");
        }
        if (opt_cuda_cublas != trinary::unset) {
            _putenv_s("GGML_CUDA_FORCE_CUBLAS", opt_cuda_cublas == trinary::on ? "1" : "0");
        }
    #else
        if (opt_cuda_mmq != trinary::unset) {
            setenv("GGML_CUDA_FORCE_MMQ", opt_cuda_mmq == trinary::on ? "1" : "0", 1);
        }
        if (opt_cuda_cublas != trinary::unset) {
            setenv("GGML_CUDA_FORCE_CUBLAS", opt_cuda_cublas == trinary::on ? "1" : "0", 1);
        }
    #endif

    int i = 0;
        for (auto&& backend : backends) {
            backend_device backend_device = initialize_backend(backend);
            for (auto&& model : models) {
                print(format(
                    line, "[{: <2}/{: <2}] Running {} on {}...\n", ++i, n_tests, model.first,
                    backend));

                // Run selected model/arch benchmark
                bench_result r = benchmark_model(model.first, model.second, backend_device);
                results.push_back(r);
            }
        }

        printf("\n");
        print(format(
            line, "| {: <10} | {: <30} | {: <6} | {: <6} | {: >11} | {: >6} |\n", "Arch", "Model", "Device", "Wgts", "Avg", "Dev"));
        printf("|:-----------|:-------------------------------|:-------|:------|------------:|-------:|\n");
        for (const auto& result : results) {
            auto model = result.model.substr(std::max(int(result.model.length()) - 30, 0));
            print(format(
                line, "| {: <10} | {: <30} | {: <6} | {: <6} | {:8.1f} ms | {:6.1f} |\n", result.arch, model,
                result.backend, ggml_type_name(result.weight_type), result.time.mean, result.time.stdev));
        }
        printf("\n");
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
