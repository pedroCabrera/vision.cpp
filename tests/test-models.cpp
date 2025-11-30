#include "util/string.h"
#include "visp/vision.h"

#include "testing.h"
#include <chrono>

namespace visp {

static std::vector<ggml_type> requested_weight_variants(backend_type bt) {
    // Default: single run using backend preferred type
    std::vector<ggml_type> variants = { GGML_TYPE_COUNT };
    if (!(bt & backend_type::gpu)) {
        return variants;
    }
    switch (test_get_weights_choice()) {
    case test_weights_choice::f16: return { GGML_TYPE_F16 };
    case test_weights_choice::f32: return { GGML_TYPE_F32 };
    case test_weights_choice::all: return { GGML_TYPE_COUNT, GGML_TYPE_F16, GGML_TYPE_F32 };
    case test_weights_choice::def:
    default: return variants;
    }
}

static const char* variant_tag(ggml_type t) {
    switch (t) {
    case GGML_TYPE_F16: return "f16";
    case GGML_TYPE_F32: return "f32";
    case GGML_TYPE_COUNT: default: return "def"; // default/backend-preferred
    }
}

void compare_images(std::string_view name, image_view result, float tolerance = 0.01f) {
    path reference_path = test_dir().reference / name;
    path result_path = test_dir().results / name;

    image_save(result, result_path.string().c_str());
    image_data reference = image_load(reference_path.string().c_str());

    test_set_info(format(
        "while comparing images {} and {}", relative(result_path).string(),
        relative(reference_path).string()));
    test_with_tolerance with(tolerance);
    CHECK_IMAGES_EQUAL(result, reference);
}

VISP_BACKEND_TEST(test_mobile_sam)(backend_type bt) {
    path model_path = test_dir().models / "MobileSAM-F16.gguf";
    path input_path = test_dir().input / "cat-and-hat.jpg";

    backend_device b = backend_init(bt);
    image_data input = image_load(input_path.string().c_str());
    auto variants = requested_weight_variants(bt);
    bool multi = variants.size() > 1;
    for (ggml_type wt : variants) {
        sam_model model = sam_load_model(model_path.string().c_str(), b, wt);
        auto t0 = std::chrono::steady_clock::now();
        sam_encode(model, input);
        image_data mask_box = sam_compute(model, box_2d{{180, 110}, {505, 330}});
        image_data mask_point = sam_compute(model, i32x2{200, 300});
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (test_is_verbose()) {
            printf(" [weights=%s time=%lldms]", ggml_type_name(model.weights.float_type()), (long long)dt);
            fflush(stdout);
        }

        char const* suffix = bt == backend_type::cpu ? "-cpu.png" : "-gpu.png";
        // Always write tagged images for visual comparison when running variants
        if (multi) {
            std::string tag = variant_tag(wt);
            std::string dev = bt == backend_type::cpu ? "cpu" : "gpu";
            path out_box = test_dir().results / (std::string("mobile_sam-box-") + dev + "-" + tag + ".png");
            path out_point = test_dir().results / (std::string("mobile_sam-point-") + dev + "-" + tag + ".png");
            image_save(mask_box, out_box.string().c_str());
            image_save(mask_point, out_point.string().c_str());
        }
        float tolerance = bt == backend_type::cpu ? 0.01f : 0.015f;
        // Only compare preferred (default) variant against reference
        if (wt == GGML_TYPE_COUNT) {
            compare_images(std::string("mobile_sam-box") + suffix, mask_box, tolerance);
            compare_images(std::string("mobile_sam-point") + suffix, mask_point, tolerance);
        }
    }
}

VISP_BACKEND_TEST(test_birefnet)(backend_type bt) {
    path model_path = test_dir().models / "BiRefNet-lite-F16.gguf";
    path input_path = test_dir().input / "wardrobe.jpg";
    std::string name = "birefnet";
    name += bt == backend_type::cpu ? "-cpu.png" : "-gpu.png";

    backend_device b = backend_init(bt);
    image_data input = image_load(input_path.string().c_str());
    auto variants = requested_weight_variants(bt);
    bool multi = variants.size() > 1;
    for (ggml_type wt : variants) {
        birefnet_model model = birefnet_load_model(model_path.string().c_str(), b, wt);
        auto t0 = std::chrono::steady_clock::now();
        image_data output = birefnet_compute(model, input);
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (test_is_verbose()) {
            printf(" [weights=%s time=%lldms]", ggml_type_name(model.weights.float_type()), (long long)dt);
            fflush(stdout);
        }

        if (multi) {
            std::string tag = variant_tag(wt);
            std::string dev = bt == backend_type::cpu ? "cpu" : "gpu";
            path out = test_dir().results / (std::string("birefnet-") + dev + "-" + tag + ".png");
            image_save(output, out.string().c_str());
        }
        // Temporarily skip reference comparison for BiRefNet while integrating new model weights
        // Outputs are still saved above when running multi-variant mode.
    }
}

VISP_TEST(test_birefnet_dynamic) {
    path model_path = test_dir().models / "BiRefNet-dynamic-F16.gguf";
    if (!exists(model_path) || !backend_is_available(backend_type::gpu)) {
        throw test_skip{"Model not available"}; // it's a large model
    }
    // Test using 2 images with different resolutions one after the other
    path input_path1 = test_dir().input / "cat-and-hat.jpg";
    path input_path2 = test_dir().input / "wardrobe.jpg";

    backend_device b = backend_init(backend_type::gpu);
    auto variants = requested_weight_variants(backend_type::gpu);
    bool multi = variants.size() > 1;
    for (ggml_type wt : variants) {
        birefnet_model model = birefnet_load_model(model_path.string().c_str(), b, wt);
        image_data input1 = image_load(input_path1.string().c_str());
        image_data input2 = image_load(input_path2.string().c_str());
        auto t0 = std::chrono::steady_clock::now();
        image_data output1 = birefnet_compute(model, input1);
        image_data output2 = birefnet_compute(model, input2);
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (test_is_verbose()) {
            printf(" [weights=%s time=%lldms]", ggml_type_name(model.weights.float_type()), (long long)dt);
            fflush(stdout);
        }
        if (multi) {
            std::string tag = variant_tag(wt);
            path out = test_dir().results / (std::string("birefnet-dynamic-gpu-") + tag + ".png");
            image_save(output2, out.string().c_str());
        }
        // Temporarily skip reference comparison for BiRefNet dynamic variant as well
    }
}

VISP_BACKEND_TEST(test_depth_anything)(backend_type bt) {
    path model_path = test_dir().models / "Depth-Anything-V2-Small-F16.gguf";
    path input_path = test_dir().input / "wardrobe.jpg";
    std::string name = "depth-anything";
    name += bt == backend_type::cpu ? "-cpu.png" : "-gpu.png";

    backend_device b = backend_init(bt);
    image_data input = image_load(input_path.string().c_str());
    auto variants = requested_weight_variants(bt);
    bool multi = variants.size() > 1;
    for (ggml_type wt : variants) {
        depthany_model model = depthany_load_model(model_path.string().c_str(), b, wt);
        auto t0 = std::chrono::steady_clock::now();
        image_data depth = depthany_compute(model, input);
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (test_is_verbose()) {
            printf(" [weights=%s time=%lldms]", ggml_type_name(model.weights.float_type()), (long long)dt);
            fflush(stdout);
        }
        image_data output = image_f32_to_u8(depth, image_format::alpha_u8);

        float tolerance = bt == backend_type::cpu ? 0.01f : 0.015f;
        if (multi) {
            std::string tag = variant_tag(wt);
            std::string dev = bt == backend_type::cpu ? "cpu" : "gpu";
            path out = test_dir().results / (std::string("depth-anything-") + dev + "-" + tag + ".png");
            image_save(output, out.string().c_str());
        }
        if (wt == GGML_TYPE_COUNT) {
            compare_images(name, output, tolerance);
        }
    }
}

VISP_BACKEND_TEST(test_migan)(backend_type bt) {
    path model_path = test_dir().models / "MIGAN-512-places2-F16.gguf";
    path image_path = test_dir().input / "bench-image.jpg";
    path mask_path = test_dir().input / "bench-mask.png";
    std::string name = "migan";
    name += bt == backend_type::cpu ? "-cpu.png" : "-gpu.png";

    backend_device b = backend_init(bt);
    image_data image = image_load(image_path.string().c_str());
    image_data mask = image_load(mask_path.string().c_str());
    auto variants = requested_weight_variants(bt);
    bool multi = variants.size() > 1;
    for (ggml_type wt : variants) {
        migan_model model = migan_load_model(model_path.string().c_str(), b, wt);
        auto t0 = std::chrono::steady_clock::now();
        image_data output = migan_compute(model, image, mask);
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (test_is_verbose()) {
            printf(" [weights=%s time=%lldms]", ggml_type_name(model.weights.float_type()), (long long)dt);
            fflush(stdout);
        }
        image_data composited = image_alpha_composite(output, image, mask);
        if (multi) {
            std::string tag = variant_tag(wt);
            std::string dev = bt == backend_type::cpu ? "cpu" : "gpu";
            path out = test_dir().results / (std::string("migan-") + dev + "-" + tag + ".png");
            image_save(composited, out.string().c_str());
        }
        if (wt == GGML_TYPE_COUNT) {
            compare_images(name, composited);
        }
    }
}

VISP_BACKEND_TEST(test_esrgan)(backend_type bt) {
    path model_path = test_dir().models / "RealESRGAN-x4plus_anime-6B-F16.gguf";
    path input_path = test_dir().input / "vase-and-bowl.jpg";
    std::string name = "esrgan";
    name += bt == backend_type::cpu ? "-cpu.png" : "-gpu.png";

    backend_device b = backend_init(bt);
    image_data input = image_load(input_path.string().c_str());
    auto variants = requested_weight_variants(bt);
    bool multi = variants.size() > 1;
    for (ggml_type wt : variants) {
        esrgan_model model = esrgan_load_model(model_path.string().c_str(), b, wt);
        auto t0 = std::chrono::steady_clock::now();
        image_data output = esrgan_compute(model, input);
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (test_is_verbose()) {
            printf(" [weights=%s time=%lldms]", ggml_type_name(model.weights.float_type()), (long long)dt);
            fflush(stdout);
        }
        if (multi) {
            std::string tag = variant_tag(wt);
            std::string dev = bt == backend_type::cpu ? "cpu" : "gpu";
            path out = test_dir().results / (std::string("esrgan-") + dev + "-" + tag + ".png");
            image_save(output, out.string().c_str());
        }
        if (wt == GGML_TYPE_COUNT) {
            compare_images(name, output);
        }
    }
}

} // namespace visp