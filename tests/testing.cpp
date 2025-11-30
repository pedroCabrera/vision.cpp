#include "testing.h"
#include "visp/ml.h"

#include <chrono>
#include <filesystem>
#include <string_view>

using std::chrono::steady_clock;

namespace visp {
// Globals
float tolerance = 1e-5f;
std::string extra_info;
static bool g_verbose = false;
bool test_is_verbose() { return g_verbose; }

// Tests CLI options
static test_weights_choice g_weights_choice = test_weights_choice::def;
test_weights_choice test_get_weights_choice() { return g_weights_choice; }
} // namespace visp

#ifndef VISP_TEST_NO_MAIN

int main(int argc, char** argv) {
    using namespace visp;

    ggml_backend_load_all();

    auto& registry = test_registry_instance();

    int passed = 0;
    int failed = 0;
    int errors = 0;
    int skipped = 0;

    std::string_view filter;
    bool exclude_gpu = false;
    bool only_gpu = false;
    bool verbose = false;
    std::string weights_flag; // "def" | "f16" | "f32" | "all"

    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "--no-gpu") {
            exclude_gpu = true;
        } else if (arg == "--only-gpu") {
            only_gpu = true;
        } else if (arg == "--weights") {
            weights_flag = std::string(argv[++i]);
        } else {
            filter = arg;
        }
    }

    // Apply weights flag if provided
    if (!weights_flag.empty()) {
        using enum visp::test_weights_choice;
        if (weights_flag == "f16") visp::g_weights_choice = f16;
        else if (weights_flag == "f32") visp::g_weights_choice = f32;
        else if (weights_flag == "all") visp::g_weights_choice = all;
        else visp::g_weights_choice = def;
    }

    auto run = [&](test_case const& test, char const* name, backend_type backend) {
        auto t0 = steady_clock::now();
        try {
            if (!filter.empty() && name != filter && test.name != filter) {
                return; // test not selected
            }
            if (verbose) {
                printf("%s", name);
                fflush(stdout);
            }

            if (test.is_backend_test) {
                test.backend_func(backend);
            } else {
                test.func();
            }

            auto t1 = steady_clock::now();
            long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            ++passed;
            if (verbose) {
                printf(" %s (%lldms)\n", "\033[32mPASSED\033[0m", ms);
            }
        } catch (const visp::test_failure& e) {
            auto t1 = steady_clock::now();
            long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            ++failed;
            printf("%s %s (%lldms)\n", verbose ? "" : name, "\033[31mFAILED\033[0m", ms);
            printf("  \033[90m%s:%d:\033[0m Assertion failed\n", e.file, e.line);
            printf("  \033[93m%s\033[0m\n", e.condition);
            if (e.eval) {
                printf("  \033[93m%s\033[0m\n", e.eval.c_str());
            }
            if (!visp::extra_info.empty()) {
                printf("  %s\n", visp::extra_info.c_str());
            }
        } catch (const visp::test_skip&) {
            auto t1 = steady_clock::now();
            long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            ++skipped;
            if (verbose) {
                printf(" %s (%lldms)\n", "\033[33mSKIPPED\033[0m", ms);
            }
        } catch (const std::exception& e) {
            auto t1 = steady_clock::now();
            long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            ++errors;
            printf("%s %s (%lldms)\n", verbose ? "" : name, "\033[31mERROR\033[0m", ms);
            printf("  \033[90m%s:%d:\033[0m Unhandled exception\n", test.file, test.line);
            printf("  \033[93m%s\033[0m\n", e.what());
        } catch (...) {
            auto t1 = steady_clock::now();
            long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            ++errors;
            printf("%s %s (%lldms)\n", verbose ? "" : name, "\033[31mERROR\033[0m", ms);
            printf("  \033[90m%s:%d:\033[0m Unhandled exception\n", test.file, test.line);
        }
        visp::extra_info.clear();
    };

    auto time_start = steady_clock::now();
    fixed_string<128> name;

    visp::g_verbose = verbose;

    for (auto& test : registry.tests) {
        if (test.is_backend_test) {
            if (!only_gpu) {
                run(test, format(name, "{}[cpu]", test.name), backend_type::cpu);
            }
            if (!exclude_gpu) {
                run(test, format(name, "{}[gpu]", test.name), backend_type::gpu);
            }
        } else {
            run(test, test.name, backend_type::cpu);
        }
    }

    auto time_end = steady_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();

    char const* color = (failed > 0 || errors > 0) ? "\033[31m" : "\033[32m";
    if (verbose || failed > 0 || errors > 0) {
        printf("%s----------------------------------------------------------------------\n", color);
    }
    if (failed > 0) {
        printf("\033[31m%d failed, ", failed);
    }
    if (errors > 0) {
        printf("\033[31m%d errors, ", errors);
    }
    if (skipped > 0) {
        printf("\033[33m%d skipped, ", skipped);
    }
    printf("\033[92m%d passed %sin %lldms\033[0m\n", passed, color, (long long)duration);

    return (failed > 0 || errors > 0) ? 1 : 0;
}

#endif // VISP_TEST_NO_MAIN

namespace visp {

test_registry& test_registry_instance() {
    static test_registry registry;
    return registry;
}

test_registration::test_registration(
    char const* name, test_function f, char const* file, int line) {
    test_case t;
    t.name = name;
    t.file = file;
    t.line = line;
    t.func = f;
    t.is_backend_test = false;
    test_registry_instance().tests.push_back(t);
}

test_registration::test_registration(
    char const* name, test_backend_function f, char const* file, int line) {
    test_case t;
    t.name = name;
    t.file = file;
    t.line = line;
    t.backend_func = f;
    t.is_backend_test = true;
    test_registry_instance().tests.push_back(t);
}

test_directories const& test_dir() {
    static test_directories const result = []() {
        path cur = std::filesystem::current_path();
        while (!exists(cur / "README.md")) {
            cur = cur.parent_path();
            if (cur.empty()) {
                throw std::runtime_error("root directory not found");
            }
        }
        test_directories dirs{
            .root = cur,
            .models = cur / "models",
            .test = cur / "tests",
            .input = cur / "tests" / "input",
            .results = cur / "tests" / "results",
            .reference = cur / "tests" / "reference"};
        if (!exists(dirs.results)) {
            create_directories(dirs.results);
        }
        return dirs;
    }();
    return result;
}

void test_set_info(std::string_view info) {
    extra_info = info;
}

float& test_tolerance_value() {
    return tolerance;
}

test_failure test_failure_image_mismatch(
    char const* file, int line, char const* condition, float rms) {
    test_failure result(file, line, condition);
    format(result.eval, "-> rmse {:.5f} > {:.5f} tolerance", rms, test_tolerance_value());
    return result;
}

} // namespace visp
