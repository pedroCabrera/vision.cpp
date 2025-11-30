# vision.cpp Developer Guide

## Project Overview

vision.cpp is a C++ library for computer vision ML inference built on [ggml](https://github.com/ggml-org/ggml) (similar to llama.cpp). It provides efficient inference on consumer hardware (CPU, NVIDIA, AMD, Intel GPUs) for models like MobileSAM, BiRefNet, Depth-Anything, MI-GAN, and ESRGAN.

### Architecture

**Three-layer API design** (in `include/visp/`):
- `image.h` - Image data structures and processing (loading, saving, conversion)
- `ml.h` - ML infrastructure wrapping ggml (backend devices, model loading, graph execution)
- `vision.h` - High-level model APIs (e.g., `sam_compute()`, `birefnet_compute()`)

**Core components:**
- `src/visp/` - Main library implementation
  - `image.cpp`, `ml.cpp`, `nn.cpp` - Core infrastructure
  - `arch/` - Model implementations (birefnet, mobile-sam, depth-anything, esrgan, migan, swin, dino)
- `src/cli/cli.cpp` - Command-line interface
- `depend/ggml/` - Fork of ggml tensor library (submodule)
- `scripts/convert.py` - PyTorch → GGUF weight conversion
- `tests/` - C++ and Python tests, reference image validation

## Build & Test Workflow

**Windows builds** (separate backends):
```powershell
# Build both backends
.\build.bat  # Runs build_CUDA.bat and build_VULKAN.bat

# Or build individually:
.\build_CUDA.bat      # cmake -B build_CUDA -D VISP_CUDA=ON -D VISP_DEV=ON
.\build_VULKAN.bat    # cmake -B build_VULKAN -D VISP_VULKAN=ON -D VISP_DEV=ON

# Run tests
.\test.bat            # Runs ctest in both build directories
```

**CMake options:**
- `VISP_DEV=ON` - Enable debug symbols + asserts that break into debugger (recommended for development)
- `VISP_CI=ON` - CI mode (asserts throw exceptions instead of breaking)
- `VISP_CUDA=ON` / `VISP_VULKAN=ON` - Backend selection
- `VISP_TESTS=ON` - Build tests (default when top-level project)

**Test with Python:**
```powershell
# Setup Python env (uv recommended)
uv sync

# Run Python tests
pytest tests/
```

## Coding Conventions

**Critical rules from CONTRIBUTING.md:**

1. **Plain structs & functions** - Avoid classes, hierarchies, template metaprogramming
2. **snake_case everywhere** except macros and template type params
3. **Naming convention:** `<group>_<verb>_*` - e.g., `image_load()`, `sam_encode()`, `tensor_copy()`
4. **Use ASSERT liberally** for invariants - configured by build type:
   - `VISP_DEV`: breaks into debugger
   - `VISP_CI`: throws exception
   - Release: disabled
5. **Throw `visp::exception`** only for user-recoverable exceptional errors
6. **RAII for resources** - use `std::unique_ptr` or small wrapper types, avoid manual free/delete
7. **Avoid heap allocations** - prefer `std::array`, `fixed_string<N>`
8. **Free functions preferred** over member functions (except small property-like methods)

**Example pattern:**
```cpp
tensor some_module(model_ref m, tensor x, ...) {
    // model_ref provides graph building + weight access by name
    // Functions compose to build complete models
    return result;
}
```

## ggml Integration

**Key differences from PyTorch:**
- **Dimension order reversed:** ggml uses `[rows, cols, channels, batch]` (most→least contiguous), PyTorch uses `[batch, channels, height, width]`
- **Permute semantics differ** - cannot copy 1:1 from PyTorch
- **C++ uses ggml convention**, Python files use PyTorch convention

**Layout conventions:**
- `WHCN` vs `CWHN` - configurable via `model_build_flag::cwhn`
- Helper functions: `permute_cwhn_to_whcn()`, `contiguous_2d_to_whcn()`, `nelements_whcn()`
- "Contiguous 2D" = preferred layout for 2D ops based on backend flags

**model_ref abstraction** (`ml.h`):
- Replaces `ggml_context*` in function signatures
- Tracks parent modules for weight name resolution
- Builds compute graphs while accessing weights by hierarchical names

## Model Implementation Workflow

Follow `docs/model-implementation-guide.md` for implementing new architectures:

1. **Analyze** PyTorch model (print modules, inspect state_dict, identify custom ops)
2. **Convert weights** - Add function to `scripts/convert.py` (PyTorch → GGUF)
3. **Implement compute graph incrementally:**
   - Copy PyTorch module to Python test file
   - Implement forward function in C++ (in `src/visp/arch/`)
   - Expose via `tests/workbench.cpp` for Python testing
   - Run reference vs C++ on dummy data, compare outputs
   - Repeat for each module/layer
4. **Add pre/post-processing** in C++
5. **Integrate into CLI** (`src/cli/cli.cpp`)
6. **Add high-level API** (`include/visp/vision.h`, `src/visp/vision.cpp`)
7. **Add to test suite** (`tests/test-models.cpp`, Python tests)

**Testing infrastructure:**
- `tests/workbench.py` - FFI bridge for comparing C++ vs PyTorch
- `tests/test_primitives.py` - Unit tests for nn building blocks
- `tests/reference/` - Reference images for regression testing
- Use `workbench.invoke_test()` to call C++ functions from Python with torch tensors

## Dependencies & Subdirectories

- `depend/ggml/` - ggml submodule (configured via GGML_VULKAN, GGML_CUDA in CMakeLists.txt)
- `depend/stb/` - STB image loading (stb_image.h)
- `depend/fmt/` - Optional formatting library (use `VISP_FMT_LIB=ON` instead of C++20 `<format>`)
- `models/` - Downloaded GGUF model files (not in repo)
- Python deps in `pyproject.toml` - torch, timm, gguf, spandrel, opencv, pytest

## File Organization Patterns

**Headers expose minimal API:**
- Public: `include/visp/*.h`
- Private: `src/visp/*.h` (e.g., `nn.h` for internal building blocks)

**Model architecture implementations:**
- Header: `src/visp/arch/<model>.h` (params, functions)
- Source: `src/visp/arch/<model>.cpp` (compute graph implementation)

**Tests mirror structure:**
- C++: `tests/test-<component>.cpp`
- Python: `tests/test_<model>.py`

## Key Gotchas

1. **Batch norm must be fused** - ASSERT fails if `running_mean`/`running_var` exist in weights
2. **Max 4D tensors in ggml** - models with >4D need reshaping
3. **Windows batch files call both backends** - `build.bat` builds CUDA + Vulkan, tests run on both
4. **Python env uses uv** - Script shebangs include inline dependencies: `#!/usr/bin/env -S uv run --script`
5. **Memory layouts matter** - Check `is_whcn(m)` vs `is_cwhn(m)` before 2D ops
6. **ASSERT behavior changes** - Set `VISP_DEV=ON` for breakpoints vs `VISP_CI=ON` for exceptions
7. **CUDA backend currently unsupported** - CUDA requires stricter tensor stride alignment than Vulkan. The issue manifests in `ggml-cuda/binbcast.cu:249` where binary broadcast operations (add, mul, etc.) fail with misaligned strides. This is a known limitation - ggml supports CUDA, but vision.cpp's tensor operations create stride patterns that Vulkan tolerates but CUDA rejects. Investigation needed: tensor views, reshapes, permutes that create non-element-aligned strides before binary ops.
