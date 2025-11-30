@echo off
setlocal ENABLEDELAYEDEXPANSION

REM benchmark.bat - run vision-bench for Vulkan and CUDA builds with forwarded flags
REM Usage:
REM   benchmark.bat [args]
REM Examples:
REM   benchmark.bat --trace --iters 8 --no-gpu-mul --weights f32
REM   benchmark.bat --include-transfer --cuda-graphs on --cuda-allow-f16-interp off

set SCRIPT_DIR=%~dp0
set VULKAN_EXE="%SCRIPT_DIR%build_VULKAN\bin\Release\vision-bench.exe"
set CUDA_EXE="%SCRIPT_DIR%build_CUDA\bin\Release\vision-bench.exe"

if exist %VULKAN_EXE% (
    echo [VULKAN] Running: %VULKAN_EXE% -b vulkan %*
    %VULKAN_EXE% -b vulkan %*
) else (
    echo [VULKAN] Skipping: %VULKAN_EXE% not found. Build with build_VULKAN.bat first.
)

echo.

if exist %CUDA_EXE% (
    echo [CUDA]    Running: %CUDA_EXE% -b cuda %*
    %CUDA_EXE% -b cuda %*
) else (
    echo [CUDA]    Skipping: %CUDA_EXE% not found. Build with build_CUDA.bat first.
)

echo.
echo Done.
endlocal
