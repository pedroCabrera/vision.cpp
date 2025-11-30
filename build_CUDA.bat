cmake . -B build_CUDA -D VISP_DEV=ON -D VISP_CUDA=ON -D CMAKE_BUILD_TYPE=Release
cmake --build build_CUDA --config Release