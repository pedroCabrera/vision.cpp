echo off
echo "______Testing_VULKAN______"
.\build_VULKAN\bin\Release\test-models.exe -v %*
echo "______Testing_CUDA______"
.\build_CUDA\bin\Release\test-models.exe -v %*