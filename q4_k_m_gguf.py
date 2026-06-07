# Build llama-quantize with CMake
!cd /content/llama_cpp && mkdir -p build && cd build && cmake .. && cmake --build . --target llama-quantize -j

# Quantize
GGUF_OUTPUT = "/content/drive/MyDrive/SlayCal/SlayCal-Qwen2.5-VL-3B-GGUF"

!/content/llama_cpp/build/bin/llama-quantize \
    {GGUF_OUTPUT}/SlayCal-3B-F16.gguf \
    {GGUF_OUTPUT}/SlayCal-3B-Q4_K_M.gguf \
    Q4_K_M

print("Q4_K_M GGUF done!")
