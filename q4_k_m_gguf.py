# Step 2: Quantize F16 to Q4_K_M
!cd /content/llama_cpp && make -j llama-quantize

!./content/llama_cpp/build/bin/llama-quantize \
    {GGUF_OUTPUT}/SlayCal-3B-F16.gguf \
    {GGUF_OUTPUT}/SlayCal-3B-Q4_K_M.gguf \
    Q4_K_M

print("Q4_K_M GGUF done!")
