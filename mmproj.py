import os
GGUF_OUTPUT = "/content/drive/MyDrive/SlayCal/SlayCal-Qwen2.5-VL-3B-GGUF"

# Remove broken mmproj
os.remove(f"{GGUF_OUTPUT}/SlayCal-3B-mmproj.gguf")

# Regenerate from original Qwen2.5-VL-3B (not merged, because mmproj = vision encoder, unchanged)
!python /content/llama_cpp/convert_hf_to_gguf.py \
    Qwen/Qwen2.5-VL-3B-Instruct \
    --outfile {GGUF_OUTPUT}/SlayCal-3B-mmproj.gguf \
    --outtype f16 \
    --mmproj \
    --remote

print("mmproj done!")
