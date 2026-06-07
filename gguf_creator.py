import os
MERGED_PATH = "/content/drive/MyDrive/SlayCal/SlayCal-Qwen2.5-VL-3B-Merged"
DRIVE_PATH = "/content/drive/MyDrive/SlayCal"
GGUF_OUTPUT = f"{DRIVE_PATH}/SlayCal-Qwen2.5-VL-3B-GGUF"

# Remove the wrong index file
os.remove(f"{MERGED_PATH}/model.safetensors.index.json")
print("Index file removed")

os.makedirs(GGUF_OUTPUT, exist_ok=True)

# Step 1: Convert to F16 GGUF
!python /content/llama_cpp/convert_hf_to_gguf.py \
    {MERGED_PATH} \
    --outfile {GGUF_OUTPUT}/SlayCal-3B-F16.gguf \
    --outtype f16

print("F16 GGUF done!")
