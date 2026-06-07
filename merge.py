# Manual merge: Load base model + LoRA, merge, save
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

DRIVE_PATH = "/content/drive/MyDrive/SlayCal"
LORA_PATH = f"{DRIVE_PATH}/SlayCal-Qwen2.5-VL-3B-LoRA"
MERGED_PATH = f"{DRIVE_PATH}/SlayCal-Qwen2.5-VL-3B-Merged"

print("Loading base model...")
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="cpu",
)

print("Loading LoRA adapter...")
model_with_lora = PeftModel.from_pretrained(base_model, LORA_PATH)

print("Merging...")
merged_model = model_with_lora.merge_and_unload()

print("Saving merged model...")
merged_model.save_pretrained(MERGED_PATH)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
processor.save_pretrained(MERGED_PATH)

print(f"Merged model saved to: {MERGED_PATH}")
