import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
from unsloth import FastVisionModel, UnslothVisionDataCollator, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
import glob
import json, os
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
import torch
torch._dynamo.config.suppress_errors = True


DRIVE_PATH = "D:/Projeler/SlayCal"
LOCAL_INGREDIENTS = "D:/Projeler/SlayCal/images/ingredients"
OUTPUT_DIR = "D:/Projeler/SlayCal/SlayCal-Recipe-LoRA"

print("Loading model...")
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-3B-Instruct",
    load_in_4bit=True,
)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print("Model loaded!")

with open(f"{DRIVE_PATH}/dataset_recipe.json", 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

def convert(entry):
    user_text = entry["messages"][0]["content"]
    assistant_text = entry["messages"][1]["content"]
    img_files = []
    for raw_path in entry["images"]:
        fixed = raw_path.replace("\\", "/")
        filename = fixed.split("/")[-1]
        img_path = None
        for root, dirs, files in os.walk(LOCAL_INGREDIENTS):
            if filename in files:
                img_path = os.path.join(root, filename)
                break
        if not img_path:
            return None
        img_files.append(img_path)
    if not img_files:
        return None
    clean_text = entry["messages"][0]["content"].replace("<image>\n", "").replace("<image>", "").strip()
    return {"image_paths": img_files, "user_text": clean_text, "assistant_text": entry["messages"][1]["content"]}

converted = [r for entry in raw_data if (r := convert(entry)) is not None]
print(f"Dataset: {len(converted)} / {len(raw_data)}")

class LazyRecipeDataset(TorchDataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        entry = self.data[idx]
        images = []
        for path in entry["image_paths"]:
            img = Image.open(path).convert("RGB")
            img.thumbnail((384, 384), Image.LANCZOS)
            images.append(img)
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": entry["user_text"]})
        return {
            "messages": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": entry["assistant_text"]}]}
            ]
        }

lazy_dataset = LazyRecipeDataset(converted)
print(f"Ready: {len(lazy_dataset)} samples")

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=lazy_dataset,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        num_train_epochs=2,
        learning_rate=5e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_steps=50,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=1024,
        dataloader_num_workers=0,
    ),
)

print(f"Total steps: ~{len(lazy_dataset) * 3 // 4}")
trainer_stats = trainer.train()
print(f"Training loss: {trainer_stats.training_loss:.4f}")

# === SAVE ===
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}")