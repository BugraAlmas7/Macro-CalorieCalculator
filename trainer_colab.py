import shutil, os

IMAGE_BASE = "/content/macro_images"
DRIVE_IMAGES = "/content/drive/MyDrive/SlayCal/images/macro"

if not os.path.exists(IMAGE_BASE):
    print("Görseller yerel diske kopyalanıyor...")
    shutil.copytree(DRIVE_IMAGES, IMAGE_BASE)
    print(f"Kopyalandı: {len(os.listdir(IMAGE_BASE))} dosya")
else:
    print(f"Zaten mevcut: {len(os.listdir(IMAGE_BASE))} dosya")

import json, os
from PIL import Image
from datasets import Dataset

DRIVE_PATH = "/content/drive/MyDrive/SlayCal"
IMAGE_BASE = "/content/macro_images"

with open(f"{DRIVE_PATH}/dataset_macro.json", 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

def convert(entry):
    user_text = entry["messages"][0]["content"].replace("<image>\n", "").strip()
    assistant_text = entry["messages"][1]["content"]
    img_filename = entry["images"][0].replace("\\", "/").split("/")[-1]
    img_path = os.path.join(IMAGE_BASE, img_filename)
    try:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail((384, 384), Image.LANCZOS)
    except:
        return None
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": assistant_text}
            ]}
        ]
    }

print("Dataset dönüştürülüyor...")
converted = [r for entry in raw_data if (r := convert(entry)) is not None]
print(f"Başarılı: {len(converted)}")

dataset = Dataset.from_list(converted)
print(f"Dataset boyutu: {len(dataset)}")

from unsloth import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported, FastVisionModel

DRIVE_PATH = "/content/drive/MyDrive/SlayCal"

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 20,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        save_steps = 200,
        save_total_limit = 3,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = f"{DRIVE_PATH}/checkpoints_qwen",
        report_to = "none",
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_seq_length = 1024,
    ),
    data_collator = UnslothVisionDataCollator(model, tokenizer),
)

trainer_stats = trainer.train()
print(f"Training loss: {trainer_stats.training_loss:.4f}")
