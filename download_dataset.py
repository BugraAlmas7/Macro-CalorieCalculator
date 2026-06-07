from datasets import load_dataset
import os

IMAGE_DIR = "D:/Projeler/SlayCal/images/macro"
os.makedirs(IMAGE_DIR, exist_ok=True)

print("[INFO] Sağlıklı yemek (Nutrition5k) görselleri indiriliyor...")
dataset = load_dataset("mmathys/food-nutrients", split="test")

print(f"[INFO] Toplam {len(dataset)} görsel diske yazılıyor...")
for i, item in enumerate(dataset):
    try:
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(os.path.join(IMAGE_DIR, f"food_{i}.jpg"))
        
        if (i+1) % 500 == 0:
            print(f"[INFO] {i+1} görsel kaydedildi...")
    except Exception as e:
        pass

print(f"\n[SUCCESS] Tüm resimler {IMAGE_DIR} klasörüne çıkartıldı!")