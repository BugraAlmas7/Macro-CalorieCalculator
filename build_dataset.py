import os
import json
import time
import random
import re
import sys
from PIL import Image
from google import genai
from google.genai import types

# --- HER WORKER İÇİN API KEY ---
API_KEYS = {
    1: "",
    2: "",
    3: "",
    4: "",
}

# --- CONFIGURATION ---
IMAGE_DIR = "D:/Projeler/SlayCal/images/macro"
MAIN_OUTPUT = "D:/Projeler/SlayCal/dataset_macro.json"
BATCH_SIZE = 10
MAX_RETRIES = 3
NUM_WORKERS = len(API_KEYS) 

MODELS_TO_TRY = [
    'gemini-3.5-flash',
    'gemini-3-flash-preview',
    'gemini-3.1-flash-lite',
    'gemini-2.5-flash',
    'gemini-2.5-flash-lite',
]

LANG_CONFIGS = {
    "tr": {"prompt": "Bu yemeği analiz et ve makrolarını JSON olarak ver.", "keys": '{"yemek_adi": "...", "kalori": 0, "protein": 0, "karbonhidrat": 0, "yag": 0}'},
    "en": {"prompt": "Analyze this meal and provide its macros in JSON.", "keys": '{"meal_name": "...", "calories": 0, "protein": 0, "carbs": 0, "fat": 0}'},
    "de": {"prompt": "Analysieren Sie diese Mahlzeit und geben Sie ihre Makros in JSON an.", "keys": '{"mahlzeit_name": "...", "kalorien": 0, "protein": 0, "kohlenhydrate": 0, "fett": 0}'},
    "fr": {"prompt": "Analysez ce repas et donnez ses macros en JSON.", "keys": '{"nom_repas": "...", "calories": 0, "proteines": 0, "glucides": 0, "lipides": 0}'},
    "es": {"prompt": "Analiza esta comida y proporciona sus macros en JSON.", "keys": '{"nombre_comida": "...", "calorias": 0, "proteinas": 0, "carbohidratos": 0, "grasas": 0}'},
    "it": {"prompt": "Analizza questo pasto e fornisci i suoi macro in JSON.", "keys": '{"nome_pasto": "...", "calorie": 0, "proteine": 0, "carboidrati": 0, "grassi": 0}'},
    "zh": {"prompt": "分析这顿饭并以JSON格式提供其宏量营养素。", "keys": '{"菜名": "...", "卡路里": 0, "蛋白质": 0, "碳水化合物": 0, "脂肪": 0}'},
    "ja": {"prompt": "この食事を分析し、マクロ栄養素をJSONで提供してください。", "keys": '{"食事名": "...", "カロリー": 0, "タンパク質": 0, "炭水化物": 0, "脂肪": 0}'},
    "ko": {"prompt": "이 식사를 분석하고 매크로를 JSON으로 제공하십시오.", "keys": '{"식사_이름": "...", "칼로리": 0, "단백질": 0, "탄수화물": 0, "지방": 0}'}
}

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_all_files():
    """Tüm görselleri sıralı getir"""
    all_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_files.sort(key=lambda f: int(re.search(r'\d+', f).group()) if re.search(r'\d+', f) else 0)
    return all_files

def get_already_processed():
    """Tüm worker dosyalarından + ana dosyadan işlenmiş dosyaları topla"""
    processed = set()
    
    # Ana dosya
    if os.path.exists(MAIN_OUTPUT):
        try:
            with open(MAIN_OUTPUT, 'r', encoding='utf-8') as f:
                for entry in json.load(f):
                    processed.add(os.path.basename(entry["images"][0]))
        except:
            pass
    
    # Worker dosyaları
    for w in range(1, NUM_WORKERS + 1):
        wfile = MAIN_OUTPUT.replace(".json", f"_worker{w}.json")
        if os.path.exists(wfile):
            try:
                with open(wfile, 'r', encoding='utf-8') as f:
                    for entry in json.load(f):
                        processed.add(os.path.basename(entry["images"][0]))
            except:
                pass
    
    return processed

def process_batch(client, batch_files):
    contents = []
    batch_meta = {}
    
    system_instruction = (
        "You are an expert Dietitian and AI Vision model. Analyze the attached top-down food plates.\n"
        "Identify the food, visually estimate the portion size, and calculate highly realistic macros (Calories, Protein, Carbs, Fat) based on your estimation.\n"
        "You must return a single JSON array containing objects for each image.\n"
        "Each object in the array MUST follow this exact structure:\n"
        "{\n"
        "  \"filename\": \"example.jpg\",\n"
        "  \"analysis\": { ... localized keys ... }\n"
        "}\n"
        "CRITICAL: Do NOT use markdown code blocks like ```json. Return pure JSON string.\n"
        "Here are the specific languages and JSON structures required for each file:\n"
    )
    
    contents.append(system_instruction)
    
    for idx, filename in enumerate(batch_files):
        img_path = os.path.join(IMAGE_DIR, filename)
        try:
            img = Image.open(img_path)
            contents.append(f"Image {idx+1}:")
            contents.append(img)
            
            lang = random.choice(list(LANG_CONFIGS.keys()))
            cfg = LANG_CONFIGS[lang]
            contents.append(f"- Image {idx+1} ({filename}): Language '{lang}'. Use exactly these JSON keys: {cfg['keys']}\n")
            batch_meta[filename] = {"prompt": cfg["prompt"], "path": img_path}
        except Exception as e:
            print(f"  [ERROR] Failed to open {filename}: {e}")
            
    for model_name in MODELS_TO_TRY:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2 
                )
            )
            print(f"  [OK] Model: {model_name}")
            return json.loads(response.text), batch_meta
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                print(f"  [429] {model_name} — rate limit")
            elif "503" in error_str:
                print(f"  [503] {model_name} — server busy")
            else:
                print(f"  [ERR] {model_name}: {error_str[:80]}...")
            continue
            
    return None, batch_meta

def run_worker(worker_id):
    """Belirli bir worker'ı çalıştır"""
    if worker_id not in API_KEYS:
        print(f"[ERROR] Worker {worker_id} için API key tanımlanmamış!")
        return
    
    api_key = API_KEYS[worker_id]
    if "BURAYA" in api_key:
        print(f"[ERROR] Worker {worker_id} API key'ini değiştirmeyi unuttun!")
        return
    
    client = genai.Client(api_key=api_key)
    output_file = MAIN_OUTPUT.replace(".json", f"_worker{worker_id}.json")
    
    # Tüm dosyaları al ve işlenmişleri çıkar
    all_files = get_all_files()
    processed = get_already_processed()
    remaining = [f for f in all_files if f not in processed]
    
    # Bu worker'ın payını hesapla
    files_per_worker = len(remaining) // NUM_WORKERS
    start_idx = (worker_id - 1) * files_per_worker
    end_idx = start_idx + files_per_worker if worker_id < NUM_WORKERS else len(remaining)
    my_files = remaining[start_idx:end_idx]
    
    print(f"{'='*60}")
    print(f"  WORKER {worker_id} / {NUM_WORKERS}")
    print(f"  Toplam kalan: {len(remaining)} | Benim payım: {len(my_files)}")
    print(f"  Aralık: {my_files[0] if my_files else 'YOK'} → {my_files[-1] if my_files else 'YOK'}")
    print(f"  Çıktı: {output_file}")
    print(f"{'='*60}\n")
    
    # Mevcut worker verisini yükle
    existing = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except:
            pass
    
    dataset = existing
    failed_batches = []
    
    for batch_idx, batch in enumerate(chunk_list(my_files, BATCH_SIZE)):
        retry_count = 0
        success = False
        
        while retry_count < MAX_RETRIES:
            wait_time = 20 if retry_count == 0 else 60 * (2 ** (retry_count - 1))
            
            print(f"[W{worker_id}] Batch {batch_idx+1} | Try {retry_count+1}/{MAX_RETRIES} | Wait {wait_time}s | {batch[0]}..{batch[-1]}")
            time.sleep(wait_time)
            
            result_json, meta = process_batch(client, batch)
            
            if result_json:
                for item in result_json:
                    fname = item.get("filename")
                    analysis = item.get("analysis")
                    if fname in meta and analysis:
                        dataset.append({
                            "messages": [
                                {"role": "user", "content": f"<image>\n{meta[fname]['prompt']}"},
                                {"role": "assistant", "content": json.dumps(analysis, ensure_ascii=False)}
                            ],
                            "images": [meta[fname]["path"]]
                        })
                        print(f"  [+] {fname} | Total: {len(dataset)}")
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                success = True
                break
            
            retry_count += 1
        
        if not success:
            print(f"  [SKIP] Batch failed: {batch}")
            failed_batches.append(batch)
    
    print(f"\n{'='*60}")
    print(f"  WORKER {worker_id} BİTTİ!")
    print(f"  Kayıt: {len(dataset)} | Başarısız: {len(failed_batches)} batch")
    print(f"  Dosya: {output_file}")
    print(f"{'='*60}")

def merge_all():
    """Tüm worker dosyalarını + ana dosyayı birleştir"""
    merged = []
    seen_images = set()
    
    # Önce ana dosyayı oku
    if os.path.exists(MAIN_OUTPUT):
        try:
            with open(MAIN_OUTPUT, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entry in data:
                    img = os.path.basename(entry["images"][0])
                    if img not in seen_images:
                        merged.append(entry)
                        seen_images.add(img)
        except:
            pass
    
    # Worker dosyalarını ekle
    for w in range(1, NUM_WORKERS + 1):
        wfile = MAIN_OUTPUT.replace(".json", f"_worker{w}.json")
        if os.path.exists(wfile):
            try:
                with open(wfile, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    count = 0
                    for entry in data:
                        img = os.path.basename(entry["images"][0])
                        if img not in seen_images:
                            merged.append(entry)
                            seen_images.add(img)
                            count += 1
                    print(f"[MERGE] Worker {w}: +{count} yeni kayıt ({wfile})")
            except Exception as e:
                print(f"[MERGE] Worker {w} okunamadı: {e}")
    
    # Kaydet
    with open(MAIN_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    print(f"\n[DONE] Toplam: {len(merged)} kayıt → {MAIN_OUTPUT}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım:")
        print("  python build_dataset_parallel.py 1      # Worker 1 çalıştır")
        print("  python build_dataset_parallel.py 2      # Worker 2 çalıştır")
        print("  python build_dataset_parallel.py merge   # Tüm sonuçları birleştir")
        sys.exit(1)
    
    arg = sys.argv[1]
    
    if arg == "merge":
        merge_all()
    else:
        try:
            worker_id = int(arg)
            run_worker(worker_id)
        except ValueError:
            print(f"[ERROR] Geçersiz argüman: {arg}")
