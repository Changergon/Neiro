import os
import json
import time
from googletrans import Translator

dataset_dir = r'C:\Users\Дмитрий\Desktop\rp2k_dataset\all'
mapping_file = 'mapping.json'

translator = Translator()

def safe_translate(text, retries=3, delay=2):
    """Безопасный перевод с повторной попыткой при ошибке"""
    for attempt in range(retries):
        try:
            result = translator.translate(text, src='zh-cn', dest='en')
            if result and result.text:
                return result.text
        except Exception as e:
            print(f"[{attempt + 1}/{retries}] Ошибка перевода '{text}': {e}")
        time.sleep(delay)
    return text  # если не удалось перевести, вернуть оригинал

def generate_mapping():
    mapping = {}

    for dirpath, dirnames, _ in os.walk(dataset_dir):
        for dirname in dirnames:
            if dirname.startswith('.'):
                continue
            if dirname in mapping:
                continue
            translated = safe_translate(dirname)
            mapping[dirname] = translated
            print(f"{dirname} → {translated}")

    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)
    print(f"\n✅ Файл перевода '{mapping_file}' сохранён.")

if __name__ == "__main__":
    generate_mapping()
