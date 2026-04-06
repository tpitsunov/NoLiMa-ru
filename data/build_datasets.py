import os
import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import argparse

def build_haystack(tokenizer_name, output_file, chunk_size=250, max_books=30):
    print(f"[haystack] Загружаем токенизатор {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    print("\n[haystack] Загружаем корпус BooksSummarizationRU (HuggingFace)...")
    try:
        ds = load_dataset("slon-hk/BooksSummarizationRU", split="train")
    except Exception as e:
        print(f"Ошибка загрузки датасета: {e}")
        return
        
    texts = ds['Full Text']
    books_to_process = list(texts)
    
    # Сидируем шаффл для воспроизводимости сена
    random.seed(42)
    random.shuffle(books_to_process)
    
    print(f"[haystack] Токенизация и нарезка (чанками по {chunk_size} токенов)...")
    
    haystack_cache = []
    
    for text in tqdm(books_to_process[:max_books], desc="Обработка текстов"):
        if not text: continue
        tokens = tokenizer.encode(text)
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i+chunk_size]
            if len(chunk) >= chunk_size // 2: # отбрасываем огрызки
                haystack_cache.append(chunk)
                
    print(f"[haystack] Сформировано {len(haystack_cache)} уникальных чанков.")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(haystack_cache, f)
        
    print(f"✅ Кэш сена сохранен в {output_file}")

def validate_needles(needles_file):
    print(f"\n[needles] Валидация датасета {needles_file}...")
    with open(needles_file, 'r', encoding='utf-8') as f:
        needles = json.load(f)
        
    # Простейшая эвристика на лексическое совпадение
    # В идеале должны совпадать 0 слов (кроме стоп-слов и имени {CHAR})
    stop_words = {"в", "на", "с", "и", "а", "но", "как", "по", "к", "из", "кто", "это", "что"}
    
    warnings = 0
    for idx, pair in enumerate(needles):
        needle_words = set([w.lower() for w in pair['needle'].split() if len(w) > 2]) - stop_words
        q_words = set([w.lower() for w in pair['question'].split() if len(w) > 2]) - stop_words
        
        overlap = needle_words.intersection(q_words)
        # Игнорируем совпадения из-за {CHAR} и служебных слов
        overlap = {w for w in overlap if "{char}" not in w and "внимание" not in w and "вопрос" not in w and "логический" not in w and "персонажей" not in w}
        
        if len(overlap) > 0:
            print(f"⚠️ Падение: пара #{idx} имеет лексическое совпадение: {overlap}")
            warnings += 1
            
    print(f"✅ Валидация завершена. Найдено {warnings} потенциальных лексических совпадений из {len(needles)} пар.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="evilfreelancer/ruGPT3XL-8k", help="HF Tokenizer name")
    parser.add_argument("--haystack_out", type=str, default="data/haystack/haystack_cache.json")
    parser.add_argument("--needles_in", type=str, default="data/needles_ru.json")
    args = parser.parse_args()
    
    validate_needles(args.needles_in)
    build_haystack(args.tokenizer, args.haystack_out)
