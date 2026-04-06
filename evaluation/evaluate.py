import os
import json
import argparse
import random
import torch
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_mixed_haystack(haystack_cache, num_tokens):
    random.shuffle(haystack_cache)
    result = []
    for chunk in haystack_cache:
        result.extend(chunk)
        if len(result) >= num_tokens:
            break
    return result[:num_tokens]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--run_config", required=True)
    args = parser.parse_args()

    model_config = load_json(args.model_config)
    run_config = load_json(args.run_config)

    # Устройство
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"🖥 Initializing device: {device.upper()}")

    # Загрузка
    model_name = model_config["model_name"]
    tokenizer_name = model_config["tokenizer_name"]
    dtype_str = model_config.get("dtype", "bfloat16")
    
    # MPS не всегда хорошо кушает bfloat16 в HF, поэтому под MPS часто используется float16
    dtype = torch.float16 if device == "mps" and dtype_str == "bfloat16" else (torch.bfloat16 if dtype_str == "bfloat16" else torch.float32)

    print(f"🤖 Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=model_config.get("trust_remote_code", True))
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=model_config.get("trust_remote_code", True)
    ).to(device)
    model.eval()

    needles = load_json(run_config["needles_path"])
    haystack_cache = load_json(run_config["haystack_cache_path"])

    # База имен для заполнения {CHAR}
    CHARACTERS = [
        "Иннокентий Боголюбов", "Авдотья Смирнова", "Евлампий Расторгуев", "Агриппина Меньшова", "Ипполит Воронцов", 
        "Прасковья Туманова", "Лукьян Державин", "Глафира Вяземская", "Еремей Романов", "Степанида Морозова",
        "Фома Апраксин", "Марфа Голицына", "Демьян Строганов", "Ульяна Шереметева", "Архип Демидов",
        "Пелагея Юсупова", "Макар Трубецкой", "Феврония Багратион", "Савелий Оболенский", "Евдокия Волконская"
    ]

    contexts = run_config["context_lengths"].get(model_name, [])
    if not contexts:
        contexts = run_config.get("context_lengths", []) if isinstance(run_config["context_lengths"], list) else []

    depth_nodes = run_config.get("depth_nodes", 10)
    depth_percentages = np.linspace(0, 1.0, depth_nodes)

    os.makedirs(run_config["results_dir"], exist_ok=True)
    
    results_list = []
    
    print("\n🚀 Starting NoLiMa benchmark...")
    for total_ctx in contexts:
        print(f"\n📏 Context: {total_ctx} tokens")
        total_tests = len(depth_percentages) * len(needles)
        pbar = tqdm(total=total_tests, desc=f"Ctx {total_ctx}")

        for depth_pct in depth_percentages:
            for task_idx, task in enumerate(needles):
                
                if device == "mps": torch.mps.empty_cache()
                elif device == "cuda": torch.cuda.empty_cache()

                char_name = CHARACTERS[task_idx % len(CHARACTERS)]
                needle_str = task["needle"].replace("{CHAR}", char_name)
                q_str = task["question"]

                n_tokens = tokenizer.encode(needle_str)
                q_tokens = tokenizer.encode(q_str)

                haystack_size = total_ctx - len(n_tokens) - len(q_tokens)
                if haystack_size <= 0:
                    pbar.update(1)
                    continue

                haystack_tokens = get_mixed_haystack(haystack_cache, haystack_size)
                insert_idx = int(haystack_size * depth_pct)

                search_range = haystack_tokens[insert_idx:insert_idx+100]
                if 19 in search_range:
                    insert_idx = insert_idx + search_range.index(19) + 1

                input_ids = haystack_tokens[:insert_idx] + n_tokens + haystack_tokens[insert_idx:] + q_tokens
                input_tensor = torch.tensor([input_ids]).to(device)

                with torch.no_grad():
                    out = model.generate(
                        input_tensor,
                        max_new_tokens=run_config.get("max_new_tokens", 15),
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                resp = tokenizer.decode(out[0][input_tensor.shape[1]:], skip_special_tokens=True).strip()
                
                first_name, last_name = char_name.split()
                is_correct = first_name.lower() in resp.lower() or last_name.lower() in resp.lower()

                results_list.append({
                    "model": model_name,
                    "context_length": total_ctx,
                    "depth_pct": round(depth_pct, 2),
                    "task_idx": task_idx,
                    "character": char_name,
                    "response": resp,
                    "is_correct": is_correct
                })
                pbar.update(1)

        pbar.close()

    safe_name = model_name.replace("/", "_")
    details_file = os.path.join(run_config["results_dir"], f"raw_{safe_name}.json")
    with open(details_file, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Raw logs saved to {details_file}")

if __name__ == "__main__":
    main()
