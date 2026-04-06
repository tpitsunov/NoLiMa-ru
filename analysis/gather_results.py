import os
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

def parse_results(results_dir):
    all_files = glob.glob(os.path.join(results_dir, "raw_*.json"))
    
    if not all_files:
        print("Нет файлов для анализа в", results_dir)
        return
        
    for file_path in all_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not data:
            continue
            
        df = pd.DataFrame(data)
        
        # Агрегация точности по дистанции и глубине
        agg = df.groupby(['model', 'context_length', 'depth_pct'])['is_correct'].mean().reset_index()
        agg.rename(columns={'is_correct': 'accuracy'}, inplace=True)
        
        model_name = data[0]["model"]
        safe_name = model_name.replace("/", "_")
        
        # Строим Heatmap
        pivot_df = agg.pivot(index="depth_pct", columns="context_length", values="accuracy")
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={'label': 'Accuracy'})
        
        plt.title(f"NoLiMa Heatmap: {model_name}")
        plt.xlabel("Context Length (tokens)")
        plt.ylabel("Needle Depth (%)")
        
        plot_path = os.path.join(results_dir, f"heatmap_{safe_name}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"📊 Heatmap сохранен: {plot_path}")
        
        # Вычисляем Base Score (условимся, что это минимальный контекст < 1k, обычно 256-500)
        short_ctx_df = agg[agg['context_length'] <= 1000]
        if not short_ctx_df.empty:
            base_score = short_ctx_df['accuracy'].mean()
            print(f"🧮 Base Score (<1K) для {model_name}: {base_score:.2f}")
            thresh = base_score * 0.85
            
            # Ищем Effective Length (максимальный контекст, где средняя точность >= 85% от Base Score)
            effective_length = "<1K"
            ctx_means = agg.groupby('context_length')['accuracy'].mean()
            
            for ctx in sorted(ctx_means.index):
                if ctx_means[ctx] >= thresh:
                    effective_length = str(ctx)
                else:
                    break
                    
            print(f"📉 Effective Length (>= {thresh:.2f}): {effective_length}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="nolima_ru/results")
    args = parser.parse_args()
    
    parse_results(args.results_dir)
