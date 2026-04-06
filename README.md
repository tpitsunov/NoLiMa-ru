# NoLiMa Russian Benchmark

This project is a Russian language adaptation of the NoLiMa (Long-Context Evaluation Beyond Literal Matching) benchmark.

The goal of this benchmark is to evaluate the ability of LLMs to retrieve facts from long contexts without relying on direct lexical matching (a common shortcut used by models in standard Passkey tests).

## Requirements
Ensure the following packages are installed:
```bash
pip install transformers torch datasets pandas seaborn tqdm matplotlib
```

## Running the Benchmark

### 1. Preparing the Haystack
To ensure consistent "noise" across all tests, first generate the tokenized "haystack" (cache):
```bash
python data/build_datasets.py \
  --tokenizer evilfreelancer/ruGPT3XL-8k \
  --haystack_out data/haystack/haystack_cache.json \
  --needles_in data/needles_ru.json
```
This script also validates `needles_ru.json` to ensure no lexical overlap between the needles and the questions.

### 2. Running Inference
The `configs/` directory contains model presets and testing parameters. To run the inference:
```bash
python evaluation/evaluate.py \
  --model_config configs/model_configs/rugpt3xl_8k.json \
  --run_config configs/run_configs/standard_test.json
```
*Note: This step is expected to be performed on high-performance rented hardware.*

### 3. Analytics
Results are saved in the format `results/raw_<model_name>.json`. To aggregate them, obtain metrics (Base Score, Effective Length), and generate plots (Heatmaps), run:
```bash
python analysis/gather_results.py --results_dir results/
```
