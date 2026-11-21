<div align="center">

# 🐼 PandaLoRA

<img src="assets/Panda_with_Laptop.png" alt="PandaLoRA exploring neural networks and low-rank adapters" width="300" />

**Small Adapters. Powerful Models.**

*A use-case-driven AI project focused on parameter-efficient LLM fine-tuning using LoRA and QLoRA, with model optimization, performance evaluation, and benchmarking..*

</div>

## Overview

PandaLoRA documents the foundations of **Low-Rank Adaptation (LoRA)** and plans to expand into hands-on fine-tuning experiments, evaluation, and reproducible benchmarks. It focuses on reducing trainable parameters and GPU memory requirements while adapting pre-trained language models to downstream tasks.

> **Project status:** Documentation phase. Fine-tuning scripts, evaluation implementations, and benchmark results are planned, not yet included.

## Repository structure

```text
panda-lora/
├── README.md
├── docs/
│   ├── lora_fundas.md
├── src/
│   ├── train_lora.py
│   └── train_qlora.py
├── configs/
│   └── lora_config.yaml
├── evaluation/
│   └── evaluate.py
├── benchmarks/
│   └── metrics.csv
├── requirements.txt
└── .gitignore
```

## Quick start

Requires Python 3.10+ and enough memory to load the model. QLoRA additionally requires a compatible NVIDIA CUDA GPU and `bitsandbytes` installation. Model weights are downloaded from Hugging Face on first use; check their license and access requirements.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train_lora.py --config configs/lora_config.yaml
```

For 4-bit QLoRA on a compatible CUDA machine:

```bash
python src/train_qlora.py --config configs/lora_config.yaml
```

The default dataset contains **four tiny educational samples**. It is only a pipeline smoke test, not meaningful model training. Replace it with your own JSONL training file containing `{"text": "..."}` per line, and set `train_file` in the YAML config. Use a separate held-out evaluation dataset.

## Evaluate

Create `eval.txt` with held-out text (do not reuse training samples), then run:

```bash
python evaluation/evaluate.py --text-file eval.txt
python evaluation/evaluate.py --text-file eval.txt --adapter outputs/panda-lora
```

This computes loss and perplexity on the first `max_length` tokens, **not** classification accuracy or task-specific quality. Compare models on the same evaluation text and tokenization. Training metrics are saved under `outputs/`; add verified experimental results manually to `benchmarks/metrics.csv`.

## Documentation

- [LoRA fundamentals (Markdown)](docs/lora_fundas.md)

## Roadmap

- [x] LoRA and QLoRA starter scripts
- [x] Configuration-driven training
- [x] Perplexity evaluation script
- [ ] Real task-specific datasets and data validation
- [ ] Quality metrics (accuracy/F1) and robust held-out evaluation
- [ ] Reproducible GPU memory, throughput, and latency benchmarks
- [ ] Base vs LoRA vs QLoRA comparison report

## Reference

Hu et al. (2021), [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
