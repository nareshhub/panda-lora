# LoRA (Low-Rank Adaptation)

## 1. Introduction

LoRA is a Parameter-Efficient Fine-Tuning (PEFT) technique that adapts large pre-trained models like GPT-3 (175B parameters) to downstream tasks. It freezes the original model weights and introduces small, trainable low-rank matrices into selected Transformer layers, reducing computational cost and GPU memory requirements.

## 2. Advantages

- Parameter Efficiency: Reduces trainable parameters by up to 10,000 times in reported configurations.

- Memory Efficiency: Reduces GPU memory requirements by approximately 3 times.

- Task Adaptability: Supports multiple downstream tasks by switching LoRA adapters (matrices A and B).

- Inference Efficiency: No additional inference latency when adapter weights are merged.

- Compatibility: Can be combined with prefix-tuning.

## 3. Training Objective

LoRA freezes the pre-trained weights W₀ and trains only the low-rank matrices A and B using gradient-based optimization, such as Adam. The objective is to maximize the conditional probability of target outputs for a downstream task.

## 4. Mathematical Formulation

$$
W = W_0 + \frac{\alpha}{r}BA
$$

- W₀: Frozen pre-trained weights.

- B ∈ ℝᵈˣʳ: Trainable matrix.

- A ∈ ℝʳˣᵏ: Trainable matrix.

- r: Low rank, where r ≪ min(d, k).

- α/r: Scaling factor controlling the update magnitude.

For different tasks, LoRA replaces the adapter matrices without retraining the base model.

## 5. Application in Transformer Architecture

LoRA can be applied to selected weight matrices:

- Self-Attention: Query (Wq), Key (Wk), Value (Wv), Output (Wo).

- MLP: Feed-forward weight matrices.

Typically, LoRA adapts Query and Value projections, while the original attention and MLP weights remain frozen.

## 6. GPT-3 175B Example

| Metric | Full Fine-Tuning | LoRA |

| --- | --- | --- |

| GPU Memory | 1.2 TB | 350 GB |

| Trainable Parameters | 175B | Significantly fewer |

| Rank | N/A | r = 4 |

| Adapted Matrices | All | Query and Value |

| Training Cost | High | Lower |

Result: Approximately 71% reduction in GPU memory, enabling training with fewer GPUs and smaller checkpoints.

## 7. Interview Summary

LoRA is a parameter-efficient fine-tuning technique that freezes pre-trained model weights and trains only small, low-rank matrices within selected Transformer layers. It reduces trainable parameters, GPU memory, and computational costs while supporting multiple task-specific adapters without retraining the entire model.
