# Beyond Point Predictions: Manifold Expansion and Dual Alignment for Robust Time Series Distillation

**[Anonymous Repository for Peer Review]**

This repository contains the official PyTorch implementation of the paper: **"Beyond Point Predictions: Manifold Expansion and Dual Alignment for Robust Time Series Distillation"**.

## 📌 Abstract

We propose **Dynamic Structural Distillation (DSD)**, a robust framework for compressing heavy Transformer-based forecasting models (Teachers) into lightweight linear models (Students). Unlike traditional methods that only mimic point-wise output, DSD introduces:
1.  **Regime-Aware Distillation**: Adaptive weighting based on teacher uncertainty to mitigate negative transfer.
2.  **Manifold Expansion**: Projecting the student's latent space to match the teacher's dimensionality.
3.  **Dual Alignment**: Combining Macro-structural (SP) and Micro-geometric (OT) alignment to transfer rich structural knowledge.

## 📂 Project Structure

```text
.
├── data/                   # Dataset files (ETTh1, Electricity, etc.)
├── layers/                 # Model layers (DSD modules, Projectors)
├── models/                 # Model definitions (PatchTST, DistilDLinear/LMP-Net)
├── scripts/                # Shell scripts for reproduction
├── utils/                  # Utility functions (metrics, tools)
├── run.py   # Main execution script for distillation
├── requirements.txt        # Dependencies
└── README.md

```

## 🚀 Getting Started

### 1. Environment Setup

The code is tested with Python 3.8+ and PyTorch 1.10+.

```bash
# Create a virtual environment
conda create -n dsd python=3.8
conda activate dsd

# Install dependencies
pip install -r requirements.txt

```

### 2. Data Preparation

Please download the standard time-series benchmarks (ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Traffic, Weather, Exchange) and place them in the `./data/` directory.

### 3. Training Teachers (Optional)

You can train the teacher models (e.g., PatchTST) from scratch or use the pre-trained checkpoints provided in the `./baseline/` folder (if available).

## 🏃 Usage & Examples

### Distillation Training (DSD)

To distill the knowledge from a Teacher (PatchTST) to a Student (LMP-Net/DistilDLinear) using our DSD framework:

```bash
# Example: Distilling on ETTh1 with Prediction Length = 96
python run.py \
  --task_name hetero_dsd \
  --is_training 1 \
  --root_path ./data/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model DistilDLinear \
  --teacher_model PatchTST \
  --teacher_ckpt ./baseline/path_to_teacher_checkpoint.pth \
  --lambda_kd 0.5 \
  --lambda_spkd 0.1 \
  --lambda_ot 0.1 \
  --kd_gamma 0.5 \
  --gpu 0

```

### Key Arguments

| Argument | Description | Default         |
| --- | --- |-----------------|
| `--model` | The student model (e.g., `DistilDLinear` for LMP-Net) | `DistilDLinear` |
| `--teacher_model` | The teacher architecture | `PatchTST`      |
| `--lambda_kd` | Weight for Regime-Aware Output Distillation () | `0.5`           |
| `--lambda_align` | Weight for Macro-Structural Alignment () | `0.5`           |
| `--alpha_ot` | Weight for Micro-Geometric Alignment () | `0.1`           |
| `--kd_gamma` | Focusing parameter for uncertainty weighting () | `0.5`           |
