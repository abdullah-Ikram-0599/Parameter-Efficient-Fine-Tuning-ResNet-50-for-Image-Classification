# Parameter-Efficient Fine-Tuning of ResNet-50 for Image Classification

**Convolutional Adapter-based PEFT on ResNet-50 — achieving 89.2% accuracy on Oxford Flowers 102 dataset with only 2.65% trainable parameters.**

## Overview

This project implements **Convolutional Adapter–based Parameter-Efficient Fine-Tuning (PEFT)** on a pre-trained ResNet-50 backbone for image classification on the Oxford Flowers 102 dataset.

Instead of updating all model weights (which involves ~25 Million parameters), lightweight convolutional adapter modules are injected into the frozen ResNet-50 layers stage-wise. Only these adapter weights are trained, drastically reducing compute and memory requirements while preserving the rich ImageNet representations learned by the backbone.

The trained model is deployed through a **production-grade inference pipeline** and deployed as an **interactive Gradio web application** for real-time flower classification.

## Highlights

| Metric | Value |
|---|---|
| 🎯 Test Accuracy | **89.2%** |
| 🔢 Trainable Parameters | **~2.65%** of total |
| 📉 Parameter Reduction | **~97.35%** vs full fine-tuning |
| 📊 Accuracy Retention | **~95%** of full fine-tuning accuracy |
| 🗂️ Dataset | Oxford-102 Flowers |
| 🏗️ Backbone | ResNet-50 (ImageNet pre-trained) |

## Training Techniques

To ensure stable and generalisable training, the following techniques were incorporated:

- **Gradient Clipping** — prevents exploding gradients during adapter training
- **Early Stopping** — stops training when validation performance plateaus, reducing overfitting
- **Learning Rate Scheduling** — cosine/step decay for smooth convergence
- **Frozen Backbone** — ImageNet representations are fully preserved; only adapters learn

---

## Instructions:

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/abdullah-Ikram-0599/Parameter-Efficient-Fine-Tuning-ResNet-50-for-Image-Classification.git
cd Parameter-Efficient-Fine-Tuning-ResNet-50-for-Image-Classification

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Gradio App

Launch the interactive web application for real-time image classification directly from the project root:

```bash
python3 -m src.inference.gradio_app
```

Once running, open the local URL printed in your terminal (e.g., `http://127.0.0.1:7860`) in your browser. Upload any flower image and get instant predictions with confidence scores.

---

## Results

The Conv-Adapter PEFT approach achieves compelling results compared to full fine-tuning:

| Approach | Trainable Params | Test Accuracy |
|---|---|---|
| Full Fine-Tuning (baseline) | 100% | 94.6% |
| **Conv-Adapter PEFT ** | **~2.65%** | **89.2%** |

- **97.35% fewer trainable parameters** with only a modest accuracy trade-off
- Retains **~95% of full fine-tuning accuracy**, demonstrating the efficiency of adapter-based PEFT
