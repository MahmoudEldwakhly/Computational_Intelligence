# Optimization & Deep Learning Case Study (MNIST + Airbnb Pricing + LLM)

This repository contains a **Computational Intelligence** project implemented across two milestones, focusing on neural network optimization, regression modeling, and Large Language Model (LLM) integration for interpretability.

---

## Project Overview

### Milestone 1 — MNIST Optimizer Analysis
This milestone studies how different optimization algorithms influence neural network training behavior on the MNIST handwritten digit dataset.

**Implementation details:**
- Dataset loading and preprocessing (normalization and vectorization of images).
- Fixed neural network architecture (fully connected layers).
- Training with multiple optimizers:
  - Stochastic Gradient Descent (SGD)
  - SGD with Momentum
  - Adagrad
  - Adam
  - Custom Adaptive Learning Rate using Linear Search
- Measurement and visualization of:
  - Training loss and accuracy curves
  - Training time
  - Test evaluation
  - Confusion matrices (raw and heatmap form)

**Custom Adaptive Learning Rate:**
For each training batch, gradients are computed once, then multiple candidate learning rates are tested. The learning rate that produces the lowest temporary loss is selected and applied to update the model parameters.

---

### Milestone 2 — Airbnb Price Prediction & LLM-Based Interpretation
This milestone addresses a real-world regression problem: predicting Airbnb nightly prices using structured listing data.

**Key concepts:**
- Price treated as a continuous target, making regression the primary task.
- Optional categorization of prices (Low / Medium / High) used only for analysis and comparison.

**Data handling and preprocessing:**
- Dataset split into training, validation, and test sets with stratification.
- Missing data handling:
  - Numeric features: median imputation
  - Categorical features: most-frequent imputation
- Feature scaling using StandardScaler (Z-score normalization).
- One-hot encoding for categorical variables using scikit-learn pipelines.
- Feature engineering from amenities:
  - Extraction of luxury-related features
  - Creation of `luxury_count` and descriptive `luxury_items`

**Modeling:**
- Deep feedforward neural network built with TensorFlow/Keras.
- Use of Batch Normalization and Dropout layers to improve stability and reduce overfitting.
- Huber loss used to reduce sensitivity to outliers.
- Early stopping and learning-rate scheduling for controlled training.

**Evaluation and analysis:**
- Regression metrics and extensive diagnostic plots:
  - Prediction vs. ground truth
  - Residual distributions
  - Absolute and relative error analysis
- Confusion matrices created after mapping predicted prices to categories.

---

### Large Language Model (LLM) Integration
Transformers-based LLMs are integrated using the Hugging Face `transformers` library.

**LLM usage includes:**
- Few-shot classification of Airbnb listings into price categories using compact feature prompts.
- Generation of structured, human-readable explanations that justify predicted prices based on:
  - Location
  - Room and property type
  - Capacity and layout
  - Review scores
  - Amenities and luxury features
  - Booking and cancellation policies

LLMs are used for interpretability and reasoning rather than numeric prediction.

---

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- Hugging Face Transformers
- PyTorch (LLM inference backend)

---

## Suggested Repository Structure
```
.
├── milestone1_mnist_optimizers/
│   ├── mnist_optimizers.py
│   └── outputs/
├── milestone2_airbnb_pricing/
│   ├── airbnb_pricing_dnn_llm.ipynb
│   └── outputs/
├── reports/
│   ├── Milestone1_Report.pdf
│   └── Milestone2_Report.pdf
└── README.md
```

---

## Setup

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn transformers accelerate sentencepiece torch
```

---

## Running the Project

### MNIST Optimizer Comparison
```bash
python milestone1_mnist_optimizers/mnist_optimizers.py
```

### Airbnb Price Prediction
1. Download the Airbnb dataset used in the project.
2. Update the dataset path inside the notebook or script.
3. Run the notebook from start to end.

---

## Key Learning Outcomes
- Practical understanding of optimizer behavior and convergence dynamics.
- Construction of robust preprocessing pipelines for real-world data.
- Design of deep neural networks with regularization and stability techniques.
- Evaluation of regression models using numeric metrics and visual diagnostics.
- Application of Large Language Models for explainability and reasoning in machine learning workflows.
