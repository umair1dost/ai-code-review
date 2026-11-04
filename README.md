# AI-Powered Code Review and Bug Prediction System

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://tensorflow.org)
[![Angular](https://img.shields.io/badge/Angular-16-red.svg)](https://angular.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Final Year Project** — Institute of Management Sciences (IMSciences), Peshawar  
> **Authors:** Muhammad Nasir Khan · Ahmed Khan Khisro · Umair Dost  
> **Supervisor:** Mr. Omar Bin Samin  
> **Session:** 2021–2025

---

## Original Final-Year Project Report

The original final-year project report submitted to the **Institute of Management Sciences (IMSciences), Peshawar** is available below.

**[Read the Original Final-Year Project Report (PDF)](docs/New_Umair_Final_Year_Project_AI_Code_Review_Bug_Prediction.pdf)**

**Project:** AI-Powered Code Review and Bug Prediction System  
**Authors:** Muhammad Nasir Khan · Ahmed Khan Khisro · Umair Dost  
**Supervisor:** Mr. Omar Bin Samin  
**Session:** 2021–2025

---

## Overview

This project presents a hybrid AI system that automates software code review and predicts bug-prone modules. It combines a **Random Forest classifier** (trained on 87 software metrics) with an **LSTM neural network** (operating on tokenised source code) in a soft-voting ensemble, achieving **87.1% accuracy** and **AUC-ROC of 0.94** on a held-out test set of 2,812 labelled modules.

The system generates **structured, actionable review suggestions** with SHAP-based metric explanations and exposes its functionality through a **FastAPI REST backend** and an **Angular 16 web dashboard**.

---

## Key Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Random Forest | 83.4% | 0.81 | 0.86 | 0.83 | 0.91 |
| LSTM | 81.7% | 0.79 | 0.88 | 0.83 | 0.90 |
| **Ensemble (Ours)** | **87.1%** | **0.84** | **0.89** | **0.87** | **0.94** |

Compared to baselines:

| System | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| SonarQube (rule-based) | 71.2% | 0.68 | N/A |
| PMD (rule-based) | 68.5% | 0.65 | N/A |
| Logistic Regression | 76.3% | 0.74 | 0.82 |
| **Proposed Ensemble** | **87.1%** | **0.87** | **0.94** |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Angular Frontend                      │
│         Code Submission · Results Dashboard · Reports    │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP REST
┌────────────────────────▼────────────────────────────────┐
│                  FastAPI Backend                         │
│   /predict/metrics   ·   /predict/code   ·   /health    │
└──────────┬──────────────────────────┬───────────────────┘
           │                          │
  ┌────────▼────────┐       ┌─────────▼────────┐
  │  Random Forest  │       │   LSTM Network   │
  │  (61 metrics)   │       │ (token sequences) │
  └────────┬────────┘       └─────────┬────────┘
           │                          │
           └──────────┬───────────────┘
                      │
             ┌────────▼────────┐
             │ Soft Voting      │
             │ Ensemble (0.55/  │
             │ 0.45 weights)    │
             └────────┬────────┘
                      │
             ┌────────▼────────┐
             │ SHAP Explainer  │
             │ + Review Report │
             └─────────────────┘
```

---

## Repository Structure

```
ai-code-review/
├── backend/                    # FastAPI REST API
│   ├── app/
│   │   ├── api/                # Route handlers
│   │   ├── models/             # Pydantic schemas
│   │   ├── services/           # ML inference logic
│   │   └── utils/              # Helpers (metrics extractor, tokenizer)
│   ├── tests/                  # Pytest unit & integration tests
│   └── requirements.txt
├── frontend/                   # Angular 16 dashboard
│   └── src/
├── ml/                         # Training & evaluation scripts
│   ├── notebooks/              # Jupyter EDA & training notebooks
│   └── scripts/                # CLI training scripts
├── data/
│   ├── raw/                    # GitHub mining output (gitignored)
│   └── processed/              # Feature matrices (gitignored)
├── configs/                    # Hyperparameter YAML configs
├── docs/                       # Thesis PDF and architecture diagrams
├── .github/workflows/          # CI/CD pipeline (GitHub Actions)
├── docker-compose.yml
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker & Docker Compose (optional)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/ai-code-review.git
cd ai-code-review
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

### 3. Frontend Setup

```bash
cd frontend
npm install
ng serve
```

Dashboard available at: `http://localhost:4200`

### 4. Docker (Full Stack)

```bash
docker-compose up --build
```

---

## API Reference

### `POST /predict/metrics`
Accepts a JSON payload of extracted software metrics and returns a bug probability score.

**Request:**
```json
{
  "loc": 245,
  "cyclomatic_complexity": 18,
  "cbo": 12,
  "wmc": 24,
  "lcom": 0.73,
  "rfc": 45,
  "fan_in": 6,
  "fan_out": 9,
  "num_authors": 4,
  "code_churn": 312
}
```

**Response:**
```json
{
  "bug_probability": 0.87,
  "prediction": "BUGGY",
  "severity": "HIGH",
  "top_features": [
    {"feature": "cbo", "shap_value": 0.34},
    {"feature": "wmc", "shap_value": 0.28},
    {"feature": "loc", "shap_value": 0.19}
  ]
}
```

### `POST /predict/code`
Accepts raw source code and returns LSTM-derived code smell classifications and ensemble prediction.

**Request:**
```json
{
  "source_code": "def process_data(...):\n    ...",
  "language": "python"
}
```

**Response:**
```json
{
  "bug_probability": 0.91,
  "prediction": "BUGGY",
  "severity": "HIGH",
  "code_smells": ["God Class", "Long Method", "Duplicate Code"],
  "review_suggestion": "This module exhibits high coupling (CBO = 18) and excessive method complexity (average WMC = 24). Refactoring is recommended: decompose into smaller single-responsibility classes and reduce inter-module dependencies."
}
```

---

## ML Pipeline

### Data Collection
- 512 open-source GitHub repositories (Python & Java)
- Selection criteria: ≥500 commits, issue tracker with bug labels, ≥1,000 LOC
- Bug labelling via SZZ algorithm on commit messages

### Feature Engineering
- **87 raw metrics** extracted per module → **61 features** after correlation pruning
- Tools: Radon (Python), CKJMExtended (Java)
- Class imbalance handled with SMOTE (training split only)

### Models
| Component | Details |
|---|---|
| Random Forest | 200 estimators, balanced class weight, 5-fold CV grid search |
| LSTM | Embedding(10k vocab, 128-dim) → LSTM(128) → LSTM(64) → Dense(64) → Sigmoid |
| Ensemble | Soft voting: RF×0.55 + LSTM×0.45, threshold=0.5 |

### To Train from Scratch

```bash
cd ml/scripts
python mine_repositories.py --config ../../configs/mining_config.yaml
python extract_features.py --input ../../data/raw --output ../../data/processed
python train_random_forest.py --config ../../configs/rf_config.yaml
python train_lstm.py --config ../../configs/lstm_config.yaml
python evaluate_ensemble.py
```

---

## Testing

```bash
cd backend
pytest tests/ -v --cov=app --cov-report=html
```

---

## Citation

If you use this work, please cite:

```bibtex
@misc{khan2025aireview,
  title     = {AI-Powered Code Review and Bug Prediction System},
  author    = {Muhammad Nasir Khan and Ahmed Khan Khisro and Umair Dost},
  year      = {2025},
  school    = {Institute of Management Sciences, Peshawar},
  note      = {BCS (Hons.) Final Year Project}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
