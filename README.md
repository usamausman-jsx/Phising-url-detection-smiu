# 🚨 Phishing URL Detection System

📖 About This Project

This project started as an academic endeavor to reproduce the research paper **"Detection of Phishing URLs Using Term Frequency Inverse Document Frequency (TF-IDF)"** by Sibhathallah M. and Dr. D. Sathya Srinivas (IJFMR 2024).

While the original paper achieved **96.6% accuracy** using traditional ML (Logistic Regression/Naive Bayes) with TF-IDF features, we took it further by implementing advanced Deep Learning architectures and robust feature engineering, achieving **99.80% accuracy** with a Hybrid CNN-RNN model.

🎯 Project Objectives

| Phase | Focus | Achievement |
|-------|-------|-------------|
| **Phase I: Reproduction** | Replicate original paper's TF-IDF + Logistic Regression approach | ✅ 96.6% accuracy achieved |
| **Phase II: Extension** | Implement Deep Learning models with enhanced features | ✅ 99.80% accuracy with Hybrid CNN-RNN |

🔬 Research Paper vs Our Implementation

| Aspect | Original Paper | Our Implementation |
|--------|---------------|-------------------|
| **Models Used** | Logistic Regression, Naive Bayes | ✅ LR, NB + Random Forest, Gradient Boosting, CNN, LSTM, GRU, Hybrid CNN-RNN |
| **Feature Engineering** | Basic TF-IDF only | ✅ 37+ handcrafted features + TF-IDF (3036 total) |
| **Deep Learning** | Only mentioned (no implementation) | ✅ Full CNN, LSTM, GRU, Hybrid architectures with training |
| **Advanced Features** | Not included | ✅ Shannon Entropy, Hex/IP detection, Suspicious TLDs, URL shorteners |
| **Class Balancing** | None | ✅ SMOTE for balanced 50:50 training |
| **Model Optimization** | None | ✅ Hyperparameter tuning, EarlyStopping, ReduceLROnPlateau |
| **Evaluation Metrics** | Accuracy only | ✅ Accuracy, Precision, Recall, F1-Score, ROC-AUC |
| **Deployment** | Not applicable | ✅ Real-time prediction class with ensemble voting |

🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT URL                               │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌───────────────────────────┐
│   REPRODUCTION BRANCH   │     │     EXTENSION BRANCH       │
│   (Original Paper)       │     │     (Our Contribution)    │
├─────────────────────────┤     ├───────────────────────────┤
│ • Regex Tokenization    │     │ • 37 Handcrafted Features │
│ • Snowball Stemming     │     │ • Shannon Entropy         │
│ • TF-IDF Vectorization  │     │ • Hex/IP Detection        │
│ • Logistic Regression   │     │ • Suspicious TLD Check    │
│ • Naive Bayes           │     │ • URL Shortener Detection │
└─────────────────────────┘     └───────────────────────────┘
         │                                   │
         └───────────────┬───────────────────┘
                         ▼
        ┌─────────────────────────────────────┐
        │         DEEP LEARNING MODELS        │
        │  (Our Extension - 99.80% Accuracy)  │
        ├─────────────────────────────────────┤
        │  • CNN  • LSTM  • GRU  • Hybrid     │
        │  • Bidirectional Layers              │
        │  • Batch Normalization               │
        │  • Dropout Regularization            │
        │  • Global Max Pooling                 │
        └─────────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────┐
        │         ENSEMBLE PREDICTION         │
        │  • Weighted Average of 8 Models     │
        │  • Confidence Score                  │
        │  • Real-time Analysis                │
        └─────────────────────────────────────┘
```

🧠 Models Implemented

Traditional ML (Reproduction + Extension)

| Model | Type | Accuracy |
|-------|------|----------|
| Logistic Regression | Reproduction (Original Paper) | 96.60% |
| Naive Bayes | Reproduction (Original Paper) | 98.90% |
| Random Forest | Our Extension | 99.63% |
| Gradient Boosting | Our Extension | 99.77% |

Deep Learning (Our Extension)

| Model | Architecture | Accuracy | Key Features |
|-------|--------------|----------|--------------|
| **CNN** | Conv1D (128,64) + GlobalMaxPooling | 99.80% | Local pattern detection |
| **LSTM** | Bidirectional LSTM (64,32) | 99.79% | Long-range dependencies |
| **GRU** | Bidirectional GRU (64,32) | 99.77% | Efficient sequence learning |
| **Hybrid CNN-RNN** | CNN + LSTM Parallel | **99.80%** | Best of both worlds |

🔍 Advanced Feature Engineering (Our Contribution)

We identified limitations in the original paper's feature set and added:

1. Shannon Entropy
```python
H(X) = -∑ P(xᵢ) log₂ P(xᵢ)
```
- **Purpose**: Detect algorithmically generated domains (DGA)
- **Example**: `google.com` (low entropy: 3.66) vs `x7z9q2a.com` (high entropy: >4.5)

2. Obfuscation Detection
| Feature | Description | Example |
|---------|-------------|---------|
| IP Address Detection | Flags raw IP URLs | `http://192.168.1.100/login` |
| Hex Encoding | Detects %xx patterns | `%61%64%6D%69%6E` (admin) |
| URL Shorteners | Identifies 50+ services | `bit.ly`, `tinyurl`, `goo.gl` |

3. Suspicious TLDs
```python
suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.club']
```

4. 37 Handcrafted Features
- **Length-based**: URL length, hostname length, path length
- **Character counts**: dots, hyphens, underscores, slashes
- **Ratios**: digit/letter/special character ratios
- **Patterns**: consecutive digits, consecutive letters

📊 Performance Comparison

| Metric | Original Paper | Our Hybrid Model | Improvement |
|--------|---------------|------------------|-------------|
| Accuracy | 96.60% | **99.80%** | ↑ 3.2% |
| Precision | 0.92 | **0.998** | ↑ 8.5% |
| Recall | 0.96 | **0.998** | ↑ 4.0% |
| F1-Score | 0.94 | **0.998** | ↑ 6.2% |

Model Comparison Dashboard

```
┌─────────────────────┬────────────┬──────────┬────────────┐
│ Model               │ Accuracy   │ Precision│ F1-Score   │
├─────────────────────┼────────────┼──────────┼────────────┤
│ Logistic Regression │ 96.60%     │ 0.92     │ 0.94       │
│ Naive Bayes         │ 98.90%     │ 0.99     │ 0.99       │
│ Random Forest       │ 99.63%     │ 0.99     │ 0.99       │
│ Gradient Boosting   │ 99.77%     │ 1.00     │ 1.00       │
│ CNN                 │ 99.80%     │ 0.998    │ 0.998      │
│ LSTM                │ 99.79%     │ 0.99     │ 0.99       │
│ GRU                 │ 99.77%     │ 0.99     │ 0.99       │
│ HYBRID CNN-RNN      │ 99.80%     │ 0.998    │ 0.998      │
└─────────────────────┴────────────┴──────────┴────────────┘
```

🚀 Quick Start

Installation

```bash
# Clone the repository
git clone https://github.com/usamausman-jsx/Phising-url-detection-smiu.git
cd Phising-url-detection-smiu/phishing_url_detection_TF-IDF_CNN%26RNN\ - Copy

# Install dependencies
pip install -r requirements.txt
```

Basic Usage

```python
from phishing_detector import PhishingURLDetector

# Initialize detector (loads all 8 models)
detector = PhishingURLDetector()
detector.load_models()

# Analyze a URL
result = detector.analyze_url("https://secure-login-paypal.com/verify")

print(f"Prediction: {'🔴 PHISHING' if result['is_phishing'] else '🟢 LEGITIMATE'}")
print(f"Confidence: {result['confidence']*100:.1f}%")
print(f"Entropy Score: {result['features']['entropy']:.2f}")
```

Example Output

```
============================================================
ANALYZING URL: https://secure-login-paypal.com/verify-account
============================================================

Prediction: 🔴 PHISHING
Confidence: 100.0%

Key Features:
  • URL Length: 46
  • Has HTTPS: Yes
  • Has IP Address: No
  • Phishing Keywords: 6
  • Suspicious TLD: No
  • URL Shortener: No
  • Entropy: 4.322 (HIGH - suspicious)

Model Scores:
  • LR      : 1.000
  • RF      : 1.000
  • GB      : 0.999
  • NB      : 1.000
  • CNN     : 1.000
  • LSTM    : 1.000
  • GRU     : 1.000
  • HYBRID  : 1.000
============================================================
```

## 📁 Project Structure

```
phishing_url_detection_TF-IDF_CNN&RNN/
├── 📓 phishing_URL.ipynb              # Main implementation notebook
├── 📄 Final Project Report Phishing URL.pdf  # Complete project report
├── 📄 comparision.pdf                  # Paper vs Implementation analysis
├── 📄 requirements.txt                 # Python dependencies
├── 📄 README.md                        # This file
│
├── 📁 saved_models/                     # Trained models (8 total)
│   ├── 📦 phishing_lr_model.pkl         # Logistic Regression
│   ├── 📦 phishing_nb_model.pkl         # Naive Bayes
│   ├── 📦 phishing_rf_model.pkl         # Random Forest
│   ├── 📦 phishing_gb_model.pkl         # Gradient Boosting
│   ├── 🧠 phishing_cnn_model.keras      # CNN
│   ├── 🧠 phishing_lstm_model.keras     # LSTM
│   ├── 🧠 phishing_gru_model.keras      # GRU
│   └── 🧠 phishing_hybrid_model.keras   # Hybrid CNN-RNN
│
└── 📁 assets/                            # Images and diagrams
```

📊 Dataset

The system uses a balanced dataset of ~137,000 URLs:

| Class | Source | Count |
|-------|--------|-------|
| Legitimate (0) | Open-source repositories | ~68,500 |
| Phishing (1) | PhishTank + Malicious DBs | ~68,500 |

SMOTE was applied to ensure perfect 50:50 class balance during training.

🛠️ Key Contributions (What We Added)

Beyond reproducing the original paper, we contributed:

1. 8 Models Instead of 2
- Added Random Forest, Gradient Boosting, CNN, LSTM, GRU, and Hybrid CNN-RNN
- Implemented proper training pipelines for each

2. 37 Advanced Features
- Shannon Entropy for DGA detection
- Hexadecimal encoding detection
- IP address detection in URLs
- URL shortening service identification
- Suspicious TLD checking
- Character distribution ratios

3. Production-Ready Code
- PhishingURLDetector` class for real-time predictions
- Model checkpointing and loading system
- Ensemble voting for robust predictions
- Confidence scoring and feature explanation

4. Comprehensive Evaluation
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrices
- Model comparison tables
- Training history visualization

📈 Results Visualization

Model Accuracy Comparison

Accuracy (%)
100 │                                        ████  ████
 99 │                             ████  ████  ████  ████  ████
 98 │                   ████  ████  ████  ████  ████  ████  ████
 97 │         ████  ████  ████  ████  ████  ████  ████  ████  ████
 96 │ ████  ████  ████  ████  ████  ████  ████  ████  ████  ████
    └──────────────────────────────────────────────────────────
       LR    NB    RF    GB    CNN   LSTM  GRU   Hybrid
       
       ■ Original Paper (96.6%)    ■ Our Implementation (99.8%)

🔮 Future Work

Based on our findings, we propose:

1. Real-time API Deployment - Package the hybrid model as a REST API for browser extensions
2. Transformer Models - Implement BERT/DistilBERT for semantic understanding
3. Adversarial Training - Test against adversarial examples to harden the system
4. Live Feed Integration - Connect with real-time phishing feeds for continuous learning


📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

⭐ Acknowledgments

- Original research paper authors for their foundational work
- Our instructor Muhammad Osama for guidance
- PhishTank for providing phishing URL dataset
