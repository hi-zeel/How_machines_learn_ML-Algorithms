# 🧠 Machine Learning Playground — Student Learning Portfolio

A collection of small Python projects I built while learning **Machine Learning**.  
Each script focuses on one idea — from **Linear Regression** and **Logistic Regression** to **K-Means**, **PCA**, and **CNNs**.  
The goal is to demonstrate practical understanding of both **supervised** and **unsupervised** ML, with clean, reproducible code.

---

## 📂 Repository Structure (typical)

> Your folder names or files may differ slightly — this README explains how each *type* of script fits into the bigger picture.

```
ML/
│
├── ANN/                        # Feedforward neural nets (tabular/classification)
│   ├── basic_ann.py
│   ├── dataset.csv
│   └── requirements.txt
│
├── CNN/                        # Convolutional networks (images)
│   ├── mnist_cnn.py
│   ├── cifar10_cnn.py
│   └── utils.py
│
├── DecisionTree/
│   └── decision_tree_example.py
│
├── KMeans/
│   └── kmeans_clustering.py
│
├── LinearRegression/
│   └── linear_regression.py
│
├── LogisticRegression/
│   └── logistic_regression.py
│
├── NaiveBayes/
│   └── naive_bayes.py
│
├── PCA/
│   └── pca_example.py
│
├── Perceptron/
│   └── simple_perceptron.py
│
├── RandomForest/
│   └── random_forest_example.py
│
├── SVM/
│   └── svm_example.py
│
├── utils/                      # Shared helpers
│   ├── data_loader.py
│   └── plot_utils.py
│
├── requirements.txt
└── README.md
```

If your local layout is different, just adjust the paths in the run commands below — the learning themes still apply.

---

## 🧪 What I Practiced (quick map)

| Area | Script(s) | What I practiced |
|---|---|---|
| **Regression (Supervised)** | `LinearRegression/linear_regression.py` | Fitting lines, train/test split, R², MAE/MSE, residual plots |
| **Binary/Multiclass Classification (Supervised)** | `LogisticRegression/logistic_regression.py`, `SVM/svm_example.py`, `NaiveBayes/naive_bayes.py`, `DecisionTree/decision_tree_example.py`, `RandomForest/random_forest_example.py` | Confusion matrix, precision/recall/F1, class imbalance handling, cross‑validation |
| **Clustering (Unsupervised)** | `KMeans/kmeans_clustering.py` | Choosing K (elbow/silhouette), scaling effects, cluster visualization |
| **Dimensionality Reduction (Unsupervised)** | `PCA/pca_example.py` | Explained variance, projection, preprocessing pipelines |
| **Perceptron & ANN (Deep Learning basics)** | `Perceptron/simple_perceptron.py`, `ANN/basic_ann.py` | Activation functions, loss surfaces, learning rates, early stopping |
| **CNNs for Images** | `CNN/mnist_cnn.py`, `CNN/cifar10_cnn.py` | Convolutions, pooling, overfitting control (dropout, augmentation) |
| **Utilities** | `utils/data_loader.py`, `utils/plot_utils.py` | Reusable loaders, clean plots, experiment hygiene |

---

## ⚙️ Setup

> Python 3.8–3.11 recommended.

```bash
# clone
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# (optional) create env
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# install
pip install -r requirements.txt
```

If some folders have their own `requirements.txt` (e.g., `ANN/requirements.txt`), install those when you work in that subproject.

---

## ▶️ How to Run (examples)

> Replace paths if your layout differs.

### Linear Regression
```bash
python LinearRegression/linear_regression.py
```
- **Outputs:** coefficients, metrics (R², MAE/MSE), and an optional predicted-vs-actual plot.

### Logistic Regression
```bash
python LogisticRegression/logistic_regression.py
```
- **Outputs:** confusion matrix, precision/recall/F1, ROC-AUC (if implemented).

### K-Means Clustering
```bash
python KMeans/kmeans_clustering.py
```
- **Outputs:** cluster centers, inertia, silhouette; 2D scatter with color-coded clusters.

### PCA
```bash
python PCA/pca_example.py
```
- **Outputs:** explained variance plot, 2D projection for visualization.

### SVM / Decision Trees / Random Forest
```bash
python SVM/svm_example.py
python DecisionTree/decision_tree_example.py
python RandomForest/random_forest_example.py
```
- **Outputs:** classification metrics and decision boundary plots (for 2D demos).

### Perceptron & ANN
```bash
python Perceptron/simple_perceptron.py
python ANN/basic_ann.py
```
- **Outputs:** training curves (loss/accuracy), final metrics; may save models.

### CNNs (MNIST / CIFAR-10)
```bash
python CNN/mnist_cnn.py
python CNN/cifar10_cnn.py
```
- **Outputs:** accuracy curves, test accuracy; may download datasets automatically.

---

## 📊 Experiment Hygiene (what I focused on)

- **Reproducibility:** setting random seeds when practical.
- **Preprocessing:** scaling for SVM/K-Means; one‑hot encoding for neural nets.
- **Validation:** `train_test_split()` and **K-fold CV** for robust estimates.
- **Metrics:** regression (R², MAE, RMSE), classification (F1, ROC-AUC), clustering (silhouette).  
- **Visualization:** consistent plots (learning curves, confusion matrices, PCA/K-Means projections).

---

## 🧩 Notes on Datasets

- Most scripts use either **toy datasets** from scikit-learn or classics like **MNIST/CIFAR-10** (downloaded automatically).
- If a script expects a local CSV (e.g., `ANN/dataset.csv`), it will either generate one or print a clear error with the expected format.

---

## 🛠️ Troubleshooting

- **Import errors:** verify your working directory (`pwd`) and `PYTHONPATH`; or run scripts from repo root.
- **Matplotlib issues:** use a non-interactive backend (`Agg`) if running in headless mode.
- **CUDA/TensorFlow warnings:** DL scripts fall back to CPU if GPU toolkits aren’t installed.

---

## 🗺️ Roadmap / Next Steps

- Add **hyperparameter search** (GridSearchCV/Optuna).
- Add **time-series** models (ARIMA/LSTM).
- Add **NLP** experiments (TF‑IDF, word embeddings, Transformers-lite).

---
