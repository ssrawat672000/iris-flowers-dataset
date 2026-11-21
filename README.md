# 🌸 Iris Flower Species Classification

A beginner-friendly end-to-end Machine Learning classification project built with Python and Scikit-Learn to predict Iris flower species (*Setosa*, *Versicolor*, and *Virginica*).

## 🛠️ Workflow
1. **Train-Test Split:** 80/20 stratified split (`random_state=42`).
2. **Feature Scaling:** Applied `StandardScaler` properly (fitted only on training data to prevent data leakage).
3. **Model Training & Comparison:** Evaluated Logistic Regression, K-Nearest Neighbors (KNN), and Gaussian Naive Bayes.

## 📊 Model Performance (Test Set)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **KNN (`k=7`)** | **96.67%** | 0.97 | 0.97 | 0.97 |
| **Gaussian Naive Bayes** | **96.67%** | 0.97 | 0.97 | 0.97 |
| **Logistic Regression** | *<Add your score>* | *<Score>* | *<Score>* | *<Score>* |

## 🚀 How to Run
1. Clone this repo: `git clone https://github.com`
2. Install dependencies: `pip install scikit-learn pandas numpy`
3. Open and run the `.ipynb` notebook.
