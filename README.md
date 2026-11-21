# 🌸 Iris Flower Species Classification (End-to-End ML Project)

I built this machine learning project from scratch using **Python** and **Scikit-Learn** to classify Iris flowers into three species (*Setosa*, *Versicolor*, and *Virginica*) based on sepal and petal measurements. 

The goal of this project was to practice a clean, leak-free ML workflow—comparing multiple classification algorithms using both a hold-out test split and 5-fold cross-validation (`cross_val_score`).

---

## 🛠️ Project Workflow

1. **Train-Test Split:** Split the 150-sample dataset into **80% training** (120 samples) and **20% testing** (30 samples) using `train_test_split` (`random_state=42`).
2. **Feature Scaling:** Applied `StandardScaler` strictly fitted on `x_train` and transformed on `x_test` to prevent data leakage.
3. **Model Training & Evaluation:** Trained three classifiers and evaluated them using **Accuracy**, **Macro Precision**, **Macro Recall**, **Macro F1-Score**, and **Confusion Matrices**.
4. **Cross-Validation (`cross_val_score`):** Performed **5-fold cross-validation** on the scaled training data to verify how consistently each model generalizes across different data splits.

---

## 📊 Model Performance Comparison

### 1. Test Set Evaluation (20% Hold-Out Split)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Gaussian Naive Bayes** | **96.67%** | **0.9697** | **0.9667** | **0.9666** |
| **K-Nearest Neighbors (`k=7`)** | **96.67%** | **0.9697** | **0.9667** | **0.9666** |
| **Logistic Regression** | 93.33% | 0.9333 | 0.9333 | 0.9333 |

### 2. 5-Fold Cross-Validation (`cross_val_score`)

| Model | Mean CV Accuracy | Standard Deviation |
| :--- | :---: | :---: |
| **Gaussian Naive Bayes** | **95.83%** | `+/- 0.0264` |
| **Logistic Regression** | **95.83%** | `+/- 0.0264` |
| **K-Nearest Neighbors (`k=7`)** | **95.83%** | `+/- 0.0373` |

---

## 🔍 Confusion Matrices & Key Takeaways

* **Gaussian Naive Bayes (`96.67%` Test Accuracy):**
  ```text
  [[10  0  0]
   [ 0  9  1]
   [ 0  0 10]]
  ```
  *Predicted Setosa and Virginica with 100% accuracy; misclassified only 1 Versicolor sample.*

* **K-Nearest Neighbors (`96.67%` Test Accuracy):**
  ```text
  [[10  0  0]
   [ 0 10  0]
   [ 0  1  9]]
  ```
  *Predicted Setosa and Versicolor with 100% accuracy; misclassified only 1 Virginica sample.*

* **Logistic Regression (`93.33%` Test Accuracy):**
  ```text
  [[10  0  0]
   [ 0  9  1]
   [ 0  1  9]]
  ```
  *Classified all Setosa samples accurately with 2 minor overlaps between Versicolor and Virginica.*

### 💡 Key Insight
While **Naive Bayes** and **KNN** achieved higher accuracy (`96.67%`) on the 30-sample test set, **5-fold cross-validation (`cross_val_score`)** showed that all three models achieve an identical **95.83% mean accuracy** across the full training set—with **Gaussian Naive Bayes** and **Logistic Regression** showing slightly lower variance (`+/- 0.0264`) than KNN (`+/- 0.0373`).
   ```
3. Open the Jupyter Notebook (`.ipynb`) and run all cells.
