# 🧪 Diabetes Prediction App using KNN

This project is a full machine learning pipeline that predicts whether a patient is diabetic based on medical data. It includes model development, evaluation, and a fully deployed web app using **Streamlit**.

> ✅ **[Try the Live App Here](https://your-app-name.streamlit.app)**

---

## 📌 Table of Contents

* [🔍 Problem Statement](#-problem-statement)
* [📦 Dataset](#-dataset)
* [🛠️ Tools Used](#-tools-used)
* [🧠 ML Approach](#-ml-approach)
* [📈 Model Evaluation](#-model-evaluation)
* [🖥️ Live App](#-live-app)
* [📊 Visualizations](#-visualizations)
* [🗂️ Project Structure](#-project-structure)
* [🚀 How to Run Locally](#-how-to-run-locally)
* [💡 Future Improvements](#-future-improvements)

---

## 🔍 Problem Statement

Diabetes affects millions worldwide. Early diagnosis can help prevent serious complications. This app helps predict the likelihood of diabetes using user-inputted medical features and a trained machine learning model.

---

## 📦 Dataset

* **Name:** PIMA Indians Diabetes Database
* **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
* **Samples:** 768
* **Features:** Pregnancies, Glucose, BMI, Age, etc.
* **Target:** Outcome (1 = Diabetic, 0 = Not Diabetic)

---

## 🛠️ Tools Used

* `Python`, `Pandas`, `NumPy`
* `scikit-learn` (KNN, SVM, metrics)
* `Matplotlib`, `Seaborn` for plots
* `Streamlit` for deployment
* `Pickle` for model serialization

---

## 🧠 ML Approach

* ✅ Cleaned and preprocessed data (replaced zeroes, scaled features)
* ✅ Trained and tuned a **K-Nearest Neighbors (KNN)** model
* ✅ Compared with Support Vector Machine (SVM)
* ✅ Selected KNN (`k=13`) based on cross-validation performance
* ✅ Saved model using `pickle`
* ✅ Deployed app using **Streamlit Cloud**

---

## 📈 Model Evaluation

| Metric   | KNN (k=13) |
| -------- | ---------- |
| Accuracy | 77.3%      |
| F1 Score | 0.65       |
| ROC AUC  | 0.78       |

Confusion Matrix:

```
Predicted
     0     1
0  [86,   13]
1  [22,   33]
```

> ✅ KNN outperformed SVM in this dataset.

---

## 🖥️ Live App

🎯 **Try the model live** on Streamlit:

🔗 [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)

Just enter patient values and click **Predict** to see the result!

---

## 📊 Visualizations

The notebook includes:

* 🔹 Side-by-side confusion matrices
* 🔹 ROC curve comparison (KNN vs SVM)
* 🔹 Accuracy and F1 bar chart
* 📁 All images saved in `images/` folder (optional)

---

## 🗂️ Project Structure

```
📁 diabetes-prediction-knn
├── 📁 app/
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── knn_model.pkl
├── diabetes_knn_svm.ipynb
├── README.md
└── images/ (optional)
