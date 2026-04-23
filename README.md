# 🧠 Customer Review Sentiment Analysis

A production-ready NLP application that analyzes customer reviews and classifies them as **Positive** or **Negative** using Machine Learning.

🔗 **Live App:** https://customer-review-sentiment.streamlit.app

---

## 🚀 Project Overview

This project focuses on solving a real-world business problem:
**understanding customer sentiment at scale.**

The system processes raw customer reviews, transforms them into numerical features using NLP techniques, and predicts sentiment using a trained machine learning model.

---

## 🎯 Key Features

* ✔ Real-time sentiment prediction
* ✔ Clean and minimal Streamlit UI
* ✔ Handles imbalanced data using SMOTE
* ✔ End-to-end ML pipeline (Data → Model → Deployment)
* ✔ Deployed on Streamlit Cloud

---

## 🧠 Tech Stack

* **Language:** Python
* **Libraries:**

  * scikit-learn
  * pandas, numpy
  * imbalanced-learn (SMOTE)
  * streamlit

---

## ⚙️ Machine Learning Pipeline

1. **Data Cleaning**

   * Lowercasing
   * Removing special characters
   * Text normalization

2. **Feature Engineering**

   * TF-IDF Vectorization
   * N-grams (1,2)

3. **Handling Imbalance**

   * Applied SMOTE to balance classes

4. **Model Training**

   * LinearSVC classifier

5. **Evaluation**

   * Accuracy: ~94%
   * Balanced performance across classes

---

## 📊 Model Performance

| Metric    | Score    |
| --------- | -------- |
| Accuracy  | 94%      |
| Precision | High     |
| Recall    | Balanced |
| F1-score  | Strong   |

---

## 🖥️ How to Run Locally

```bash
git clone https://github.com/VaishaliMehta2003/Customer-Review-Sentiment.git
cd Customer-Review-Sentiment
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deployment

The application is deployed using **Streamlit Cloud** with environment compatibility handled via:

* Python version control
* Model serialization (pickle)
* Dependency management

---

## 💡 Business Use Case

This system can help companies:

* Analyze customer feedback automatically
* Improve product quality
* Detect negative trends early
* Support data-driven decision making

---

## 📌 Future Improvements

* Add sentiment confidence score
* Use deep learning models (LSTM/BERT)
* Multi-class sentiment (Positive/Neutral/Negative)
* Dashboard analytics

---

## 👩‍💻 Author

**Vaishali Mehta**
Machine Learning Enthusiast
