# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A Random Forest Classifier trained to predict whether an individual's salary is >$50K or <=$50K per year based on census data.
- **Model Type:** Random Forest Classifier
- **Model Version:** 1.0.0
- **Framework:** scikit-learn
- **Date:** December 2025

## Intended Use
- **Primary Use:** To classify salary range based on demographic features.
- **Intended Users:** Researchers and data scientists studying census data.
- **Out of Scope:** This model is not intended for use in hiring decisions or credit scoring without further fairness auditing.

## Training Data
- **Source:** UCI Machine Learning Repository (Census Income Dataset).
- **Split:** 80% training 20% testing.
- **Preprocessing:** Categorical features were One-Hot Encoded. The target label (salary) was binarized.

## Evaluation Data
- **Dataset:** 20% of the original census dataset.
- **Method:** Evaluated using Precision, Recall, and F1-score.

## Metrics
- **Precision:** Measures the accuracy of positive predictions.
- **Recall:** Measures the ability to find all positive instances.
- **F1 Score:** Harmonic mean of precision and recall.
**Overall Performance:**
- Precision: ~0.72
- Recall: ~0.62
- F1: ~0.67

## Ethical Considerations
- **Bias:** The dataset contains demographic information which could lead to biased predictions if not carefully monitored.
- **Privacy:** Data is publicly available and anonymized, but model predictions should be used responsibly.

## Caveats and Recommendations
- The model performs better on frequent classes.
- Further hyperparameter tuning could improve recall.