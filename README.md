## **Asteroid Hazard Prediction using Machine Learning**

## a. Problem Statement
The goal of this project is to build a Machine Learning based classification system to predict whether an asteroid is potentially hazardous (PHA) or not, using orbital and observational features provided in the NASA asteroid dataset.  

The project implements multiple classification models, evaluates them using standard classification metrics, and deploys an interactive Streamlit web application for real-time prediction and evaluation.

---

## **b. Dataset Description**
**Dataset Name:** Asteroid Features for Hazardous Prediction (NASA)  
**Source:** Kaggle (NASA dataset)  
**Type:** Binary Classification Dataset  

### Target Variable:
- **pha**  
  - `0` → Not Hazardous  
  - `1` → Hazardous  

### Features Used:
The dataset contains asteroid orbital and observational parameters such as:
- Semi-major axis (`a`)
- Eccentricity (`e`)
- Inclination (`i`)
- Longitude of ascending node (`om`)
- Argument of perihelion (`w`)
- Perihelion distance (`q`)
- Aphelion distance (`ad`)
- Orbital period (`per_y`, `per`)
- Mean anomaly (`ma`)
- Absolute magnitude (`H`)
- Minimum orbit intersection distance (`moid`)
- One-hot encoded asteroid classes (`class_*`)

### Dataset Size:
- More than **500 instances**
- More than **12 features** (after preprocessing and one-hot encoding)

Missing values were handled by dropping high-null columns and imputing required numeric values.

---

## **c. Models Used and Evaluation Metrics**

The following six classification models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (GaussianNB)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

Each model was evaluated using the following metrics:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## Model Comparison Table

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression      | 0.6200   | 0.9992 | 1.0000    | 0.2400 | 0.3871 | 0.3693 |
| Decision Tree            | 1.0000   | 1.0000 | 1.0000    | 1.0000 | 1.0000 | 1.0000 |
| KNN                      | 0.7700   | 1.0000 | 1.0000    | 0.5400 | 0.7013 | 0.6082 |
| Naive Bayes              | 0.9800   | 0.9800 | 0.9800    | 0.9800 | 0.9800 | 0.9600 |
| Random Forest (Ensemble) | 1.0000   | 1.0000 | 1.0000    | 1.0000 | 1.0000 | 1.0000 |
| XGBoost (Ensemble)       | 0.9900   | 1.0000 | 1.0000    | 0.9800 | 0.9899 | 0.9802 |

---

## Observations on Model Performance

| ML Model Name            | Observation about model performance |
| ------------------------ | ----------------------------------- |
| Logistic Regression      | Very high AUC but low recall. It misses many hazardous asteroids, meaning it is not reliable for detecting the minority class. |
| Decision Tree            | Achieved perfect performance on the dataset. It captured both hazardous and non-hazardous classes very effectively. |
| KNN                      | Moderate accuracy and recall. It performs reasonably well but struggles with identifying hazardous asteroids consistently. |
| Naive Bayes              | Very strong overall performance. It predicts both classes well and provides good accuracy and balanced results. |
| Random Forest (Ensemble) | Best performing model with perfect scores across all metrics. It provides excellent generalization and class separation. |
| XGBoost (Ensemble)       | Very strong model with excellent metrics. Slightly lower recall compared to Random Forest but still highly accurate. |

---

## Streamlit Web Application Features
The deployed Streamlit application includes:
- Upload option for test dataset (CSV)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix visualization
- Classification report output
- Prediction output table
- Download option for sample test dataset

Link - https://asteroid-hazard-prediction-35tk3sgu5ax6jtvhprci7v.streamlit.app/
---
