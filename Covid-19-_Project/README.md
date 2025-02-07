# COVID-19 Patient Outcome Prediction

## Overview

This project analyzes a **COVID-19 dataset** provided by the Mexican government. The dataset contains **1,048,576 anonymized patient records** with **21 unique features** related to patient demographics, pre-existing conditions, and medical interventions. The goal of this project is to build a **predictive model** that classifies patients based on their COVID-19 test findings.

## Dataset Information

- **Source**: Mexican government ([Insert Dataset Link])
- **Number of Records**: 1,048,576 patients
- **Number of Features**: 21

### Features Description

- **sex**: 1 (Female), 2 (Male)
- **age**: Age of the patient
- **classification**: COVID test results (1-3: Positive, 4+: Negative/Inconclusive)
- **patient\_type**: 1 (Returned home), 2 (Hospitalized)
- **pneumonia**: Presence of pneumonia (1: Yes, 2: No)
- **pregnancy**: Pregnancy status (1: Yes, 2: No, 97: All men, 98: Missing)
- **diabetes, copd, asthma, inmsupr, hypertension, cardiovascular, renal\_chronic, other\_disease, obesity, tobacco**: Presence of pre-existing conditions (1: Yes, 2: No)
- **usmr**: Type of medical facility
- **medical\_unit**: Institution providing care
- **intubed**: If the patient was placed on a ventilator (1: Yes, 2: No, 97: Returned home, 99: Missing)
- **icu**: ICU admission (1: Yes, 2: No, 97: Returned home, 99: Missing)
- **date\_died**: Date of death (9999-99-99: Patient survived)

## Data Analysis & Preprocessing

After a detailed data inspection, the following issues were identified and handled:

1. **Incorrect missing value representation**:

   - Only **97** represents missing data in most features (except Pregnancy, where **98** is missing data).
   - Features with missing values: `intubed`, `pregnant`, `icu`

2. **Data Correction for INTUBED and ICU**:

   - **Patients who returned home (patient\_type = 1) cannot be in ICU or intubated.**
   - Therefore, **97 values in ICU and INTUBED were replaced with 2 (No)**.
   - **99 values in ICU and INTUBED were treated as missing data.**

3. **Encoding & Feature Engineering**:

   - One-hot encoding for categorical features.
   - Imputed missing values using an **iterative imputer**.
   - Scaled numerical features.

## Model Training & Evaluation

A classification model was trained to predict COVID-19 outcomes.

### Performance Metrics

- **Accuracy**: 93.95%

#### Confusion Matrix

```
 [[186821   7654]
 [  5021  10219]]
```

#### Classification Report

```
               precision    recall  f1-score   support

           0       0.97      0.96      0.97    194475
           1       0.57      0.67      0.62     15240

    accuracy                           0.94    209715
   macro avg       0.77      0.82      0.79    209715
weighted avg       0.94      0.94      0.94    209715
```

- **High accuracy (93.95%)** indicates a well-performing model.
- **Class imbalance observed** (COVID-negative patients dominate the dataset).
- **Further improvements needed** in predicting COVID-positive patients.

## Next Steps

- **Class Imbalance Handling**: Use **SMOTE** or **class weighting** to improve minority class predictions.
- **Feature Engineering**: Explore additional transformations.
- **Model Optimization**: Experiment with **XGBoost, Random Forest, or Neural Networks**.

## *How to Use This Repository*

1. **Go to my [GitHub](https://github.com/PrasadPandit-04/Data-Science) / [Kaggle]() repository where this project is stored.**
2. **Download the dataset** from one of the sources:
    - [Mexican Government Dataset](https://datos.gob.mx/busca/dataset/informacion-referente-a-casos-covid-19-en-mexico)
    - [Kaggle COVID-19 Dataset](https://www.kaggle.com/datasets/meirnizri/covid19-dataset)
3. **Run the notebook/script**:
    - If using Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
    - If using a Python script:
     ```bash
     python train_model.py
     ```
4. **Modify and improve the model** as needed.

## Acknowledgments

- **Mexican Government** for providing the dataset.
- **Machine Learning community** for best practices in model development.

---

This project is **open-source**, and contributions are welcome! Feel free to **fork, modify, and improve** the codebase.

## Contributions & Discussions

This project is open-source, and contributions are always welcome! 🎉

- Found a bug? **Open an issue!**
- Have an improvement in mind? **Submit a pull request!**
- Want to discuss ML approaches or suggest enhancements? **Start a discussion!**

## License

If you appreciate this project and want to support future work, consider buying me [☕](https://buymeacoffee.com/prasadpandp)... (or better, donating a [GPU](https://www.amazon.in/gp/cart/view.html?ref_=nav_cart) 😆).


