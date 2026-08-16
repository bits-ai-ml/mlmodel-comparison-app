# ML Assignment 2 — Classification Model Comparison App

**Name:** Akhil Jain &nbsp;|&nbsp; **BITS ID:** 2025ac05543 &nbsp;|&nbsp; **Course:** Machine Learning (AIMLCZG565)

## a. Problem Statement

Implement multiple classification models on a single dataset, compare them using standard
evaluation metrics, and expose the trained models through an interactive Streamlit web app
that lets a user upload test data, pick a model, and view its predictions, metrics, and
confusion matrix — deployed publicly on Streamlit Community Cloud.

## b. Dataset Description

**Breast Cancer Wisconsin (Diagnostic) Dataset** — a binary classification dataset
(malignant vs benign tumor), originally from the UCI Machine Learning Repository, loaded via
`sklearn.datasets.load_breast_cancer()`.

- **Instances:** 569 (≥ 500 required)
- **Features:** 30 numeric features computed from digitized images of a breast mass fine
  needle aspirate — each of 10 base measurements (radius, texture, perimeter, area,
  smoothness, compactness, concavity, concave points, symmetry, fractal dimension) reported
  as mean, standard error, and "worst" value → 30 features total (≥ 12 required)
- **Target:** binary — `0 = malignant` (212 cases), `1 = benign` (357 cases)
- **Why this dataset:** the combined UCI Heart Disease sources either fall short of the
  500-instance minimum (Cleveland-only subset: 303 rows) or the 12-feature minimum (the
  5-database combined version only shares 11 common columns). This dataset comfortably
  satisfies both requirements and loads without any external download, which also makes it
  reliable to run on the BITS Virtual Lab.

## c. GitHub Repository Link

`<PASTE_YOUR_GITHUB_REPO_URL_HERE>`

*(e.g. https://github.com/bits-ai-ml/ml-assignment-2-akhil-jain — repo must contain
source code, requirements.txt, this README, test_data.csv, and the model/ folder)*

## d. Models Used

All 5 models were trained on the same 80/20 stratified train/test split of the dataset above.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9932 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest accuracy (0.9825) and AUC (0.9954). The 30 features are strongly linearly separable after standardization, so a linear decision boundary generalizes very cleanly on the held-out test set. |
| Decision Tree | Weakest model (accuracy 0.9123, MCC 0.8174). A single unconstrained tree overfits to noise in training data — without pruning/depth limits it captures spurious splits that don't generalize, showing up as more false negatives. |
| kNN | Strong performer (accuracy 0.9737, Recall 1.0000). After scaling, benign and malignant cases form fairly tight, well-separated clusters, so distance-based voting works very well. Perfect recall means every benign case was caught; slightly lower precision means a few malignant cases were misclassified as benign by nearby neighbors. |
| Naive Bayes | Trailed the top models (accuracy 0.9298) despite a good AUC (0.9868). Its Gaussian independence assumption is a simplification — many of the 30 features (e.g. mean radius and mean perimeter) are highly correlated, which Naive Bayes cannot model. |
| Random Forest (Ensemble) | Strong, well-balanced performer (accuracy 0.9561, MCC 0.9054). Bagging many trees fixes the overfitting seen in the single Decision Tree, though it didn't quite match Logistic Regression / kNN on this particular split. |
| **Overall Winner for your dataset?** | **Logistic Regression** — best accuracy, AUC, and MCC. Indicates the two classes are close to linearly separable once scaled, so the extra complexity of tree ensembles or non-parametric methods isn't needed for this dataset. |

## Repository Structure

```
project-folder/
│-- app.py                  # Streamlit app
│-- requirements.txt
│-- README.md                # this file
│-- test_data.csv            # held-out 20% test split (features + target)
│-- model/
│   │-- model_training.ipynb # full training notebook (EDA, training, evaluation)
│   │-- scaler.pkl
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest_ensemble.pkl
```

## Streamlit App Features

1. **Dataset upload (CSV)** — upload `test_data.csv` from the sidebar
2. **Model selection dropdown** — choose any of the 5 trained models
3. **Evaluation metrics display** — Accuracy, AUC, Precision, Recall, F1, MCC (shown when the
   uploaded CSV has a `target` column)
4. **Confusion matrix + classification report** — per-class breakdown for the selected model

## Live Links (fill in after deployment)

- **GitHub Repository:** `<PASTE_YOUR_GITHUB_REPO_URL_HERE>`
- **Live Streamlit App:** `<PASTE_YOUR_STREAMLIT_APP_URL_HERE>`
