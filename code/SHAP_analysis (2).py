# Databricks notebook source
# MAGIC %md # 1 Setup data

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier  
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score,\
recall_score, auc, roc_auc_score, roc_curve, f1_score, RocCurveDisplay, plot_roc_curve

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW data AS
# MAGIC select 
# MAGIC    NHS_NUMBER_DEID,
# MAGIC          sex,
# MAGIC        IMD_1, IMD_2,
# MAGIC        IMD_3, IMD_4, IMD_5, IMD_Unknown,
# MAGIC        age18_29, age30_39, age40_49, age50_59, age60_69, age70_79,
# MAGIC        age80_89, age90_plus, severe_mental_illness,
# MAGIC          pre_covid_cvd_event,
# MAGIC          pre_covid_cvd_event_1y,
# MAGIC          cancer,
# MAGIC          alcohol_problems,
# MAGIC          hypertension,
# MAGIC          alcoholic_liver_disease,
# MAGIC          anti_coagulant_drugs,
# MAGIC          anti_diabetic_drugs,
# MAGIC          anti_platelet_drugs,
# MAGIC          anti_hypertensive_drugs,
# MAGIC          autoimmune_liver_disease,
# MAGIC          copd,
# MAGIC          dementia,
# MAGIC          diabetes,
# MAGIC          fracture_of_hip,
# MAGIC          fracture_of_wrist,
# MAGIC          obesity,
# MAGIC          osteoporosis,
# MAGIC          statins,
# MAGIC          schizophrenia,
# MAGIC          bipolardisorder,
# MAGIC          depression,
# MAGIC          antipsychotic,
# MAGIC          erectiledysfunction,
# MAGIC          CKD,
# MAGIC          AF,
# MAGIC          RA,
# MAGIC          smoking,
# MAGIC          
# MAGIC          covid_death,
# MAGIC          covid_hospitalisation,
# MAGIC          post_covid_cvd_event_30d, 
# MAGIC          post_covid_cvd_event_90d,
# MAGIC          post_covid_cvd_event_180d,
# MAGIC          post_covid_cvd_event_1y, 
# MAGIC          post_covid_cvd_event_2y, 
# MAGIC          VTE_outcome
# MAGIC from dsa_391419_j3w9t_collab.ccu037_cohort_binarised
# MAGIC where NHS_NUMBER_DEID NOT IN (SELECT NHS_NUMBER_DEID FROM dars_nic_391419_j3w9t_collab.ccu037_follow_up_availability where covid_date > to_date(reg_date_of_death, "yyyyMMdd") or covid_date < "2020-01-23")

# COMMAND ----------

cohort = spark.table("data")

# COMMAND ----------

cohort = cohort.toPandas()

# COMMAND ----------

# MAGIC %md # 2 Train test split

# COMMAND ----------

scaler = StandardScaler()

# COMMAND ----------

df_q = cohort
print("df_q.shape = ", df_q.shape)

# setting the index to NHS number
df_q.index = df_q.NHS_NUMBER_DEID

#drop columns including additional qrisk variables severe_mental_illness and qrisk_eth
X_q = df_q[['age18_29', 'age30_39','age40_49','age50_59','age60_69','age70_79','age80_89', 'age90_plus',
            "sex", "IMD_1", "IMD_2","IMD_3","IMD_4","IMD_5","IMD_Unknown",
            "smoking","AF","CKD","diabetes","RA","severe_mental_illness",
            "anti_hypertensive_drugs","antipsychotic","erectiledysfunction"]]
print("X_q.shape = ", X_q.shape)

print("Scaling features")
X_q = scaler.fit_transform(X_q)

# COMMAND ----------

df = cohort
print("df.shape = ", df.shape)

# setting the index to NHS number
df.index = df.NHS_NUMBER_DEID

#drop columns including additional qrisk variables severe_mental_illness and keeping qrisk_eth
X = df.drop(columns=['NHS_NUMBER_DEID','anti_diabetic_drugs','covid_death','covid_hospitalisation','post_covid_cvd_event_30d','post_covid_cvd_event_90d','post_covid_cvd_event_180d','post_covid_cvd_event_1y','post_covid_cvd_event_2y','VTE_outcome', "severe_mental_illness"])
print("X.shape = ", X.shape)

# feature names and indices
feature_names = X.columns
indices = list(df.index)

print("Scaling features")
X = scaler.fit_transform(X)

# COMMAND ----------

y = df[["post_covid_cvd_event_30d", "post_covid_cvd_event_90d", "post_covid_cvd_event_180d", "post_covid_cvd_event_1y", "post_covid_cvd_event_2y", "covid_death", "covid_hospitalisation"]]

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW indices AS
# MAGIC SELECT * FROM dsa_391419_j3w9t_collab.ccu037_global_cohort_predictions

# COMMAND ----------

indices = spark.table("indices")
indices = indices.toPandas()
test_ids = indices["NHS_NUMBER_DEID"]
train_ids = df['NHS_NUMBER_DEID'][~df['NHS_NUMBER_DEID'].isin(test_ids)]

Qselect_X_test = X_q[df_q["NHS_NUMBER_DEID"].isin(test_ids)]
Qselect_X_train = X_q[df_q["NHS_NUMBER_DEID"].isin(train_ids)]

X_test = X[df["NHS_NUMBER_DEID"].isin(test_ids)]
X_train = X[df["NHS_NUMBER_DEID"].isin(train_ids)]

y_test=y[df['NHS_NUMBER_DEID'].isin(test_ids)]
y_train=y[df['NHS_NUMBER_DEID'].isin(train_ids)]

# COMMAND ----------

print("Training class balance 0:", np.mean(y_train.iloc[:, 0]))
print("Testing class balance 0:", np.mean(y_test.iloc[:, 0]))
print("------------------------------")
print("Training class balance 1:", np.mean(y_train.iloc[:, 1]))
print("Testing class balance 1:", np.mean(y_test.iloc[:, 1]))
print("------------------------------")
print("Training class balance 2:", np.mean(y_train.iloc[:, 2]))
print("Testing class balance 2:", np.mean(y_test.iloc[:, 2]))
print("------------------------------")
print("Training class balance 3:", np.mean(y_train.iloc[:, 3]))
print("Testing class balance 3:", np.mean(y_test.iloc[:, 3]))
print("------------------------------")
print("Training class balance 4:", np.mean(y_train.iloc[:, 4]))
print("Testing class balance 4:", np.mean(y_test.iloc[:, 4]))
print("------------------------------")
print("Training class balance 5:", np.mean(y_train.iloc[:, 5]))
print("Testing class balance 5:", np.mean(y_test.iloc[:, 5]))
print("------------------------------")
print("Training class balance 6:", np.mean(y_train.iloc[:, 6]))
print("Testing class balance 6:", np.mean(y_test.iloc[:, 6]))

# COMMAND ----------

# MAGIC %md # 3 Feature selection

# COMMAND ----------

# MAGIC %md ## QRISK

# COMMAND ----------

Qselect_X_train = Qselect_X_train
Qselect_X_test = Qselect_X_test

# COMMAND ----------

# MAGIC %md ## LASSO

# COMMAND ----------

model3 = Lasso(alpha=0.000093)
model3.fit(X_train, y_train.iloc[:, 3])

# COMMAND ----------

coefficients3 = model3.coef_
importance3 = np.abs(coefficients3)
print(coefficients3)

# COMMAND ----------

print("Outcome 3",np.array(feature_names)[importance3 > 0])

# COMMAND ----------

Lselect_X_train3 = X_train[:, importance3 != 0]
Lselect_X_test3 = X_test[:, importance3 != 0]
print("outcome 3 shape:", Lselect_X_train3.shape)

# COMMAND ----------

feature_names = list(feature_names)

# COMMAND ----------

# 1. Boolean mask from LASSO (already done)
selected_mask = importance3 != 0  # This is a boolean array

# 2. Get feature names from the original list using the mask
L_select = [name for name, keep in zip(feature_names, selected_mask) if keep]

# COMMAND ----------

# MAGIC %md ## Random Forest

# COMMAND ----------

RF_features = ['sex', 'age80_89', 'pre_covid_cvd_event', 'pre_covid_cvd_event_1y',
       'cancer', 'hypertension', 'anti_coagulant_drugs', 'anti_platelet_drugs',
       'copd', 'diabetes', 'obesity', 'osteoporosis', 'statins', 'CKD', 'AF',
       'smoking']
RF_selected = [feature_names.index(i) for i in RF_features]
RFselect_X_test3 = X_test[:, RF_selected]
RFselect_X_train3 = X_train[:, RF_selected]

# COMMAND ----------

# MAGIC %md ## XGB

# COMMAND ----------

XGB_features = ['age18_29', 'age30_39', 'age40_49', 'age70_79', 'pre_covid_cvd_event',
       'pre_covid_cvd_event_1y']
XGB_selected = [feature_names.index(i) for i in XGB_features]
Xselect_X_test3 = X_test[:, XGB_selected]
Xselect_X_train3 = X_train[:, XGB_selected]

# COMMAND ----------

# MAGIC %md # 4 Run models

# COMMAND ----------

Q_LR3 = LogisticRegression(max_iter=10000)
Q_LR3.fit(Qselect_X_train, y_train.iloc[:, 3])
print("Outcome 3:", Q_LR3.coef_)

L_LR3 = LogisticRegression(max_iter=10000)
L_LR3.fit(Lselect_X_train3, y_train.iloc[:, 3])
print("Outcome 3:", L_LR3.coef_)

RF_LR3 = LogisticRegression(max_iter=10000)
RF_LR3.fit(RFselect_X_train3, y_train.iloc[:, 3])
print("Outcome 3:", RF_LR3.coef_)

X_LR3 = LogisticRegression(max_iter=10000)
X_LR3.fit(Xselect_X_train3, y_train.iloc[:, 3])
print("Outcome 3:", X_LR3.coef_)

# COMMAND ----------

# MAGIC %md ## Plot

# COMMAND ----------

feature_name_map = {'sex': 'Sex',
                    'IMD_1': 'IMD 1', 
                    'IMD_2': 'IMD 2', 
                    'IMD_3': 'IMD 3', 
                    'IMD_4': 'IMD 4', 
                    'IMD_5': 'IMD 5',
                    'IMD_Unknown': 'IMD Unknown', 
                    'age18_29': 'Age 18-29', 
                    'age30_39': 'Age 30-39', 
                    'age40_49': 'Age 40-49', 
                    'age50_59': 'Age 50-59',
                    'age60_69': 'Age 60-69', 
                    'age70_79': 'Age 70-79', 
                    'age80_89': 'Age 80-89', 
                    'age90_plus': 'Age 90+',
                    'severe_mental_illness': 'Severe Mental Illness', 
                    'pre_covid_cvd_event': 'Pre-COVID CVD Event',
                    'pre_covid_cvd_event_1y': 'Pre-COVID CVD Event 1y', 
                    'cancer': 'Cancer', 
                    'alcohol_problems': 'Alcohol Problems', 
                    'hypertension': 'Hypertension',
                    'alcoholic_liver_disease': 'Alcoholic Liver Disease', 
                    'anti_coagulant_drugs': 'Anti-Coagulant Drugs',
                    'anti_diabetic_drugs': 'Anti-Diabetic Drugs', 
                    'anti_platelet_drugs': 'Anti-Platelet Drugs', 
                    'anti_hypertensive_drugs': 'Anti-Hypertensive Drugs',
                    'autoimmune_liver_disease': 'Autoimmune Liver Disease', 
                    'copd': 'COPD', 
                    'dementia': 'Dementia', 
                    'diabetes': 'Diabetes',
                    'fracture_of_hip': 'Fracture of Hip', 
                    'fracture_of_wrist': 'Fracture of Wrist', 
                    'obesity': 'Obesity',
                    'osteoporosis': 'Osteoporosis',
                    'statins': 'Statins', 
                    'schizophrenia': 'Schizophrenia', 
                    'bipolardisorder': 'Bipolar Disorder', 
                    'depression': 'Depression',
                    'antipsychotic': 'Anti-Psychotic Drugs', 
                    'erectiledysfunction': 'Erectile Dysfunction', 
                    'CKD': 'CKD', 
                    'AF': 'AF', 
                    'RA': 'RA', 
                    'smoking': 'Smoking'}

# COMMAND ----------

Lselect_clean = [feature_name_map.get(f, f) for f in L_select]
RFselect_clean = [feature_name_map.get(f, f) for f in RF_features]
Xselect_clean = [feature_name_map.get(f, f) for f in XGB_features]

# COMMAND ----------

import shap
explainer = shap.LinearExplainer(L_LR3, Lselect_X_train3)
shap_values = explainer.shap_values(Lselect_X_train3)
shap.summary_plot(shap_values,Lselect_X_train3, feature_names=Lselect_clean)

# COMMAND ----------

explainer = shap.LinearExplainer(X_LR3, Xselect_X_train3)
shap_values = explainer.shap_values(Xselect_X_train3)
shap.summary_plot(shap_values,Xselect_X_train3, feature_names=Xselect_clean)

# COMMAND ----------

from matplotlib.backends.backend_pdf import PdfPages
import os

pdf_path = r"/tmp/shap_plots_global.pdf"

# Prepare the PDF to save the plots
  # Adjust path if needed
pdf_pages = PdfPages(pdf_path)

# List of models, data, and clean feature names
models = [L_LR3, RF_LR3, X_LR3]
X_data = [Lselect_X_train3, RFselect_X_train3, Xselect_X_train3]
clean_feature_names = [Lselect_clean, RFselect_clean, Xselect_clean]

# Custom titles for the plots
plot_titles = ["LASSO + LR", "RF + LR", "XGB + LR"]

# Loop through each model and create plots
for model, X, clean_names, title in zip(models, X_data, clean_feature_names, plot_titles):
    
    # Create SHAP explainer for the model and data
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Set cleaned feature names
    shap_values.feature_names = clean_names

    # 1. SHAP Summary Plot (Beeswarm)
    plt.figure(figsize=(8.27, 11.69))
    shap.summary_plot(shap_values, show=False)  # Don't show immediately
    plt.title(f"SHAP Summary Plot - {title}")
    plt.tight_layout()
    pdf_pages.savefig()  # Save to PDF
    plt.close()

    # 2. SHAP Bar Plot (mean(|SHAP|))
    plt.figure(figsize=(8.27, 11.69))
    shap.summary_plot(shap_values, plot_type="bar", show=False)  # Bar plot
    plt.title(f"SHAP Bar Plot - {title}")
    plt.tight_layout()
    pdf_pages.savefig()  # Save to PDF
    plt.close()

# Close the PDF file
pdf_pages.close()


# COMMAND ----------

import base64 

with open(pdf_path, "rb") as f:
    data = f.read()
    b64 = base64.b64encode(data).decode()

html = f"""
<a download=shap_plots_global.pdf" href="data:application/txt;base64,{b64}" target="_blank">
  ðŸ‘‰ Click here to download your file
</a>
"""

displayHTML(html)
