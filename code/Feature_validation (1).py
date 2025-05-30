# Databricks notebook source


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

scaler = StandardScaler()

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

from sklearn.feature_selection import mutual_info_classif
import pandas as pd

mi_scores = mutual_info_classif(X_train, y_train.iloc[:, 3], random_state=42)
mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

print("Top features by MI:")
print(mi_scores.head(10))

# COMMAND ----------

mi_scores = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)

print("Top features by MI:")
print(mi_scores.head(10))

# COMMAND ----------

from sklearn.feature_selection import RFE
model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=10)  # choose top 10 features

rfe.fit(X_train, y_train.iloc[:, 3])

# COMMAND ----------

# selected_features: list of feature names chosen by RFE
selected_features = [name for name, keep in zip(feature_names, rfe.support_) if keep]

# ranking: a Series showing the RFE rank of each feature
ranking = pd.Series(rfe.ranking_, index=feature_names)

print("Selected features by RFE:")
print(selected_features)
