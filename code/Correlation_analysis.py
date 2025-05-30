# Databricks notebook source
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

features = cohort.drop(columns=['covid_death','covid_hospitalisation','post_covid_cvd_event_30d','post_covid_cvd_event_90d','post_covid_cvd_event_180d','post_covid_cvd_event_1y','post_covid_cvd_event_2y', 'VTE_outcome'])

# COMMAND ----------

# Select numeric columns
numeric_cohort = features.loc[:, [col for col in features.columns if col.startswith('numeric_') or features[col].dtype.kind in ['i', 'f']]]

# Calculate correlation matrix
corr_matrix = numeric_cohort.corr()

# Display heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# COMMAND ----------

plt.figure(figsize=(35, 30))
sns.heatmap(corr_matrix, annot=True, cmap='crest')
plt.title("Correlation Matrix")
plt.show()

# COMMAND ----------

# Convert correlation matrix to a DataFrame
corr_df = corr_matrix.reset_index()
corr_df = corr_df.melt(id_vars='index', var_name='column', value_name='correlation')

# Create a temporary view
corr_df = spark.createDataFrame(corr_df)
corr_df.createOrReplaceTempView('correlation_matrix')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM correlation_matrix
