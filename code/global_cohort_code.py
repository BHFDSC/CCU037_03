# Databricks notebook source
# MAGIC %md
# MAGIC #1. Import libraries, run starter commands

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

# MAGIC %md
# MAGIC # 2. Define cohort

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW data AS
# MAGIC select 
# MAGIC    NHS_NUMBER_DEID,
# MAGIC          sex,
# MAGIC          Bangladeshi,
# MAGIC        Black_African, Black_Caribbean, Chinese, Indian, Other_Asian,
# MAGIC        Other_Ethnic_Group, Pakistani, White_or_not_stated,
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
# MAGIC WHERE NHS_NUMBER_DEID NOT IN (SELECT NHS_NUMBER_DEID FROM dars_nic_391419_j3w9t_collab.ccu037_follow_up_availability where covid_date > to_date(reg_date_of_death, "yyyyMMdd") or covid_date < "2020-01-23")

# COMMAND ----------

cohort = spark.table("data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Check data types

# COMMAND ----------

cohort.dtypes

# COMMAND ----------

cohort = cohort.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Train test split

# COMMAND ----------

scaler = StandardScaler()
test_size = 0.2
seed = 42

# COMMAND ----------

# MAGIC %md
# MAGIC ### QRISK

# COMMAND ----------

cohort.columns

# COMMAND ----------

df_q = cohort
print("df_q.shape = ", df_q.shape)

# setting the index to NHS number
df_q.index = df_q.NHS_NUMBER_DEID

#drop columns including additional qrisk variables severe_mental_illness and qrisk_eth
X_q = df_q[['age18_29', 'age30_39','age40_49','age50_59','age60_69','age70_79','age80_89', 'age90_plus',
            "sex", "IMD_1", "IMD_2","IMD_3","IMD_4","IMD_5","IMD_Unknown",
            "smoking","AF","CKD","diabetes","RA","severe_mental_illness",
            "anti_hypertensive_drugs","antipsychotic","erectiledysfunction",
            'Bangladeshi', 'Black_African', 'Black_Caribbean', 'Chinese', 'Indian', 'Other_Asian',
            'Other_Ethnic_Group', 'Pakistani', 'White_or_not_stated']]
print("X_q.shape = ", X_q.shape)

print("Scaling features")
X_q = scaler.fit_transform(X_q)

# COMMAND ----------

# outcome 0: 30 day cvd event
y_q = df_q["post_covid_cvd_event_30d"]
Qselect_X_train, Qselect_X_test, Qy_train0, Qy_test = train_test_split(X_q, y_q, test_size=0.2, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Overall dataset

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
y_train = pd.DataFrame()
y_test = pd.DataFrame()

for column in y:
  X_train, X_test, y_train[column], y_test[column], indices_train, indices_test = train_test_split(X, y[column], indices, test_size=0.2, random_state=42)
  y_train[column] = y_train[column]
  y_test[column] = y_test[column]
  
y_train
# outcome 0: 30 day cvd event
# outcome 1: 90 day cvd event
# outcome 2: 180 day cvd event
# outcome 3: 1 year cvd event
# outcome 4: 2 year cvd event
# outcome 5: covid death event
# outcome 6: covid_hospitalisation

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

# MAGIC %md
# MAGIC #4. Feature selection

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.0. QRISK

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select

# COMMAND ----------

Qselect_X_train = Qselect_X_train
Qselect_X_test = Qselect_X_test

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1. Lasso

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hyperparameter tuning

# COMMAND ----------

y_train.isnull().sum()

# COMMAND ----------

#FREYA CODE:
from numpy import arange
from pandas import read_csv
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectFromModel
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
  
model0 = LassoCV(cv=cv, n_jobs=-1)
model0.fit(X_train, y_train.iloc[:, 0])
print('alpha0: %f' % model0.alpha_)

model1 = LassoCV(cv=cv, n_jobs=-1)
model1.fit(X_train, y_train.iloc[:, 1])
print('alpha1: %f' % model1.alpha_)

model2 = LassoCV(cv=cv, n_jobs=-1)
model2.fit(X_train, y_train.iloc[:, 2])
print('alpha2: %f' % model2.alpha_)

model3 = LassoCV(cv=cv, n_jobs=-1)
model3.fit(X_train, y_train.iloc[:, 3])
print('alpha3: %f' % model3.alpha_)

model4 = LassoCV(cv=cv, n_jobs=-1)
model4.fit(X_train, y_train.iloc[:, 4])
print('alpha4: %f' % model4.alpha_)

model5 = LassoCV(cv=cv, n_jobs=-1)
model5.fit(X_train, y_train.iloc[:, 5])
print('alpha5: %f' % model5.alpha_)

model6 = LassoCV(cv=cv, n_jobs=-1)
model6.fit(X_train, y_train.iloc[:, 6])
print('alpha6: %f' % model6.alpha_)

# COMMAND ----------

coefficients0 = model0.coef_
importance0 = np.abs(coefficients0)
print(coefficients0)

coefficients1 = model1.coef_
importance1 = np.abs(coefficients1)
print(coefficients1)

coefficients2 = model2.coef_
importance2 = np.abs(coefficients2)
print(coefficients2)

coefficients3 = model3.coef_
importance3 = np.abs(coefficients3)
print(coefficients3)

coefficients4 = model4.coef_
importance4 = np.abs(coefficients4)
print(coefficients4)

coefficients5 = model5.coef_
importance5 = np.abs(coefficients5)
print(coefficients5)

coefficients6 = model6.coef_
importance6 = np.abs(coefficients6)
print(coefficients6)

# COMMAND ----------

print("Feature imporances that are greater than 0:")
print("Outcome 0",np.array(feature_names)[importance0 > 0])
print("Outcome 1",np.array(feature_names)[importance1 > 0])
print("Outcome 2",np.array(feature_names)[importance2 > 0])
print("Outcome 3",np.array(feature_names)[importance3 > 0])
print("Outcome 4",np.array(feature_names)[importance4 > 0])
print("Outcome 5",np.array(feature_names)[importance5 > 0])
print("Outcome 6",np.array(feature_names)[importance6 > 0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select

# COMMAND ----------

# selecting the non zero features for each outcome
Lselect_X_train0 = X_train[:, importance0 != 0]
Lselect_X_test0 = X_test[:, importance0 != 0]
print("outcome 0 shape:", Lselect_X_train0.shape)
Lselect_X_train1 = X_train[:, importance1 != 0]
Lselect_X_test1 = X_test[:, importance1 != 0]
print("outcome 1 shape:", Lselect_X_train1.shape)
Lselect_X_train2 = X_train[:, importance2 != 0]
Lselect_X_test2 = X_test[:, importance2 != 0]
print("outcome 2 shape:", Lselect_X_train2.shape)
Lselect_X_train3 = X_train[:, importance3 != 0]
Lselect_X_test3 = X_test[:, importance3 != 0]
print("outcome 3 shape:", Lselect_X_train3.shape)
Lselect_X_train4 = X_train[:, importance4 != 0]
Lselect_X_test4 = X_test[:, importance4 != 0]
print("outcome 4 shape:", Lselect_X_train4.shape)
Lselect_X_train5 = X_train[:, importance5 != 0]
Lselect_X_test5 = X_test[:, importance5 != 0]
print("outcome 5 shape:", Lselect_X_train5.shape)
Lselect_X_train6 = X_train[:, importance6 != 0]
Lselect_X_test6 = X_test[:, importance6 != 0]
print("outcome 6 shape:", Lselect_X_train6.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ##4.2. RF feature selection

# COMMAND ----------

rf0 = RandomForestClassifier(random_state = 42)
rf0.fit(X_train, y_train.iloc[:, 0])

rf1 = RandomForestClassifier(random_state = 42)
rf1.fit(X_train, y_train.iloc[:, 1])

rf2 = RandomForestClassifier(random_state = 42)
rf2.fit(X_train, y_train.iloc[:, 2])

rf3 = RandomForestClassifier(random_state = 42)
rf3.fit(X_train, y_train.iloc[:, 3])

rf4 = RandomForestClassifier(random_state = 42)
rf4.fit(X_train, y_train.iloc[:, 4])

rf5 = RandomForestClassifier(random_state = 42)
rf5.fit(X_train, y_train.iloc[:, 5])

rf6 = RandomForestClassifier(random_state = 42)
rf6.fit(X_train, y_train.iloc[:, 6])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select

# COMMAND ----------

sel0 = SelectFromModel(rf0)
sel0.fit(X_train, y_train.iloc[:, 0])
print("Outcome0 features:", feature_names[(sel0.get_support())])
print("-------------------------------------")

sel1 = SelectFromModel(rf1)
sel1.fit(X_train, y_train.iloc[:, 1])
print("Outcome1 features:", feature_names[(sel1.get_support())])
print("-------------------------------------")

sel2 = SelectFromModel(rf2)
sel2.fit(X_train, y_train.iloc[:, 2])
print("Outcome2 features:", feature_names[(sel2.get_support())])
print("-------------------------------------")

sel3 = SelectFromModel(rf3)
sel3.fit(X_train, y_train.iloc[:, 3])
print("Outcome3 features:", feature_names[(sel3.get_support())])
print("-------------------------------------")

sel4 = SelectFromModel(rf4)
sel4.fit(X_train, y_train.iloc[:, 4])
print("Outcome4 features:", feature_names[(sel4.get_support())])
print("-------------------------------------")

sel5 = SelectFromModel(rf5)
sel5.fit(X_train, y_train.iloc[:, 5])
print("Outcome5 features:", feature_names[(sel5.get_support())])
print("-------------------------------------")

sel6 = SelectFromModel(rf6)
sel6.fit(X_train, y_train.iloc[:, 6])
print("Outcome6 features:", feature_names[(sel6.get_support())])
print("-------------------------------------")


# COMMAND ----------

RFselect_X_test0 = sel0.transform(X_test)
RFselect_X_train0 = sel0.transform(X_train)

RFselect_X_test1 = sel1.transform(X_test)
RFselect_X_train1 = sel1.transform(X_train)

RFselect_X_test2 = sel2.transform(X_test)
RFselect_X_train2 = sel2.transform(X_train)

RFselect_X_test3 = sel3.transform(X_test)
RFselect_X_train3 = sel3.transform(X_train)

RFselect_X_test4 = sel4.transform(X_test)
RFselect_X_train4 = sel4.transform(X_train)

RFselect_X_test5 = sel5.transform(X_test)
RFselect_X_train5 = sel5.transform(X_train)

RFselect_X_test6 = sel6.transform(X_test)
RFselect_X_train6 = sel6.transform(X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ##4.3. XGB feature selection

# COMMAND ----------

xgb0 = xgb.XGBClassifier(random_state = 42)
xgb0.fit(X_train, y_train.iloc[:, 0])

xgb1 = xgb.XGBClassifier(random_state = 42)
xgb1.fit(X_train, y_train.iloc[:, 1])

xgb2 = xgb.XGBClassifier(random_state = 42)
xgb2.fit(X_train, y_train.iloc[:, 2])

xgb3 = xgb.XGBClassifier(random_state = 42)
xgb3.fit(X_train, y_train.iloc[:, 3])

xgb4 = xgb.XGBClassifier(random_state = 42)
xgb4.fit(X_train, y_train.iloc[:, 4])

xgb5 = xgb.XGBClassifier(random_state = 42)
xgb5.fit(X_train, y_train.iloc[:, 5])

xgb6 = xgb.XGBClassifier(random_state = 42)
xgb6.fit(X_train, y_train.iloc[:, 6])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select

# COMMAND ----------

xsel0 = SelectFromModel(xgb0)
xsel0.fit(X_train, y_train.iloc[:, 0])
print("Outcome0 features:", feature_names[(xsel0.get_support())])
print("-------------------------------------")

xsel1 = SelectFromModel(xgb1)
xsel1.fit(X_train, y_train.iloc[:, 1])
print("Outcome1 features:", feature_names[(xsel1.get_support())])
print("-------------------------------------")

xsel2 = SelectFromModel(xgb2)
xsel2.fit(X_train, y_train.iloc[:, 2])
print("Outcome2 features:", feature_names[(xsel2.get_support())])
print("-------------------------------------")

xsel3 = SelectFromModel(xgb3)
xsel3.fit(X_train, y_train.iloc[:, 3])
print("Outcome3 features:", feature_names[(xsel3.get_support())])
print("-------------------------------------")

xsel4 = SelectFromModel(xgb4)
xsel4.fit(X_train, y_train.iloc[:, 4])
print("Outcome4 features:", feature_names[(xsel4.get_support())])
print("-------------------------------------")

xsel5 = SelectFromModel(xgb5)
xsel5.fit(X_train, y_train.iloc[:, 5])
print("Outcome5 features:", feature_names[(xsel5.get_support())])
print("-------------------------------------")

xsel6 = SelectFromModel(xgb6)
xsel6.fit(X_train, y_train.iloc[:, 6])
print("Outcome6 features:", feature_names[(xsel6.get_support())])
print("-------------------------------------")

# COMMAND ----------

Xselect_X_test0 = xsel0.transform(X_test)
Xselect_X_train0 = xsel0.transform(X_train)

Xselect_X_test1 = xsel1.transform(X_test)
Xselect_X_train1 = xsel1.transform(X_train)

Xselect_X_test2 = xsel2.transform(X_test)
Xselect_X_train2 = xsel2.transform(X_train)

Xselect_X_test3 = xsel3.transform(X_test)
Xselect_X_train3 = xsel3.transform(X_train)

Xselect_X_test4 = xsel4.transform(X_test)
Xselect_X_train4 = xsel4.transform(X_train)

Xselect_X_test5 = xsel5.transform(X_test)
Xselect_X_train5 = xsel5.transform(X_train)

Xselect_X_test6 = xsel6.transform(X_test)
Xselect_X_train6 = xsel6.transform(X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Run models

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.0. QRISK + Logistic Regression

# COMMAND ----------

Q_LR0 = LogisticRegression(max_iter=10000)
Q_LR0.fit(Qselect_X_train, y_train.iloc[:, 0])
print("Outcome 0:", Q_LR0.coef_)
Q_LR1 = LogisticRegression(max_iter=10000)
Q_LR1.fit(Qselect_X_train, y_train.iloc[:, 1])
print("Outcome 1:", Q_LR1.coef_)
Q_LR2 = LogisticRegression(max_iter=10000)
Q_LR2.fit(Qselect_X_train, y_train.iloc[:, 2])
print("Outcome 2:", Q_LR2.coef_)
Q_LR3 = LogisticRegression(max_iter=10000)
Q_LR3.fit(Qselect_X_train, y_train.iloc[:, 3])
print("Outcome 3:", Q_LR3.coef_)
Q_LR4 = LogisticRegression(max_iter=10000)
Q_LR4.fit(Qselect_X_train, y_train.iloc[:, 4])
print("Outcome 4:", Q_LR4.coef_)
Q_LR5 = LogisticRegression(max_iter=10000)
Q_LR5.fit(Qselect_X_train, y_train.iloc[:, 5])
print("Outcome 5:", Q_LR5.coef_)
Q_LR6 = LogisticRegression(max_iter=10000)
Q_LR6.fit(Qselect_X_train, y_train.iloc[:, 6])
print("Outcome 6:", Q_LR6.coef_)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.1. Lasso + Logistic Regression

# COMMAND ----------

L_LR0 = LogisticRegression(max_iter=10000)
L_LR0.fit(Lselect_X_train0, y_train.iloc[:, 0])
print("Outcome 0:", L_LR0.coef_)
L_LR1 = LogisticRegression(max_iter=10000)
L_LR1.fit(Lselect_X_train1, y_train.iloc[:, 1])
print("Outcome 1:", L_LR1.coef_)
L_LR2 = LogisticRegression(max_iter=10000)
L_LR2.fit(Lselect_X_train2, y_train.iloc[:, 2])
print("Outcome 2:", L_LR2.coef_)
L_LR3 = LogisticRegression(max_iter=10000)
L_LR3.fit(Lselect_X_train3, y_train.iloc[:, 3])
print("Outcome 3:", L_LR3.coef_)
L_LR4 = LogisticRegression(max_iter=10000)
L_LR4.fit(Lselect_X_train4, y_train.iloc[:, 4])
print("Outcome 4:", L_LR4.coef_)
L_LR5 = LogisticRegression(max_iter=10000)
L_LR5.fit(Lselect_X_train5, y_train.iloc[:, 5])
print("Outcome 5:", L_LR5.coef_)
L_LR6 = LogisticRegression(max_iter=10000)
L_LR6.fit(Lselect_X_train6, y_train.iloc[:, 6])
print("Outcome 6:", L_LR6.coef_)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.2. RF + Logistic Regression

# COMMAND ----------

RF_LR0 = LogisticRegression(max_iter=10000)
RF_LR0.fit(RFselect_X_train0, y_train.iloc[:, 0])
print("Outcome 0:", RF_LR0.coef_)
RF_LR1 = LogisticRegression(max_iter=10000)
RF_LR1.fit(RFselect_X_train1, y_train.iloc[:, 1])
print("Outcome 1:", RF_LR1.coef_)
RF_LR2 = LogisticRegression(max_iter=10000)
RF_LR2.fit(RFselect_X_train2, y_train.iloc[:, 2])
print("Outcome 2:", RF_LR2.coef_)
RF_LR3 = LogisticRegression(max_iter=10000)
RF_LR3.fit(RFselect_X_train3, y_train.iloc[:, 3])
print("Outcome 3:", RF_LR3.coef_)
RF_LR4 = LogisticRegression(max_iter=10000)
RF_LR4.fit(RFselect_X_train4, y_train.iloc[:, 4])
print("Outcome 4:", RF_LR4.coef_)
RF_LR5 = LogisticRegression(max_iter=10000)
RF_LR5.fit(RFselect_X_train5, y_train.iloc[:, 5])
print("Outcome 5:", RF_LR5.coef_)
RF_LR6 = LogisticRegression(max_iter=10000)
RF_LR6.fit(RFselect_X_train6, y_train.iloc[:, 6])
print("Outcome 6:", RF_LR6.coef_)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.3. XGB + Logistic Regression

# COMMAND ----------

X_LR0 = LogisticRegression(max_iter=10000)
X_LR0.fit(Xselect_X_train0, y_train.iloc[:, 0])
print("Outcome 0:", X_LR0.coef_)
X_LR1 = LogisticRegression(max_iter=10000)
X_LR1.fit(Xselect_X_train1, y_train.iloc[:, 1])
print("Outcome 1:", X_LR1.coef_)
X_LR2 = LogisticRegression(max_iter=10000)
X_LR2.fit(Xselect_X_train2, y_train.iloc[:, 2])
print("Outcome 2:", X_LR2.coef_)
X_LR3 = LogisticRegression(max_iter=10000)
X_LR3.fit(Xselect_X_train3, y_train.iloc[:, 3])
print("Outcome 3:", X_LR3.coef_)
X_LR4 = LogisticRegression(max_iter=10000)
X_LR4.fit(Xselect_X_train4, y_train.iloc[:, 4])
print("Outcome 4:", X_LR4.coef_)
X_LR5 = LogisticRegression(max_iter=10000)
X_LR5.fit(Xselect_X_train5, y_train.iloc[:, 5])
print("Outcome 5:", X_LR5.coef_)
X_LR6 = LogisticRegression(max_iter=10000)
X_LR6.fit(Xselect_X_train6, y_train.iloc[:, 6])
print("Outcome 6:", X_LR6.coef_)

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Analysis

# COMMAND ----------

# setting up a dataframe to store prediction information
df_probabilities = pd.DataFrame(columns=['NHS_NUMBER_DEID',
                                          'CVD30d_pred_lr','CVD30d_pred_lasso', 'CVD30d_pred_xgb', 'CVD30d_pred_rf',
                                          'CVD90d_pred_lr','CVD90d_pred_lasso', 'CVD90d_pred_xgb', 'CVD90d_pred_rf',
                                          'CVD180d_pred_lr','CVD180d_pred_lasso', 'CVD180d_pred_xgb', 'CVD180d_pred_rf',
                                          'CVD1y_pred_lr','CVD1y_pred_lasso', 'CVD1y_pred_xgb', 'CVD1y_pred_rf',
                                          'CVD2y_pred_lr','CVD2y_pred_lasso', 'CVD2y_pred_xgb', 'CVD2y_pred_rf',
                                          'covid_death_pred_lr','covid_death_pred_lasso', 'covid_death_pred_xgb', 'covid_death_pred_rf',
                                          'covid_hosp_pred_lr','covid_hosp_pred_lasso', 'covid_hosp_pred_xgb', 'covid_hosp_pred_rf'])
                                    

# COMMAND ----------

df_probabilities['NHS_NUMBER_DEID']=pd.Series(indices_test)

df_probabilities['CVD30d_pred_lr']=Q_LR0.predict_proba(Qselect_X_test)[:,1]
df_probabilities['CVD30d_pred_lasso']=L_LR0.predict_proba(Lselect_X_test0)[:,1]
df_probabilities['CVD30d_pred_xgb']=X_LR0.predict_proba(Xselect_X_test0)[:,1]
df_probabilities['CVD30d_pred_rf']=RF_LR0.predict_proba(RFselect_X_test0)[:,1]

df_probabilities['CVD90d_pred_lr']=Q_LR1.predict_proba(Qselect_X_test)[:,1]
df_probabilities['CVD90d_pred_lasso']=L_LR1.predict_proba(Lselect_X_test1)[:,1]
df_probabilities['CVD90d_pred_xgb']=X_LR1.predict_proba(Xselect_X_test1)[:,1]
df_probabilities['CVD90d_pred_rf']=RF_LR1.predict_proba(RFselect_X_test1)[:,1]

df_probabilities['CVD180d_pred_lr']=Q_LR2.predict_proba(Qselect_X_test)[:,1]
df_probabilities['CVD180d_pred_lasso']=L_LR2.predict_proba(Lselect_X_test2)[:,1]
df_probabilities['CVD180d_pred_xgb']=X_LR2.predict_proba(Xselect_X_test2)[:,1]
df_probabilities['CVD180d_pred_rf']=RF_LR2.predict_proba(RFselect_X_test2)[:,1]

df_probabilities['CVD1y_pred_lr']=Q_LR3.predict_proba(Qselect_X_test)[:,1]
df_probabilities['CVD1y_pred_lasso']=L_LR3.predict_proba(Lselect_X_test3)[:,1]
df_probabilities['CVD1y_pred_xgb']=X_LR3.predict_proba(Xselect_X_test3)[:,1]
df_probabilities['CVD1y_pred_rf']=RF_LR3.predict_proba(RFselect_X_test3)[:,1]

df_probabilities['CVD2y_pred_lr']=Q_LR4.predict_proba(Qselect_X_test)[:,1]
df_probabilities['CVD2y_pred_lasso']=L_LR4.predict_proba(Lselect_X_test4)[:,1]
df_probabilities['CVD2y_pred_xgb']=X_LR4.predict_proba(Xselect_X_test4)[:,1]
df_probabilities['CVD2y_pred_rf']=RF_LR4.predict_proba(RFselect_X_test4)[:,1]

df_probabilities['covid_death_pred_lr']=Q_LR5.predict_proba(Qselect_X_test)[:,1]
df_probabilities['covid_death_pred_lasso']=L_LR5.predict_proba(Lselect_X_test5)[:,1]
df_probabilities['covid_death_pred_xgb']=X_LR5.predict_proba(Xselect_X_test5)[:,1]
df_probabilities['covid_death_pred_rf']=RF_LR5.predict_proba(RFselect_X_test5)[:,1]

df_probabilities['covid_hosp_pred_lr']=Q_LR6.predict_proba(Qselect_X_test)[:,1]
df_probabilities['covid_hosp_pred_lasso']=L_LR6.predict_proba(Lselect_X_test6)[:,1]
df_probabilities['covid_hosp_pred_xgb']=X_LR6.predict_proba(Xselect_X_test6)[:,1]
df_probabilities['covid_hosp_pred_rf']=RF_LR6.predict_proba(RFselect_X_test6)[:,1]

# COMMAND ----------

global_cohort_predictions=spark.createDataFrame(df_probabilities)
global_cohort_predictions.createOrReplaceTempView("global_cohort_predictions")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW ccu037_global_cohort_predictions as
# MAGIC SELECT
# MAGIC a.NHS_NUMBER_DEID,
# MAGIC b.age,
# MAGIC b.sex,
# MAGIC b.ethnicity_5_group,
# MAGIC b.ethnicity_11_group,
# MAGIC b.PrimaryCode_ethnicity,
# MAGIC --b.SNOMED_ethnicity,
# MAGIC --b.lsoa_code,
# MAGIC --b.lsoa_name,
# MAGIC --b.region_name,
# MAGIC b.IMD_quintile,
# MAGIC b.covid_death,
# MAGIC b.covid_hospitalisation,
# MAGIC b.post_covid_cvd_event_30d,
# MAGIC b.post_covid_cvd_event_90d,
# MAGIC b.post_covid_cvd_event_180d,
# MAGIC b.post_covid_cvd_event_1y,
# MAGIC b.post_covid_cvd_event_2y,
# MAGIC a.CVD30d_pred_lr,
# MAGIC a.CVD30d_pred_lasso,
# MAGIC a.CVD30d_pred_xgb,
# MAGIC a.CVD30d_pred_rf,
# MAGIC a.CVD90d_pred_lr,
# MAGIC a.CVD90d_pred_lasso,
# MAGIC a.CVD90d_pred_xgb,
# MAGIC a.CVD90d_pred_rf,
# MAGIC a.CVD180d_pred_lr,
# MAGIC a.CVD180d_pred_lasso,
# MAGIC a.CVD180d_pred_xgb,
# MAGIC a.CVD180d_pred_rf,
# MAGIC a.CVD1y_pred_lr,
# MAGIC a.CVD1y_pred_lasso,
# MAGIC a.CVD1y_pred_xgb,
# MAGIC a.CVD1y_pred_rf,
# MAGIC a.CVD2y_pred_lr,
# MAGIC a.CVD2y_pred_lasso,
# MAGIC a.CVD2y_pred_xgb,
# MAGIC a.CVD2y_pred_rf,
# MAGIC a.covid_death_pred_lr,
# MAGIC a.covid_death_pred_lasso,
# MAGIC a.covid_death_pred_xgb,
# MAGIC a.covid_death_pred_rf,
# MAGIC a.covid_hosp_pred_lr,
# MAGIC a.covid_hosp_pred_lasso,
# MAGIC a.covid_hosp_pred_xgb,
# MAGIC a.covid_hosp_pred_rf
# MAGIC FROM global_cohort_predictions a
# MAGIC LEFT JOIN dsa_391419_j3w9t_collab.ccu037_02_cohort_jul23 b
# MAGIC ON a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID

# COMMAND ----------

#drop_table("ccu037_mixed_cohort_predictions")
#create_table("ccu037_mixed_cohort_predictions")

# COMMAND ----------

# MAGIC %sql
# MAGIC create table dsa_391419_j3w9t_collab.ccu037_global_cohort_predictions
# MAGIC as
# MAGIC select * from ccu037_global_cohort_predictions

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from dsa_391419_j3w9t_collab.ccu037_global_cohort_predictions
