# Databricks notebook source
# MAGIC %md
# MAGIC # Import libraries

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

# COMMAND ----------

# MAGIC %md
# MAGIC # Load data

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temp view data as
# MAGIC select
# MAGIC NHS_NUMBER_DEID,
# MAGIC          age,
# MAGIC          CASE WHEN sex LIKE "%2%" then 1
# MAGIC          ELSE 0 END as
# MAGIC          sex,
# MAGIC          
# MAGIC          --lsoa_name,
# MAGIC          --county_name,
# MAGIC          --region_name,
# MAGIC        
# MAGIC          IMD_quintile,
# MAGIC          ethnicity_5_group,
# MAGIC          ethnicity_11_group,
# MAGIC          CASE WHEN PrimaryCode_ethnicity = "" then "Z"
# MAGIC               WHEN PrimaryCode_ethnicity LIKE "%X%" then "Z"
# MAGIC          ELSE PrimaryCode_ethnicity END AS
# MAGIC          PrimaryCode_ethnicity,
# MAGIC          
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
# MAGIC          
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
# MAGIC FROM dsa_391419_j3w9t_collab.ccu037_02_cohort_jul23

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW data1 AS
# MAGIC select *,
# MAGIC CASE
# MAGIC WHEN age > 17 and age < 30 THEN 1
# MAGIC ELSE 0 END AS age18_29,
# MAGIC CASE
# MAGIC WHEN age > 29 and age < 40 THEN 1
# MAGIC ELSE 0 END AS age30_39,
# MAGIC CASE
# MAGIC WHEN age > 39 and age < 50 THEN 1
# MAGIC ELSE 0 END AS age40_49,
# MAGIC CASE
# MAGIC WHEN age > 49 and age < 60 THEN 1
# MAGIC ELSE 0 END AS age50_59,
# MAGIC CASE
# MAGIC WHEN age > 59 and age < 70 THEN 1
# MAGIC ELSE 0 END AS age60_69,
# MAGIC CASE
# MAGIC WHEN age > 69 and age < 80 THEN 1
# MAGIC ELSE 0 END AS age70_79,
# MAGIC CASE
# MAGIC WHEN age > 79 and age < 90 THEN 1
# MAGIC ELSE 0 END AS age80_89,
# MAGIC CASE
# MAGIC WHEN age > 89 THEN 1
# MAGIC ELSE 0 END AS age90_plus,
# MAGIC CASE 
# MAGIC WHEN ethnicity_11_group LIKE "%White%" then "White_or_not_stated"
# MAGIC WHEN ethnicity_11_group LIKE "%Unknown%" then "White_or_not_stated"
# MAGIC WHEN ethnicity_11_group LIKE "%Mixed%" then "Other_Ethnic_Group"
# MAGIC WHEN ethnicity_11_group LIKE "%Other Black%" then "Other_Ethnic_Group"
# MAGIC WHEN ethnicity_11_group LIKE "%Other Ethnic Group%" then "Other_Ethnic_Group"
# MAGIC WHEN ethnicity_11_group LIKE "%Black Caribbean%" then "Black_Caribbean"
# MAGIC WHEN ethnicity_11_group LIKE "%Black African%" then "Black_African"
# MAGIC WHEN ethnicity_11_group LIKE "%Other Asian%" then "Other_Asian"
# MAGIC ELSE ethnicity_11_group END AS qrisk_eth,
# MAGIC CASE
# MAGIC WHEN (schizophrenia=1 or bipolardisorder=1 or depression =1) THEN 1
# MAGIC ELSE 0 END AS severe_mental_illness
# MAGIC from data

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct qrisk_eth from data1

# COMMAND ----------

cohort = spark.table("data1")

# COMMAND ----------

# cast integer type
#from pyspark.sql.functions import col
#from pyspark.sql.types import IntegerType, FloatType

#cohort = cohort \
#.withColumn('sex', col('sex').cast(IntegerType())) 
#.withColumn('IMD_quintile', col('IMD_quintile').cast(IntegerType())) \

# COMMAND ----------

cohort = cohort.toPandas()

# COMMAND ----------

cohort.isnull().sum()

# COMMAND ----------

cohort = cohort.dropna()

# COMMAND ----------

cohort.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC # One hot encoding

# COMMAND ----------

# One hot encode categorical variables
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#creating instance of one-hot-encoder
labelencoder = LabelEncoder()
enc = OneHotEncoder(handle_unknown='ignore')

# Get one hot encoding of IMD
one_hot = pd.get_dummies(cohort['IMD_quintile'])
# Drop column B as it is now encoded
cohort = cohort.drop('IMD_quintile',axis = 1)
# Join the encoded df
cohort = cohort.join(one_hot)

# One hot encoding of qrisk_eth
one_hot_q = pd.get_dummies(cohort['qrisk_eth'])
cohort = cohort.drop('qrisk_eth',axis = 1)
cohort = cohort.join(one_hot_q)

# One hot encoding of PrimaryCode_ethnicity
#one_hot_p = pd.get_dummies(cohort['PrimaryCode_ethnicity'])
#cohort = cohort.drop('PrimaryCode_ethnicity',axis = 1)
#cohort = cohort.join(one_hot_p)

# COMMAND ----------

cohort.columns

# COMMAND ----------

cohort

# COMMAND ----------

cohort = cohort.astype({"age": int})

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

ccu037_cohort_binarised=spark.createDataFrame(cohort)

# COMMAND ----------

ccu037_cohort_binarised.createOrReplaceTempView("ccu037_cohort_binarised")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from ccu037_cohort_binarised

# COMMAND ----------

# MAGIC %sql
# MAGIC create table dsa_391419_j3w9t_collab.ccu037_cohort_binarised
# MAGIC as select * from ccu037_cohort_binarised

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from dsa_391419_j3w9t_collab.ccu037_cohort_binarised
