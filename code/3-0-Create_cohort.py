# Databricks notebook source
# MAGIC %md
# MAGIC **Description** This notebook creates the base cohort for CCU037_02
# MAGIC  
# MAGIC **Project(s)** CCU037_02 - Minimising bias in ethnicity data for more representative and generalisable models 
# MAGIC  
# MAGIC **Author(s)** Freya Allery
# MAGIC
# MAGIC **Reviewer(s)** 
# MAGIC  
# MAGIC **Date last updated** 19-01-2021

# COMMAND ----------

# MAGIC %md
# MAGIC # 0. Define functions

# COMMAND ----------

# Define create table function by Sam Hollings
# Source: Workspaces/dars_nic_391419_j3w9t_collab/DATA_CURATION_wrang000_functions
# Second source were pasted from: https://db.core.data.digital.nhs.uk/#notebook/2317231/command/2452227

def create_table(table_name:str, database_name:str='dars_nic_391419_j3w9t_collab', select_sql_script:str=None) -> None:
  """Will save to table from a global_temp view of the same name as the supplied table name (if no SQL script is supplied)
  Otherwise, can supply a SQL script and this will be used to make the table with the specificed name, in the specifcied database."""
  
  spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true")
  
  if select_sql_script is None:
    select_sql_script = f"SELECT * FROM global_temp.{table_name}"
  
  spark.sql(f"""CREATE TABLE {database_name}.{table_name} AS
                {select_sql_script}
             """)
  spark.sql(f"ALTER TABLE {database_name}.{table_name} OWNER TO {database_name}")
  
def drop_table(table_name:str, database_name:str='dars_nic_391419_j3w9t_collab', if_exists=True):
  if if_exists:
    IF_EXISTS = 'IF EXISTS'
  else: 
    IF_EXISTS = ''
  spark.sql(f"DROP TABLE {IF_EXISTS} {database_name}.{table_name}")

# COMMAND ----------

import databricks.koalas as ks
import pandas as pd

from functools import reduce
from operator import add
from pyspark.sql.functions import lit, col

def row_sum_across(*cols):
    return reduce(add, cols, lit(0))

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Join demographics and outcomes table to final cohort NHS numbers

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1. Ensure hospitalisation < 28 days after infection

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temp view min_hospitalisation_date as
# MAGIC select distinct person_id_deid, min(date) as min_hosp_date, 1 as covid_hosp_check from dars_nic_391419_j3w9t_collab.ccu013_covid_trajectory
# MAGIC where 
# MAGIC covid_phenotype == "02_Covid_admission" or
# MAGIC covid_phenotype == "03_ECMO_treatment" or
# MAGIC covid_phenotype == "03_ICU_admission" or
# MAGIC covid_phenotype == "03_IMV_treatment" or
# MAGIC covid_phenotype == "03_NIV_treatment"
# MAGIC group by person_id_deid

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temp view cohort_with_hosp_date_join as
# MAGIC select *
# MAGIC from dsa_391419_j3w9t_collab.ccu037_inclusion_criteria a
# MAGIC left join min_hospitalisation_date b
# MAGIC on a.nhs_number_deid == b.person_id_deid

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temp view cohort_with_hosp_date_check as
# MAGIC select 
# MAGIC NHS_NUMBER_DEID,
# MAGIC          age,
# MAGIC          sex,
# MAGIC          lsoa,
# MAGIC          IMD_quintile,
# MAGIC          ethnicity_5_group,
# MAGIC          ethnicity_11_group,
# MAGIC          death_covid,
# MAGIC CASE WHEN covid_hosp_check == 1 and min_hosp_date > (date_first_covid_event + INTERVAL 28 DAY) THEN 1 ELSE 0 END
# MAGIC AS covid_hospitalisation
# MAGIC from cohort_with_hosp_date_join

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2. Add in outcome tables

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW demographics_and_covid AS
      SELECT 
         b.NHS_NUMBER_DEID,
         b.age,
         b.sex,
         b.lsoa,
         b.IMD_quintile,
         b.ethnicity_5_group,
         b.ethnicity_11_group,
         b.PrimaryCode_ethnicity,
         b.death_covid,
         c.covid_hospitalisation
         --d.1_year_pre_covid_CVD_event,
         --d.pre_covid_CVD_event,
         --d.30_day_CVD_event as post_covid_cvd_event_30d, 
         --d.90_day_CVD_event as post_covid_cvd_event_90d, 
         --d.180_day_CVD_event as post_covid_cvd_event_180d,
         --d.1_year_CVD_event as post_covid_cvd_event_1y, 
         --d.2_years_CVD_event as post_covid_cvd_event_2y, 
         --d.VTE_outcome
      FROM dsa_391419_j3w9t_collab.ccu037_inclusion_criteria as b
      LEFT JOIN cohort_with_hosp_date_check as c
      ON b.NHS_NUMBER_DEID = c.NHS_NUMBER_DEID
      --LEFT JOIN dsa_391419_j3w9t_collab.ccu037_CVD_events as d
      --ON b.NHS_NUMBER_DEID = d.person_id_deid
      """)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Join predictors and binarise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1. Join predictors

# COMMAND ----------

spark.sql("""
CREATE OR REPLACE TEMP VIEW predictors AS
      SELECT * FROM dars_nic_391419_j3w9t_collab.ccu037_variables_phenotypes_list_final3
      UNION ALL
      SELECT * FROM dars_nic_391419_j3w9t_collab.ccu037_medications_list_final
      UNION ALL
      SELECT * FROM dsa_391419_j3w9t_collab.ccu037_CVD_events_list
      UNION ALL
      SELECT * FROM dars_nic_391419_j3w9t_collab.ccu037_other_qrisk_phenotypes
""")


# COMMAND ----------

#new try
spark.sql("""
CREATE OR REPLACE TEMP VIEW predictors AS
   SELECT a.NHS_NUMBER_DEID, b.predictor
     FROM demographics_and_covid as a
     LEFT JOIN
      (SELECT * FROM dars_nic_391419_j3w9t_collab.ccu037_variables_phenotypes_list_final3
      UNION ALL
      SELECT * FROM dars_nic_391419_j3w9t_collab.ccu037_medications_list_final
      UNION ALL
      SELECT * FROM dsa_391419_j3w9t_collab.ccu037_CVD_events_list
      UNION ALL
      SELECT * FROM dars_nic_391419_j3w9t_collab.ccu037_other_qrisk_phenotypes) AS b
      ON a.NHS_NUMBER_DEID = b.person_id_deid  
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2. Binarise

# COMMAND ----------

predictors = spark.sql(f"""
SELECT NHS_NUMBER_DEID, predictor, 1 as value FROM predictors""")

# COMMAND ----------

predictors = predictors \
  .to_koalas() \
  .pivot(index='NHS_NUMBER_DEID', 
         columns='predictor', 
         values='value') \
  .fillna(0) \
  .reset_index() \
  .to_spark()
# Reset index to breakout ids to separate col

# COMMAND ----------

predictors.createOrReplaceTempView("predictors")

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct person_id_deid) from predictors

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct NHS_NUMBER_DEID) from predictors

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from predictors

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Left join predictors onto cohort

# COMMAND ----------

spark.sql("""
CREATE OR REPLACE TEMP VIEW ccu037_02_risk_prediction_cohort_final AS
      SELECT
      
         a.NHS_NUMBER_DEID,
         a.age,
         a.sex,
         a.lsoa,
         a.IMD_quintile,
         a.ethnicity_5_group,
         a.ethnicity_11_group,
         a.PrimaryCode_ethnicity,
         
         b.pre_covid_CVD_event as pre_covid_cvd_event,
         b.1_year_pre_covid_CVD_event as pre_covid_cvd_event_1y,
         b.cancer,
         b.alcohol_problems,
         b.hypertension,
         b.alcoholic_liver_disease,
         b.anti_coagulant_drugs,
         b.anti_diabetic_drugs,
         b.anti_platelet_drugs,
         b.anti_hypertensive_drugs,
         b.autoimmune_liver_disease,
         b.copd,
         b.dementia,
         b.diabetes,
         b.fracture_of_hip,
         b.fracture_of_wrist,
         b.obesity,
         b.osteoporosis,
         b.statins,
         
         b.schizophrenia,
         b.bipolardisorder,
         b.depression,
         b.antipsychotic,
         b.erectiledysfunction,
         b.CKD,
         b.AF,
         b.RA,
         b.smoking,
         
         a.death_covid,
         a.covid_hospitalisation,

         b.30_day_CVD_event as post_covid_cvd_event_30d, 
         b.90_day_CVD_event as post_covid_cvd_event_90d, 
         b.180_day_CVD_event as post_covid_cvd_event_180d,
         b.1_year_CVD_event as post_covid_cvd_event_1y, 
         b.2_years_CVD_event as post_covid_cvd_event_2y, 
         b.VTE_outcome
     FROM demographics_and_covid as a
     LEFT JOIN predictors as b
     ON a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID        
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from ccu037_02_risk_prediction_cohort_final where post_covid_cvd_event_1y is null

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1. Add in LSOA names

# COMMAND ----------

spark.sql(f""" 
CREATE OR REPLACE TEMP VIEW lsoa_ids AS
SELECT DISTINCT NHS_NUMBER_DEID, LSOA AS lsoa_code
FROM ccu037_02_risk_prediction_cohort_final
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW geography AS
# MAGIC SELECT lsoa.NHS_NUMBER_DEID, 
# MAGIC lsoa.lsoa_code,
# MAGIC geog.lsoa_name,
# MAGIC geog.county_name,
# MAGIC geog.region_name
# MAGIC FROM lsoa_ids AS lsoa
# MAGIC LEFT JOIN dars_nic_391419_j3w9t_collab.curr901a_lsoa_region_lookup AS geog ON lsoa.lsoa_code = geog.lsoa_code

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW cohort AS
SELECT 
         a.NHS_NUMBER_DEID,
         a.age,
         a.sex,
         
         a.lsoa_code,
         --b.lsoa_name,
         --b.county_name,
         --b.region_name,
         
         a.IMD_quintile,
         a.ethnicity_5_group,
         a.ethnicity_11_group,
         a.PrimaryCode_ethnicity,
         
         a.pre_covid_cvd_event,
         a.pre_covid_cvd_event_1y,
         a.cancer,
         a.alcohol_problems,
         a.hypertension,
         a.alcoholic_liver_disease,
         a.anti_coagulant_drugs,
         a.anti_diabetic_drugs,
         a.anti_platelet_drugs,
         a.anti_hypertensive_drugs,
         a.autoimmune_liver_disease,
         a.copd,
         a.dementia,
         a.diabetes,
         a.fracture_of_hip,
         a.fracture_of_wrist,
         a.obesity,
         a.osteoporosis,
         a.statins,
         
         a.schizophrenia,
         a.bipolardisorder,
         a.depression,
         a.antipsychotic,
         a.erectiledysfunction,
         a.CKD,
         a.AF,
         a.RA,
         a.smoking,
         
         a.death_covid,
         a.covid_hospitalisation,
         a.post_covid_cvd_event_30d, 
         a.post_covid_cvd_event_90d, 
         a.post_covid_cvd_event_180d,
         a.post_covid_cvd_event_1y, 
         a.post_covid_cvd_event_2y, 
         a.VTE_outcome
         
FROM global_temp.ccu037_02_risk_prediction_cohort_final a
LEFT JOIN global_temp.geography b
ON a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2. Ensure correct demographic info format

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW info_check AS
    SELECT
         NHS_NUMBER_DEID,
         age,
         sex,
         
         lsoa,
         --lsoa_name,
         --county_name,
         --region_name,
         
         IMD_quintile,
         ethnicity_5_group,
         ethnicity_11_group,
         PrimaryCode_ethnicity,
         
         pre_covid_cvd_event,
         pre_covid_cvd_event_1y,
         cancer,
         alcohol_problems,
         hypertension,
         alcoholic_liver_disease,
         anti_coagulant_drugs,
         anti_diabetic_drugs,
         anti_platelet_drugs,
         anti_hypertensive_drugs,
         autoimmune_liver_disease,
         copd,
         dementia,
         diabetes,
         fracture_of_hip,
         fracture_of_wrist,
         obesity,
         osteoporosis,
         statins,
         
         schizophrenia,
         bipolardisorder,
         depression,
         antipsychotic,
         erectiledysfunction,
         CKD,
         AF,
         RA,
         smoking,
         
         CASE WHEN (death_covid IS NULL OR death_covid = 0) THEN 0 ELSE 1 END as covid_death,
         covid_hospitalisation,
         post_covid_cvd_event_30d, 
         post_covid_cvd_event_90d,
         post_covid_cvd_event_180d,
         post_covid_cvd_event_1y, 
         post_covid_cvd_event_2y, 
         VTE_outcome
         
         FROM ccu037_02_risk_prediction_cohort_final
         -- WHERE age IS NOT null AND sex IS NOT null AND sex != "Unknown" AND PrimaryCode_ethnicity IS NOT NULL AND PrimaryCode_ethnicity != "" """)

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temp view imd as
# MAGIC select
# MAGIC NHS_NUMBER_DEID,
# MAGIC          age,
# MAGIC          sex,
# MAGIC          
# MAGIC          lsoa,
# MAGIC          --lsoa_name,
# MAGIC          --county_name,
# MAGIC          --region_name,
# MAGIC        
# MAGIC          CASE WHEN IMD_quintile LIKE "%1%" THEN "IMD_1"
# MAGIC          WHEN IMD_quintile LIKE "%2%"  THEN "IMD_2"
# MAGIC          WHEN IMD_quintile LIKE "%3%"  THEN "IMD_3"
# MAGIC          WHEN IMD_quintile LIKE "%4%"  THEN "IMD_4"
# MAGIC          WHEN IMD_quintile LIKE "%5%"  THEN "IMD_5"
# MAGIC          ELSE "IMD_Unknown"
# MAGIC          END as IMD_quintile,
# MAGIC          ethnicity_5_group,
# MAGIC          CASE 
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%0%" THEN "White"
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%1%" THEN "Black Caribbean"
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%2%" THEN "Black African" 
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%3%" THEN "Other Black" 
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%4%" THEN "Indian" 
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%5%" THEN "Pakistani"
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%6%" THEN "Bangladeshi"
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%7%" THEN "Chinese"
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%8%" THEN "Other Ethnic Group"
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%W%" THEN "Other Ethnic Group"
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%9%" THEN "Unknown"
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%99%" THEN "Unknown"
# MAGIC          ELSE ethnicity_11_group 
# MAGIC          END as ethnicity_11_group,
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
# MAGIC FROM info_check

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temp view ccu037_02_cohort as
# MAGIC select
# MAGIC NHS_NUMBER_DEID,
# MAGIC          age,
# MAGIC          sex,
# MAGIC          
# MAGIC          lsoa,
# MAGIC          --lsoa_name,
# MAGIC          --county_name,
# MAGIC          --region_name,
# MAGIC        
# MAGIC          IMD_quintile,
# MAGIC          CASE
# MAGIC          WHEN ethnicity_11_group LIKE "%Chinese%" THEN "Asian or Asian British"
# MAGIC          ELSE ethnicity_5_group END AS
# MAGIC          ethnicity_5_group,
# MAGIC          ethnicity_11_group,
# MAGIC          CASE 
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%0%" THEN "C"
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%1%" THEN "M"
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%2%" THEN "N" 
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%3%" THEN "P" 
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%4%" THEN "H" 
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%5%" THEN "J"
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%6%" THEN "K"
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%7%" THEN "R"
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%8%" THEN "S"
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%9%" THEN "Z"
# MAGIC          WHEN PrimaryCode_ethnicity LIKE "%99%" THEN "Z"
# MAGIC
# MAGIC          ELSE PrimaryCode_ethnicity
# MAGIC          END as 
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
# MAGIC FROM imd

# COMMAND ----------

# MAGIC %sql
# MAGIC create table dsa_391419_j3w9t_collab.ccu037_02_cohort_jun23
# MAGIC as select * from ccu037_02_cohort

# COMMAND ----------

# MAGIC %sql
# MAGIC -- most up to date
# MAGIC create table dsa_391419_j3w9t_collab.ccu037_02_cohort_jul23
# MAGIC as select * from ccu037_02_cohort

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from ccu037_02_cohort where post_covid_cvd_event_30d is null

# COMMAND ----------

# MAGIC %sql
# MAGIC create table dsa_391419_j3w9t_collab.ccu037_02_cohort_jun23_updated
# MAGIC as select * from ccu037_02_cohort

# COMMAND ----------

#drop_table("ccu037_02_cohort")
#create_table("ccu037_02_cohort")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from dsa_391419_j3w9t_collab.ccu037_02_cohort_jul23
