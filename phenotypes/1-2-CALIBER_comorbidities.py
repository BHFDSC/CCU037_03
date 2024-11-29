# Databricks notebook source
# MAGIC %md
# MAGIC Builds a binary feature matrix of CALIBER phenotypes prior to 2022-07-29 (study end date)

# COMMAND ----------

# MAGIC %run /Workspaces/dars_nic_391419_j3w9t_collab/CCU013/COVID-19-SEVERITY-PHENOTYPING/CCU013_00_helper_functions

# COMMAND ----------

import databricks.koalas as ks
import pandas as pd

from functools import reduce
from operator import add
from pyspark.sql.functions import lit, col

def row_sum_across(*cols):
    return reduce(add, cols, lit(0))

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

# Table names
demographics_table = "dars_nic_391419_j3w9t_collab.ccu037_cohort_base_over18_posi_test"
phenotypes_table = "dars_nic_391419_j3w9t_collab.ccu037_caliber"
gdppr_table = "dars_nic_391419_j3w9t_collab.gdppr_dars_nic_391419_j3w9t_archive"
production_date = spark.sql("SELECT MAX(ProductionDate) FROM dars_nic_391419_j3w9t_collab.wrang002b_data_version_batchids").first()[0]
print("ProductionDate:", production_date)

# without dars_nic_391419_j3w9t_collab. prefix
output_table = "ccu037_caliber_phenotypes_binary"

# COMMAND ----------

# MAGIC %md
# MAGIC # Removing individuals with < 1 year of records available from the covid date

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM dars_nic_391419_j3w9t_collab.gdppr_dars_nic_391419_j3w9t_archive WHERE ProductionDate = "2022-06-29 00:00:00.000000"

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT NHS_NUMBER_DEID) FROM dars_nic_391419_j3w9t_collab.gdppr_dars_nic_391419_j3w9t_archive WHERE ProductionDate = "2022-06-29 00:00:00.000000"

# COMMAND ----------

# Selecting the minimum phenotype date for a given person ID (their earliest record)
spark.sql(f"""
    CREATE OR REPLACE GLOBAL TEMP VIEW earliest_pheno_record as
    SELECT NHS_NUMBER_DEID phenotype, min(date) as min_pheno_date, code, terminology FROM {phenotypes_table}
    GROUP BY NHS_NUMBER_DEID, phenotype, code, terminology
""")

# COMMAND ----------

# Selecting the minimum gdppr date date for a given person ID (their earliest record)
spark.sql(f"""
    CREATE OR REPLACE GLOBAL TEMP VIEW earliest_gdppr_record as
    SELECT NHS_NUMBER_DEID, min(RECORD_DATE) as min_record_date FROM {gdppr_table} WHERE ProductionDate = "2022-06-29 00:00:00.000000"
    GROUP BY NHS_NUMBER_DEID
""")

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE GLOBAL TEMP VIEW earliest_pheno_record as
WITH added_row_number AS (
  SELECT
    *,
    ROW_NUMBER() OVER(PARTITION BY person_id_deid ORDER BY date ASC) AS row_number
  FROM {phenotypes_table}
)
SELECT
  person_id_deid, phenotype, date as min_pheno_date, code, terminology
FROM added_row_number
WHERE row_number = 1
""")

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE GLOBAL TEMP VIEW earliest_gdppr_record as
WITH added_row_number AS (
  SELECT
    *,
    ROW_NUMBER() OVER(PARTITION BY NHS_NUMBER_DEID ORDER BY RECORD_DATE ASC) AS row_number
  FROM {gdppr_table} WHERE ProductionDate = "2022-06-29 00:00:00.000000"
)
SELECT
  NHS_NUMBER_DEID, RECORD_DATE as min_record_date
FROM added_row_number
WHERE row_number = 1
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM global_temp.earliest_pheno_record ORDER BY PERSON_ID_DEID

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM global_temp.earliest_gdppr_record

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM global_temp.earliest_pheno_record

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT PERSON_ID_DEID) FROM global_temp.earliest_pheno_record

# COMMAND ----------

# Joining  the earliest pheno record date to date of first covid event for cohort, subtracting by 365 days and filtering for null and less than 365 days records
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW record_date_check AS
    SELECT *
    FROM
    (SELECT *
    FROM {demographics_table}
    LEFT JOIN global_temp.earliest_gdppr_record as b
    ON {demographics_table}.NHS_NUMBER_DEID = b.person_id_deid)
    WHERE min_pheno_date IS NOT null AND min_pheno_date < (date_first_covid_event - INTERVAL 365 DAY)
""")

# COMMAND ----------

# Joining  the earliest pheno record date to date of first covid event for cohort, subtracting by 365 days and filtering for null and less than 365 days records
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW gdppr_record_date_check AS
    SELECT *
    FROM
    (SELECT *
    FROM {demographics_table}
    LEFT JOIN global_temp.earliest_gdppr_record as b
    ON {demographics_table}.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID)
    WHERE min_record_date IS NOT null AND min_record_date < (date_first_covid_event - INTERVAL 365 DAY)
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM global_temp.record_date_check

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM global_temp.record_date_check

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM global_temp.gdppr_record_date_check

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_cohort_all_inclusion_criteria AS
SELECT NHS_NUMBER_DEID, 
 dob, 
 age, 
 sex, 
 lsoa, 
 IMD_quintile, 
 5_group_ethnicity,
 11_group_ethnicity,
 PrimaryCode_ethnicity,
 SNOMED_ethnicity,
 covid_first_event,
 date_first_covid_event,
 death,
 date_death,
 death_covid,
 01_Covid_positive_test,
 01_GP_covid_diagnosis,
 02_Covid_admission,
 03_ECMO_treatment,
 03_ICU_admission,
 03_IMV_treatment,
 03_NIV_treatment,
 04_Covid_inpatient_death,
 04_Fatal_with_covid_diagnosis,
 04_Fatal_without_covid_diagnosis,
 0_Covid_infection,
 covid_severity,
 ventilatory_support
FROM global_temp.record_date_check
""")

# COMMAND ----------

drop_table("ccu037_cohort_all_inclusion_criteria")
create_table("ccu037_cohort_all_inclusion_criteria")


# COMMAND ----------

demographics_table = "dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria"

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria

# COMMAND ----------

# MAGIC %md
# MAGIC # Adding in comorbidities

# COMMAND ----------

comorbidities = spark.sql(f"""
  SELECT 
    base.NHS_NUMBER_DEID as person_id_deid,
    phenotype,
    value
  FROM 
    {demographics_table} as base
  FULL JOIN
    (SELECT 
      person_id_deid, 
      date, 
      phenotype, 
      1 as value 
    FROM 
      {phenotypes_table}) 
      as phenos
  ON 
    base.NHS_NUMBER_DEID = phenos.person_id_deid
""")

# COMMAND ----------

display(comorbidities)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pivot
# MAGIC Use `koalas` to use `pandas` like pivot syntax but parallelisable for large dataframes

# COMMAND ----------

comorbidities = comorbidities \
  .to_koalas() \
  .pivot(index='person_id_deid', 
         columns='phenotype', 
         values='value') \
  .fillna(0) \
  .reset_index() \
  .to_spark()
# Reset index to breakout ids to separate col

# COMMAND ----------

display(comorbidities)

# COMMAND ----------

display(comorbidities)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add multimorbidity variable

# COMMAND ----------

phenos = comorbidities.schema.names[1:]

vars = [col(x) for x in phenos]

comorbidities = comorbidities \
  .fillna(0) \
  .withColumn('multimorbidity', row_sum_across(*vars))

# COMMAND ----------

display(comorbidities)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Commit

# COMMAND ----------

comorbidities.createOrReplaceGlobalTempView(output_table)
drop_table(output_table)
create_table(output_table)
