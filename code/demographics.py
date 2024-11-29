# Databricks notebook source
# MAGIC %md
# MAGIC # Master Demographics
# MAGIC  
# MAGIC **Description** 
# MAGIC
# MAGIC This notebook runs a list of `SQL` queries to:  
# MAGIC   
# MAGIC 1. Extract demographics from central `curr302_patient_skinny_record`  
# MAGIC     1.1 Map ethnicity categories  
# MAGIC 2. Add geographical & demographic data  
# MAGIC     2.1 Extracts Lower layer super output areas (LSOA) from GDPPR and SGSS  
# MAGIC     2.2 Extracts useful geographic & deprivation variables from dss_corporate.english_indices_of_dep_v02  
# MAGIC     2.3 Joins the geogrphic & deprivation data from 3.2, via the LSOA from 3.1  
# MAGIC     2.4 Calculates IMD quintiles   
# MAGIC 3. Cleaning  
# MAGIC 4. Create output table `X` 1 row per patient  
# MAGIC   
# MAGIC   
# MAGIC **Project(s)** CCU037
# MAGIC  
# MAGIC **Author(s)** Freya Allery (Adapted from Chris Tomlinson, project CCU013)
# MAGIC  
# MAGIC **Reviewer(s)** 
# MAGIC  
# MAGIC **Date last updated** 2022-06-20
# MAGIC  
# MAGIC **Date last reviewed** *NA*
# MAGIC  
# MAGIC **Date last run** 2022-06-20
# MAGIC
# MAGIC **Changelog**   
# MAGIC * **2021-10-06**: Reverted from `curr302_patient_skinny_record` to `curr302_patient_skinny_record_archive WHERE ProductionDate = "2021-07-29 13:39:04.161949"` due to ten-fold higher unknown ethnicity issues
# MAGIC * **2022-01-22** Updated 2022-01-22 for revised manuscript to use latest ProductionDate `2022-01-20 14:58:52.353312`
# MAGIC  
# MAGIC **Data input**   
# MAGIC * **Key demographics: `dars_nic_391419_j3w9t_collab.curr302_patient_skinny_record` **
# MAGIC * LSOA, comorbidity code search: `dars_nic_391419_j3w9t_collab.gdppr_dars_nic_391419_j3w9t_archive`  
# MAGIC * Geographic & deprivation data: `dss_corporate.english_indices_of_dep_v02`[https://db.core.data.digital.nhs.uk/#table/dss_corporate/english_indices_of_dep_v02] 
# MAGIC   
# MAGIC   
# MAGIC   
# MAGIC **Data output**  
# MAGIC * `dars_nic_391419_j3w9t_collab.ccu037_master_demographics` 1 row per patient
# MAGIC
# MAGIC **Software and versions** `SQL`, `Python`
# MAGIC  
# MAGIC **Packages and versions** `pyspark`

# COMMAND ----------

# MAGIC %run /Workspaces/dars_nic_391419_j3w9t_collab/CCU013/COVID-19-SEVERITY-PHENOTYPING/CCU013_00_helper_functions

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT distinct ProductionDate FROM dars_nic_391419_j3w9t_collab.curr302_patient_skinny_record

# COMMAND ----------

# Params
production_date = "2022-05-30 14:32:44.253705" # Notebook CCU03_01_create_table_aliases   Cell 8

# Table names
skinny_record = "dars_nic_391419_j3w9t_collab.curr302_patient_skinny_record_archive"
gdppr = "dars_nic_391419_j3w9t_collab.gdppr_dars_nic_391419_j3w9t_archive"
deprivation = "dss_corporate.english_indices_of_dep_v02"

# without dars_nic_391419_j3w9t_collab. prefix
output = "ccu037_demographics"

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Extract demographics from `curr302_patient_skinny_record`
# MAGIC This is a core table maintained by Sam Hollings from NHS Digital

# COMMAND ----------

demographics = spark.sql(f""" 
  SELECT
    NHS_NUMBER_DEID as person_id_deid,
    SEX as sex,
    DATE_OF_BIRTH as dob
  FROM 
    {skinny_record}
  WHERE
    ProductionDate == '{production_date}'
""")
display(demographics)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Calculate age
# MAGIC Calculate age at time of first covid event (`date_first` in `ccu013_covid_events`)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Deprivation data  

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Extract `lsoa` from GDPPR

# COMMAND ----------

lsoas = spark.sql(f'''
SELECT * FROM (SELECT
  NHS_NUMBER_DEID as person_id_deid,
  MAX(DATE) as date
FROM
  {gdppr}
WHERE 
  ProductionDate == "{production_date}"
GROUP BY
  NHS_NUMBER_DEID
  ) as tab1
LEFT JOIN
  (SELECT 
    NHS_NUMBER_DEID as person_id_deid,
    DATE as date,
    LSOA as lsoa 
  FROM 
    {gdppr}
  WHERE 
    ProductionDate == "{production_date}"
    ) as tab2
ON
  tab1.person_id_deid = tab2.person_id_deid
  AND
  tab1.date = tab2.date
''')
# Unsure why duplicate persist so crude approach = 
lsoas = lsoas.dropDuplicates(['person_id_deid'])
# Remove duplicate cols by only selecting necessary and using tab1. prefix
lsoas = lsoas.select("tab1.person_id_deid", "lsoa")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2. Select deprivation variables from [`dss_corporate.english_indices_of_dep_v02`](https://db.core.data.digital.nhs.uk/#table/dss_corporate/english_indices_of_dep_v02)
# MAGIC NB selected from where IMD year = 2019 so most up to date as database contains earlier years. As we're looking at COVID we want the most recent.

# COMMAND ----------

imd = spark.sql(f"""
SELECT
  LSOA_CODE_2011 as lsoa,
  DECI_IMD
FROM
  {deprivation}
WHERE
  IMD_YEAR = 2019
  """)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Join

# COMMAND ----------

# Join with Geographic data from GDPPR/SGSS to get LSOA
# LEFT join to retain all patients -> nulls where no LSOA
demographics = demographics.join(lsoas, "person_id_deid", "LEFT")

# Use LSOA to join with deprivation/geography
demographics = demographics \
  .join(imd, "lsoa", "LEFT") \

  #.drop("lsoa")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Calculate IMD quintiles
# MAGIC ** NB technically fifths as quintiles refers to the breakpoints **  
# MAGIC This seems to be a popular measure, easy to do given that we're supplied with deciles

# COMMAND ----------

from pyspark.sql.functions import col, create_map, lit
from itertools import chain

mapping = {
    '1': '1',
    '2': '1', 
    '3': '2', 
    '4': '2', 
    '5': '3', 
    '6': '3', 
    '7': '4', 
    '8': '4', 
    '9': '5', 
    '10': '5'}

mapping_expr = create_map([lit(x) for x in chain(*mapping.items())])

demographics = demographics \
  .withColumn("IMD_quintile", mapping_expr[col("DECI_IMD")]) \
  .drop("DECI_IMD")

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Cleaning

# COMMAND ----------

# Regexp cleaning
from pyspark.sql.functions import regexp_replace
demographics = demographics \
  .withColumn('sex', regexp_replace('sex', '0', 'Unknown')) \
  .withColumn('sex', regexp_replace('sex', '9', 'Unknown'))
# NB Sex: 8 = unspecified, 9 = "Home Leave" See: https://datadictionary.nhs.uk/attributes/sex_of_patients.html

# COMMAND ----------

# Coalescing nulls
from pyspark.sql.functions import col, coalesce, lit
demographics = demographics \
  .withColumn('dob', 
              coalesce(col('dob'), 
                       lit("Unknown"))) \
  .withColumn('IMD_quintile', 
              coalesce(col('IMD_quintile'), 
                       lit("Unknown"))) \
  .withColumn('lsoa', 
              coalesce(col('lsoa'), 
                       lit("Unknown"))) \
  .fillna(0)
# NA fill (0) with remaining to convert the comorbidities into binary flags

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Create table

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Check no duplicates
# MAGIC This will break the notebook if duplicates, thereby preventing the table being generated if duplicates!

# COMMAND ----------

assert demographics.select('person_id_deid').count() == demographics.select('person_id_deid').dropDuplicates(['person_id_deid']).count(), "Cohort contains duplicate ids when should be mutually exclusive"
print("Passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Commit & optimise

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

demographics.createOrReplaceGlobalTempView(output)
drop_table(output) 
create_table(output) 

# COMMAND ----------

spark.sql(f"OPTIMIZE dars_nic_391419_j3w9t_collab.{output} ZORDER BY person_id_deid")

# COMMAND ----------

display(
  spark.sql(f"""
  SELECT COUNT(*), COUNT(DISTINCT person_id_deid) FROM dars_nic_391419_j3w9t_collab.{output}
  """)
)

# COMMAND ----------

display(
  spark.sql(f"""
  SELECT * FROM dars_nic_391419_j3w9t_collab.{output}
  """)
)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM dars_nic_391419_j3w9t_collab.ccu037_demographics
