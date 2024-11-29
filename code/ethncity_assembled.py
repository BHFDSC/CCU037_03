# Databricks notebook source
# MAGIC %md
# MAGIC # Ethnicity table assembled
# MAGIC  
# MAGIC **Description** 
# MAGIC
# MAGIC This notebook runs a list of `SQL` queries to:  
# MAGIC   
# MAGIC 1. Take the latest ethnicity record from each source
# MAGIC 2. Create prioritisation to get one code per patient, from:
# MAGIC    a) SNOMED GDPPR code if not null
# MAGIC    b) Primary GDPPR code if not null
# MAGIC    c) Primary HES code if not null
# MAGIC   
# MAGIC   
# MAGIC **Project(s)** CCU037
# MAGIC  
# MAGIC **Author(s)** Freya Allery 
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
# MAGIC  
# MAGIC **Data input**   
# MAGIC * gdppr_data =  `dars_nic_391419_j3w9t_collab.ccu037_gdppr_primary_and_snomed_fa` 
# MAGIC * hes_data = `dars_nic_391419_j3w9t_collab.hes_apc_all_years`  
# MAGIC     
# MAGIC **Data output**  
# MAGIC * `dars_nic_391419_j3w9t_collab.ccu037_ethnicity` 1 row per patient
# MAGIC
# MAGIC **Software and versions** `SQL`, `Python`
# MAGIC
# MAGIC **Packages and versions** `pyspark`

# COMMAND ----------

# MAGIC %md
# MAGIC # 0. Define functions, import libraries

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

gdppr_prod_date = "2022-05-30 14:32:44.253705"
hes_prod_date = "2022-03-31 00:00:00.000000"

# COMMAND ----------

#SNOMED codes, 1 as first choice ranking
spark.sql(f"""CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_gdppr_ethnicity_SNOMED AS 
               SELECT NHS_NUMBER_DEID,
                      ConceptId as SNOMED_ETHNIC,
                      PrimaryCode as ETHNIC,
                      DATE as RECORD_DATE,
                      'GDPPR_snomed' as dataset,
                      1 as eth_rank,
               recent_rank
               FROM (select *, row_number() over (partition by NHS_NUMBER_DEID order by (CASE WHEN ETHNIC IS                NULL or TRIM(ETHNIC) IN ("","9", "99", "X" , "Z") THEN 1 ELSE 0 END),             REPORTING_PERIOD_END_DATE desc) as recent_rank 
                FROM dars_nic_391419_j3w9t_collab.gdppr_dars_nic_391419_j3w9t_archive as gdppr
                      INNER JOIN dss_corporate.gdppr_ethnicity_mappings eth on gdppr.CODE = eth.ConceptId
                WHERE gdppr.ProductionDate = '{gdppr_prod_date}')
                WHERE recent_rank = 1 """)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM global_temp.ccu037_gdppr_ethnicity_SNOMED 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT NHS_NUMBER_DEID) FROM global_temp.ccu037_gdppr_ethnicity_SNOMED 

# COMMAND ----------

# Primary codes, 2 as second choice ranking
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_gdppr_ethnicity AS 
       SELECT NHS_NUMBER_DEID, 
              ETHNIC, REPORTING_PERIOD_END_DATE as RECORD_DATE, 
              'GDPPR' as dataset, 
              2 as eth_rank,
       recent_rank
       FROM (select *, row_number() over (partition by NHS_NUMBER_DEID order by (CASE WHEN ETHNIC IS NULL or          TRIM(ETHNIC) IN ("","9", "99", "X" , "Z") THEN 1 ELSE 0 END), REPORTING_PERIOD_END_DATE desc) as              recent_rank 
       FROM dars_nic_391419_j3w9t_collab.gdppr_dars_nic_391419_j3w9t_archive as gdppr 
       WHERE ProductionDate = "{gdppr_prod_date}") 
       WHERE recent_rank = 1""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM global_temp.ccu037_gdppr_ethnicity

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT NHS_NUMBER_DEID) FROM global_temp.ccu037_gdppr_ethnicity

# COMMAND ----------

spark.sql(f"""
 CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_all_hes_apc_ethnicity AS
 SELECT DISTINCT PERSON_ID_DEID as NHS_NUMBER_DEID, 
      ETHNOS as ETHNIC, 
      EPISTART as RECORD_DATE, 
      "hes_apc" as dataset,
      3 as eth_rank,
      recent_rank
  FROM (select *, row_number() over (partition by PERSON_ID_DEID order by (CASE WHEN ETHNOS IS NULL or TRIM(ETHNOS) IN ("","9", "99", "X" , "Z") THEN 1 ELSE 0 END), EPISTART desc) as recent_rank from dars_nic_391419_j3w9t_collab.hes_apc_all_years_archive 
  WHERE ProductionDate = "{hes_prod_date}")
  WHERE recent_rank = 1""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM global_temp.ccu037_all_hes_apc_ethnicity

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT NHS_NUMBER_DEID) FROM global_temp.ccu037_all_hes_apc_ethnicity

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_ethnicity_all_codes_latest as
# MAGIC SELECT NHS_NUMBER_DEID, ETHNIC, NULL AS SNOMED_ETHNIC, RECORD_DATE, dataset, eth_rank 
# MAGIC FROM global_temp.ccu037_gdppr_ethnicity
# MAGIC UNION ALL
# MAGIC SELECT NHS_NUMBER_DEID, ETHNIC, SNOMED_ETHNIC, RECORD_DATE, dataset, eth_rank 
# MAGIC FROM global_temp.ccu037_gdppr_ethnicity_SNOMED
# MAGIC UNION ALL
# MAGIC SELECT NHS_NUMBER_DEID, ETHNIC, null AS SNOMED_ETHNIC, RECORD_DATE, dataset, eth_rank 
# MAGIC FROM global_temp.ccu037_all_hes_apc_ethnicity

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM global_temp.ccu037_ethnicity_all_codes_latest

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(*) FROM global_temp.ccu037_ethnicity_all_codes_latest WHERE SNOMED_ETHNIC IS NOT null

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Flagging the null data in the latest codes
# MAGIC CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_ethnicity_all_codes_latest_null as
# MAGIC SELECT *,
# MAGIC      CASE WHEN ETHNIC IS NULL or TRIM(ETHNIC) IN ("","9", "99", "X" , "Z") THEN 1 ELSE 0 END as ethnic_null,
# MAGIC      CASE WHEN SNOMED_ETHNIC IS NULL THEN 1 ELSE 0 END as SNOMED_ethnic_null
# MAGIC      FROM global_temp.ccu037_ethnicity_all_codes_latest

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Adding in ethnicity_rank
# MAGIC CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_ethnicity_all_codes_latest_null_rank as
# MAGIC SELECT *,
# MAGIC      CASE
# MAGIC          WHEN SNOMED_ethnic_null = 0 THEN 3 
# MAGIC          WHEN (eth_rank = 2 AND ethnic_null = 0) THEN 2
# MAGIC          WHEN (eth_rank = 3 AND ethnic_null = 0) THEN 1
# MAGIC          ELSE 0
# MAGIC        END as ethnicity_rank
# MAGIC      FROM global_temp.ccu037_ethnicity_all_codes_latest_null

# COMMAND ----------

spark.sql(f"""
 CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_all_hes_apc_ethnicity AS
 SELECT DISTINCT PERSON_ID_DEID as NHS_NUMBER_DEID, 
      ETHNOS as ETHNIC, 
      EPISTART as RECORD_DATE, 
      "hes_apc" as dataset,
      3 as eth_rank,
      recent_rank
  FROM (select *, row_number() over (partition by PERSON_ID_DEID order by (CASE WHEN ETHNOS IS NULL or TRIM(ETHNOS) IN ("","9", "99", "X" , "Z") THEN 1 ELSE 0 END), EPISTART desc) as recent_rank from dars_nic_391419_j3w9t_collab.hes_apc_all_years_archive 
  WHERE ProductionDate = "{hes_prod_date}")
  WHERE recent_rank = 1""")

# COMMAND ----------

spark.sql(f"""
 CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_ethnicity_assembled AS
 SELECT DISTINCT NHS_NUMBER_DEID, 
      ETHNIC,
      SNOMED_ETHNIC,
      RECORD_DATE, 
      dataset,
      ranking
  FROM (select *, row_number() over (partition by NHS_NUMBER_DEID order by ethnicity_rank desc) as ranking from global_temp.ccu037_ethnicity_all_codes_latest_null_rank) 
       WHERE ranking = 1""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM global_temp.ccu037_ethnicity_assembled

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT NHS_NUMBER_DEID) FROM global_temp.ccu037_ethnicity_assembled

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_ethnicity_assembled AS
SELECT 
NHS_NUMBER_DEID, 
ETHNIC, 
SNOMED_ETHNIC,
CASE WHEN ETHNIC IN ('1','2','3','N','M','P') THEN "Black or Black British"
           WHEN ETHNIC IN ('0','A','B','C', 'T') THEN "White"
           WHEN ETHNIC IN ('4','5','6','L','K','J','H', 'R') THEN "Asian or Asian British"
           WHEN ETHNIC IN ('7','8','W','S') THEN "Other Ethnic Groups"
           WHEN ETHNIC IN ('D','E','F','G') THEN "Mixed"
           WHEN ETHNIC IN (9", "99", "X" , "Z") THEN "Unknown"
           ELSE 'Unknown' END as ETHNIC_GROUP, 
RECORD_DATE, 
dataset 
FROM global_temp.ccu037_ethnicity_assembled """)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM global_temp.ccu037_ethnicity_assembled

# COMMAND ----------

drop_table("ccu037_ethnicity_assembled")
create_table("ccu037_ethnicity_assembled")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT(ETHNIC) FROM dars_nic_391419_j3w9t_collab.ccu037_ethnicity_assembled

# COMMAND ----------

# Adding in 11 levels of granularity (as per ONS) 
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_ethnicity_assembled_multiple_granularities AS
SELECT 
NHS_NUMBER_DEID,
CASE WHEN ETHNIC IN ('1','2','3','N','M','P') THEN "Black or Black British"
           WHEN ETHNIC IN ('0','A','B','C', 'T') THEN "White"
           WHEN ETHNIC IN ('4','5','6','L','K','J','H', 'R') THEN "Asian or Asian British"
           WHEN ETHNIC IN ('7','8','W','S') THEN "Other Ethnic Group"
           WHEN ETHNIC IN ('D','E','F','G') THEN "Mixed"
           WHEN ETHNIC IN ("9", "99", "X" , "Z") THEN "Unknown"
           ELSE 'Unknown' END as ETHNIC_GROUP_5_codes,           
CASE WHEN ETHNIC = "M" THEN "Black Caribbean"
           WHEN ETHNIC = "N" THEN "Black African"
           WHEN ETHNIC = "P" THEN "Other Black"
           WHEN ETHNIC = "H" THEN "Indian"
           WHEN ETHNIC = "J" THEN "Pakistani"
           WHEN ETHNIC = "K" THEN "Bangladeshi"
           WHEN ETHNIC = "L" THEN "Other Asian"
           WHEN ETHNIC IN ('D','E','F','G') THEN "Mixed"
           WHEN ETHNIC = "R" THEN "Chinese"
           WHEN ETHNIC IN ('0','A','B','C','T') THEN "White"
           WHEN ETHNIC = 'S' THEN "Other Ethnic Group"
           WHEN ETHNIC IN ("9","99","X","Z") THEN "Unknown"
           ELSE 'Unknown' END as ETHNIC_GROUP_11_codes,
ETHNIC,           
SNOMED_ETHNIC,           
RECORD_DATE, 
dataset 
FROM global_temp.ccu037_ethnicity_assembled """)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM global_temp.ccu037_ethnicity_assembled_multiple_granularities

# COMMAND ----------

drop_table("ccu037_ethnicity_assembled_multiple_granularities")
create_table("ccu037_ethnicity_assembled_multiple_granularities")
