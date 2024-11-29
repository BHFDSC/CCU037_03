# Databricks notebook source
# MAGIC %md
# MAGIC # Ethnicity table unassembled
# MAGIC  
# MAGIC **Description** 
# MAGIC
# MAGIC This notebook runs a list of `SQL` queries to:  
# MAGIC   
# MAGIC 1. Rank the prioritisation of different ethnicity codes from different sources
# MAGIC 2. Create an unassembled table with all of the ethnicity information
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

# MAGIC %md
# MAGIC # 1. Load in tables

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1. Production dates

# COMMAND ----------

gdppr_prod_date = "2022-05-30 14:32:44.253705"
hes_prod_date = "2022-03-31 00:00:00.000000"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2. GDPPR SNOMED

# COMMAND ----------

#SNOMED codes, 1 as first choice ranking
spark.sql(f"""CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_gdppr_ethnicity_SNOMED AS 
               SELECT NHS_NUMBER_DEID,
                      eth.ConceptId as SNOMED_ETHNIC,
                      eth.PrimaryCode as ETHNIC,
                      DATE as RECORD_DATE,
                      'GDPPR_snomed' as dataset,
                      1 as eth_rank
                FROM dars_nic_391419_j3w9t_collab.gdppr_dars_nic_391419_j3w9t_archive as gdppr
                      INNER JOIN dss_corporate.gdppr_ethnicity_mappings eth on gdppr.CODE = eth.ConceptId
                WHERE gdppr.ProductionDate = '{gdppr_prod_date}' """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3. GDPPR Primary Codes 

# COMMAND ----------

# Primary codes, 2 as second choice ranking
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_gdppr_ethnicity AS 
       SELECT NHS_NUMBER_DEID, 
              gdppr.ETHNIC, REPORTING_PERIOD_END_DATE as RECORD_DATE, 
              'GDPPR' as dataset, 
              2 as eth_rank 
       FROM dars_nic_391419_j3w9t_collab.gdppr_dars_nic_391419_j3w9t_archive as gdppr 
       WHERE ProductionDate = "{gdppr_prod_date}" """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.4. HES

# COMMAND ----------

spark.sql(f"""
 CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_all_hes_apc_ethnicity AS
 SELECT DISTINCT PERSON_ID_DEID as NHS_NUMBER_DEID, 
      ETHNOS as ETHNIC, 
      EPISTART as RECORD_DATE, 
      "hes_apc" as dataset,
      3 as eth_rank
  FROM dars_nic_391419_j3w9t_collab.hes_apc_all_years_archive 
  WHERE ProductionDate = "{hes_prod_date}" """)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Unassembled output table

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_ethnicity_unassembled as
# MAGIC SELECT NHS_NUMBER_DEID, ETHNIC, NULL AS SNOMED_ETHNIC, RECORD_DATE, dataset, eth_rank 
# MAGIC FROM global_temp.ccu037_gdppr_ethnicity
# MAGIC UNION ALL
# MAGIC SELECT NHS_NUMBER_DEID, ETHNIC, SNOMED_ETHNIC, RECORD_DATE, dataset, eth_rank 
# MAGIC FROM global_temp.ccu037_gdppr_ethnicity_SNOMED
# MAGIC UNION ALL
# MAGIC SELECT NHS_NUMBER_DEID, ETHNIC, null AS SNOMED_ETHNIC, RECORD_DATE, dataset, eth_rank 
# MAGIC FROM global_temp.ccu037_all_hes_apc_ethnicity

# COMMAND ----------

drop_table('ccu037_ethnicity_unassembled')
create_table('ccu037_ethnicity_unassembled') 
