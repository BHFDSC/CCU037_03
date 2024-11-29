# Databricks notebook source
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

# MAGIC %sql
# MAGIC SELECT * FROM dars_nic_391419_j3w9t_collab.ccu037_cvd_outcomes_and_dates

# COMMAND ----------

# MAGIC %md
# MAGIC ### Date of death

# COMMAND ----------

deaths = spark.sql(f'''
    SELECT * FROM dars_nic_391419_j3w9t_collab.deaths_dars_nic_391419_j3w9t_archive ''')
deaths.createOrReplaceGlobalTempView('ccu037_dp_deaths_tmp') 

# COMMAND ----------

death_single = spark.sql(f'''
SELECT * 
FROM 
  (SELECT * , row_number() OVER (PARTITION BY DEC_CONF_NHS_NUMBER_CLEAN_DEID 
                                      ORDER BY REG_DATE desc, REG_DATE_OF_DEATH desc) as death_rank
    FROM dars_nic_391419_j3w9t_collab.deaths_dars_nic_391419_j3w9t_archive) cte
WHERE death_rank = 1
AND DEC_CONF_NHS_NUMBER_CLEAN_DEID IS NOT NULL
and TO_DATE(REG_DATE_OF_DEATH, "yyyyMMdd") > '1900-01-01'
AND TO_DATE(REG_DATE_OF_DEATH, "yyyyMMdd") <= current_date()
''')
death_single.createOrReplaceGlobalTempView('ccu037_dp_single_patient_death')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM global_temp.ccu037_dp_single_patient_death

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE global_temp.ccu037_dp_single_patient_death

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT to_date("REG_DATE_OF_DEATH", 'yyyy-MM-dd') as date FROM global_temp.ccu037_dp_single_patient_death

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW follow_up_dates AS
SELECT a.NHS_NUMBER_DEID, a.covid_date, b.REG_DATE_OF_DEATH
FROM dars_nic_391419_j3w9t_collab.ccu037_cvd_outcomes_with_dates a
LEFT JOIN global_temp.ccu037_dp_single_patient_death b
ON
a.NHS_NUMBER_DEID = b.DEC_CONF_NHS_NUMBER_CLEAN_DEID
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW follow_up_dates_2 AS
SELECT *, 
(CASE WHEN REG_DATE_OF_DEATH <= "2022-06-29" THEN REG_DATE_OF_DEATH
ELSE "2022-06-29" END)
AS follow_up_date
FROM global_temp.follow_up_dates 
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM global_temp.follow_up_dates_2

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_follow_up_availability AS
SELECT *,
(CASE WHEN follow_up_date >= (COVID_DATE + INTERVAl 90 DAY) THEN 1
ELSE 0 END) AS follow_up_90d,
(CASE WHEN follow_up_date >= (COVID_DATE + INTERVAl 180 DAY) THEN 1
ELSE 0 END) AS follow_up_180d,
(CASE WHEN follow_up_date >= (COVID_DATE + INTERVAl 365 DAY) THEN 1
ELSE 0 END) AS follow_up_1y,
(CASE WHEN follow_up_date >= (COVID_DATE + INTERVAl 730 DAY) THEN 1
ELSE 0 END) AS follow_up_2y,
DATEDIFF(follow_up_date, covid_date) AS follow_up_days
FROM global_temp.follow_up_dates_2
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM global_temp.ccu037_follow_up_availability

# COMMAND ----------

drop_table("ccu037_follow_up_availability")
create_table("ccu037_follow_up_availability")

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from dars_nic_391419_j3w9t_collab.ccu037_follow_up_availability where covid_date > to_date(reg_date_of_death, "yyyyMMdd")

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from dars_nic_391419_j3w9t_collab.ccu037_follow_up_availability
