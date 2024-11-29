# Databricks notebook source
#Define necessary functions

#Define create table function by Sam Hollings
# Source: Workspaces/dars_nic_391419_j3w9t_collab/DATA_CURATION_wrang000_functions

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
# MAGIC SELECT * FROM dars_nic_391419_j3w9t.primary_care_meds_dars_nic_391419_j3w9t

# COMMAND ----------

# MAGIC %md
# MAGIC # Anti-diabetic therapies

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW antidiabeticBNF AS
# MAGIC SELECT * FROM VALUES 
# MAGIC ('0601023A0','Acarbose'),('0601023AS','Albiglutide'),('0601023AK','Alogliptin'),('0601023AJ','Alogliptin/metformin'),('0601023AM','Canagliflozin'),('0601023AP','Canagliflozin/metformin'),('0601021E0','Chlorpropamide'),('0601023AG','Dapagliflozin'),('0601023AL','Dapagliflozin/metformin'),('0601023AQ','Dulaglutide'),('0601023AN','Empagliflozin'),('0601023AY','Empagliflozin/linagliptin'),('0601023AR','Empagliflozin/metformin'),('0601023AX','Ertugliflozin'),('0601023Y0','Exenatide'),('0601021H0','Glibenclamide'),('0601021M0','Gliclazide'),('0601021A0','Glimepiride'),('0601021P0','Glipizide'),('0601023I0','Guar gum'),('0601023AU','Ins degludec/liraglutide'),('0601023AE','Linagliptin'),('0601023AF','Linagliptin/metformin'),('0601023AB','Liraglutide'),('0601023AI','Lixisenatide'),('0601022B0','Metformin hydrochloride'),('0601023W0','Metformin hydrochloride/pioglitazone'),('0601023V0','Metformin hydrochloride/rosiglitazone'),('0601023AD','Metformin hydrochloride/sitagliptin'),('0601023Z0','Metformin hydrochloride/vildagliptin'),('0601023U0','Nateglinide'),('0601023B0','Pioglitazone hydrochloride'),('0601023R0','Repaglinide'),('0601023S0','Rosiglitazone'),('0601023AC','Saxagliptin'),('0601023AV','Saxagliptin/dapagliflozin'),('0601023AH','Saxagliptin/metformin'),('0601023AW','Semaglutide'),('0601023X0','Sitagliptin'),('0601021X0','Tolbutamide'),('0601023AA','Vildagliptin')
# MAGIC AS tab(BNFcode, dantidiabetic_name);

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW anti_diabetic_drugs AS
# MAGIC SELECT * FROM dars_nic_391419_j3w9t.primary_care_meds_dars_nic_391419_j3w9t WHERE Person_ID_DEID IS NOT NULL
# MAGIC AND left(PrescribedBNFCode, 9) IN (SELECT BNFcode FROM global_temp.antidiabeticBNF)

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW anti_diabetic_drugs AS
    SELECT Person_ID_DEID, "anti_diabetic_drugs" as predictor FROM
    (SELECT a.person_id_deid, a.ProcessingPeriodDate, b.date_first_covid_event
    FROM global_temp.anti_diabetic_drugs AS a
    LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID) 
    WHERE (date_first_covid_event - INTERVAL 365 DAY) < ProcessingPeriodDate AND ProcessingPeriodDate < date_first_covid_event  """)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT Person_ID_DEID) FROM global_temp.anti_diabetic_drugs

# COMMAND ----------

# MAGIC %md
# MAGIC # Anti-hypertensive drugs

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW antihypertensiveBNF AS
# MAGIC SELECT * FROM VALUES 
# MAGIC ('Clonidine hydrochloride','0205020E0'),
# MAGIC ('Guanfacine hydrochloride','0205020G0'),
# MAGIC ('Methyldopa','0205020H0'),
# MAGIC ('Moxonidine','0205020M0'),
# MAGIC ('Ambrisentan','0205010X0'),
# MAGIC ('Bosentan','0205010U0'),
# MAGIC ('Diazoxide','0205010E0'),
# MAGIC ('Hydralazine hydrochloride','0205010J0'),
# MAGIC ('Iloprost','0205010V0'),
# MAGIC ('Macitentan','0205010AA'),
# MAGIC ('Minoxidil','0205010N0'),
# MAGIC ('Riociguat','0205010AB'),
# MAGIC ('Sildenafil (Vasodilator Antihypertensive)','0205010Y0'),
# MAGIC ('Sitaxentan sodium','0205010W0'),
# MAGIC ('Tadalafil (Vasodilator Antihypertensive)','0205010Z0'),
# MAGIC ('Vericiguat','0205010AC')
# MAGIC AS tab(antihypertensive_name,BNFcode);

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE GLOBAL TEMP VIEW anti_hypertensive_drugs AS
# MAGIC SELECT * FROM dars_nic_391419_j3w9t.primary_care_meds_dars_nic_391419_j3w9t 
# MAGIC WHERE (Person_ID_DEID IS NOT NULL AND left(PrescribedBNFCode, 9) IN (SELECT BNFcode FROM global_temp.antihypertensiveBNF))

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW anti_hypertensive_drugs AS
    SELECT Person_ID_DEID, "anti_hypertensive_drugs" as predictor FROM
    (SELECT a.person_id_deid, a.ProcessingPeriodDate, b.date_first_covid_event
    FROM global_temp.anti_hypertensive_drugs AS a
    LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID) 
    WHERE (date_first_covid_event - INTERVAL 365 DAY) < ProcessingPeriodDate AND ProcessingPeriodDate < date_first_covid_event  """)

# COMMAND ----------

# MAGIC %md #Anti-thrombotic or anti-coagulant treatment
# MAGIC
# MAGIC -- Antithrombotic drugs in routine use include antiplatelet drugs and anticoagulants 
# MAGIC
# MAGIC ie., to include anti-coagulants and antiplatelet drugs

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW anti_coagulant_drugs AS
# MAGIC SELECT * 
# MAGIC FROM
# MAGIC dars_nic_391419_j3w9t.primary_care_meds_dars_nic_391419_j3w9t WHERE Person_ID_DEID IS NOT NULL
# MAGIC AND (left(PrescribedBNFCode, 6) = '020802'
# MAGIC        AND NOT (left(PrescribedBNFCode, 8) = '0208020I' OR left(PrescribedBNFCode, 8) = '0208020W'))

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW anti_coagulant_drugs AS
    SELECT Person_ID_DEID, "anti_coagulant_drugs" as predictor FROM
    (SELECT a.person_id_deid, a.ProcessingPeriodDate, b.date_first_covid_event
    FROM global_temp.anti_coagulant_drugs AS a
    LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID) 
    WHERE (date_first_covid_event - INTERVAL 365 DAY) < ProcessingPeriodDate AND ProcessingPeriodDate < date_first_covid_event  """)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW anti_platelet_drugs AS
# MAGIC SELECT * 
# MAGIC FROM dars_nic_391419_j3w9t.primary_care_meds_dars_nic_391419_j3w9t WHERE Person_ID_DEID IS NOT NULL
# MAGIC AND (left(PrescribedBNFCode, 7) = '0209000')

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW anti_platelet_drugs AS
    SELECT Person_ID_DEID, "anti_platelet_drugs" as predictor FROM
    (SELECT a.person_id_deid, a.ProcessingPeriodDate, b.date_first_covid_event
    FROM global_temp.anti_platelet_drugs AS a
    LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID) 
    WHERE (date_first_covid_event - INTERVAL 365 DAY) < ProcessingPeriodDate AND ProcessingPeriodDate < date_first_covid_event  """)

# COMMAND ----------

# MAGIC %md
# MAGIC # Statins

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW statins AS
# MAGIC SELECT *
# MAGIC FROM  dars_nic_391419_j3w9t.primary_care_meds_dars_nic_391419_j3w9t WHERE Person_ID_DEID IS NOT NULL
# MAGIC AND (left(PrescribedBNFCode, 7) =  "0212000"   AND NOT (left(PrescribedBNFCode, 9) = '0212000U0'))

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW statins AS
    SELECT Person_ID_DEID, "statins" as predictor FROM
    (SELECT a.person_id_deid, a.ProcessingPeriodDate, b.date_first_covid_event
    FROM global_temp.statins AS a
    LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID) 
    WHERE (date_first_covid_event - INTERVAL 365 DAY) < ProcessingPeriodDate AND ProcessingPeriodDate < date_first_covid_event  """)

# COMMAND ----------

# MAGIC %md
# MAGIC # Join all medications together

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_medications_list_final AS
SELECT * FROM global_temp.anti_diabetic_drugs
UNION ALL
SELECT * FROM global_temp.anti_coagulant_drugs
UNION ALL
SELECT * FROM global_temp.anti_platelet_drugs
UNION ALL
SELECT * FROM global_temp.statins
UNION ALL
SELECT * FROM global_temp.anti_hypertensive_drugs
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM global_temp.ccu037_medications_list_final

# COMMAND ----------

drop_table("ccu037_medications_list_final")
create_table("ccu037_medications_list_final")

# COMMAND ----------

# MAGIC %md
# MAGIC # Pivot to binary

# COMMAND ----------

medications = spark.sql(f"""
SELECT person_id_deid, predictor, 1 as value FROM GLOBAL_TEMP.ccu037_medications_list """)

# COMMAND ----------

medications = medications \
  .to_koalas() \
  .pivot(index='person_id_deid', 
         columns='predictor', 
         values='value') \
  .fillna(0) \
  .reset_index() \
  .to_spark()
# Reset index to breakout ids to separate col

# COMMAND ----------

display(medications)
