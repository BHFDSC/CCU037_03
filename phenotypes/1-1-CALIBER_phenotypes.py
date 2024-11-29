# Databricks notebook source
# MAGIC %md
# MAGIC # Create skinny table of patients & CALIBER phenotypes
# MAGIC  
# MAGIC **Description**  
# MAGIC
# MAGIC 1. For each terminology in `ccu013_caliber_codelist_master`
# MAGIC 2. Join data source with codelist on `code` to get `phenotype`:
# MAGIC   * 1. `terminology = ICD` -> HES APC DIAG
# MAGIC   * 2. `terminology = OPCS` -> HES APC OP
# MAGIC   * 3. `terminology = SNOMED` -> GDPPR
# MAGIC 3. Unite & agreggate to produce a 'skinny table' of patients, `phenotype` and `date`
# MAGIC   
# MAGIC   
# MAGIC   
# MAGIC **NB this will return all codes up to the last ProductionDate**  
# MAGIC Subsetting, e.g. to before the end of this study `2022-07-29` will be done in subsequent notebooks
# MAGIC
# MAGIC **Project(s)** CCU013
# MAGIC  
# MAGIC **Author(s)** Freya Allery (adapted from original notebook by Chris Tomlinson)
# MAGIC  
# MAGIC **Reviewer(s)** 
# MAGIC  
# MAGIC **Date last updated** 2022-08-08
# MAGIC  
# MAGIC **Date last reviewed** *NA*
# MAGIC  
# MAGIC **Date last run** 2022-08-08
# MAGIC
# MAGIC **Changelog**  
# MAGIC * `21-05-19 ` V1 initial eversion - single first date of code per patient
# MAGIC * `21-07-14` V2 each instance/date of code per patient
# MAGIC * `21-09-08` V3 added parameters for table names + ProductionId
# MAGIC * `21-10-05` V4 added programatic extraction of latest `ProductionDate` + basic tests for QC
# MAGIC  
# MAGIC **Data input**  
# MAGIC * Codelist: 
# MAGIC   * `ccu013_caliber_codelist_master`
# MAGIC * Datasets: (NB working off the raw datasets, not freezes, using ProductionDate)  
# MAGIC   * GDPPR: `dars_nic_391419_j3w9t.gdppr_dars_nic_391419_j3w9t`
# MAGIC   * HES APC: `dars_nic_391419_j3w9t_collab.hes_apc_all_years`
# MAGIC   
# MAGIC   
# MAGIC **Data output**
# MAGIC * `ccu037_caliber` = 'skinny' table of each mention of phenotype per pt
# MAGIC * Intermediate outputs:  
# MAGIC   * `ccu013_caliber_tmp_pts_gdppr`  
# MAGIC   * `ccu013_caliber_tmp_data_apc_icd`  
# MAGIC   * `ccu013_caliber_tmp_data_apc_opcs`  
# MAGIC
# MAGIC   
# MAGIC **Software and versions** `python`
# MAGIC  
# MAGIC **Packages and versions** `pyspark`  

# COMMAND ----------

# MAGIC %run /Workspaces/dars_nic_391419_j3w9t_collab/CCU013/COVID-19-SEVERITY-PHENOTYPING/CCU013_00_helper_functions

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

# Params
# Use the latest ProductionDate
production_date = spark.sql("SELECT MAX(ProductionDate) FROM dars_nic_391419_j3w9t_collab.wrang002b_data_version_batchids").first()[0]
print("ProductionDate:", production_date)

# Table names
gdppr_table = "dars_nic_391419_j3w9t_collab.gdppr_dars_nic_391419_j3w9t_archive" # No non-archive equivalent
hes_apc_table = "dars_nic_391419_j3w9t_collab.hes_apc_all_years_archive" 

# without dars_nic_391419_j3w9t_collab. prefix
output_table = "ccu037_caliber"

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT MAX(ProductionDate) FROM dars_nic_391419_j3w9t_collab.hes_apc_all_years_archive

# COMMAND ----------

HES_ProductionDate = "2022-06-29 00:00:00.000000"

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT MAX(ProductionDate) FROM dars_nic_391419_j3w9t_collab.gdppr_dars_nic_391419_j3w9t_archive

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT NHS_NUMBER_DEID) FROM dars_nic_391419_j3w9t_collab.gdppr_dars_nic_391419_j3w9t_archive WHERE ProductionDate = "2022-06-29 00:00:00.000000"

# COMMAND ----------

from pyspark.sql.functions import array, col, explode, lit, struct
from pyspark.sql import DataFrame
from typing import Iterable

def melt(df: DataFrame, 
        id_vars: Iterable[str], value_vars: Iterable[str], 
        var_name: str="variable", value_name: str="value") -> DataFrame:
    """Convert :class:`DataFrame` from wide to long format."""

    # Create array<struct<variable: str, value: ...>>
    _vars_and_vals = array(*(
        struct(lit(c).alias(var_name), col(c).alias(value_name)) 
        for c in value_vars))

    # Add to the DataFrame and explode
    _tmp = df.withColumn("_vars_and_vals", explode(_vars_and_vals))

    cols = id_vars + [
            col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]]
    return _tmp.select(*cols)

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. GDPPR  
# MAGIC
# MAGIC Changelog:  
# MAGIC * `21/7/14`: 
# MAGIC   * Updated to return every instance of a code per individual, not just the first. 
# MAGIC   * Achieved by commenting out code below (`MIN(a.DATE) as date` and `GROUP BY a.NHS_NUMBER_DEID, b.phenotype, a.code`)

# COMMAND ----------

pts_gdppr = spark.sql(f"""
SELECT 
  a.NHS_NUMBER_DEID as person_id_deid,
  b.phenotype,
  a.DATE as date,
  a.CODE as code,
  'SNOMEDCT' as terminology
FROM
   {gdppr_table} as a
INNER JOIN
  dars_nic_391419_j3w9t_collab.ccu013_caliber_master_codelist as b
ON
  a.CODE = b.code
WHERE
  b.terminology = 'SNOMEDCT'
  AND
  a.ProductionDate == "{production_date}"
  """)

assert pts_gdppr.count() !=0, "Table is empty"
print("Tests passed")

pts_gdppr.createOrReplaceGlobalTempView('ccu013_caliber_tmp_pts_gdppr')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*), COUNT(DISTINCT person_id_deid), COUNT(DISTINCT phenotype)
# MAGIC FROM global_temp.ccu013_caliber_tmp_pts_gdppr
# MAGIC -- 21/05/19: 139898128   30821083   166
# MAGIC -- 21/07/14: 469772311   31131194   166
# MAGIC -- 21/09/08: 477248914   31348834   166
# MAGIC -- 21/10/05: 477248914   31348834   166
# MAGIC -- 22/07/29: 558106047   33665629   153

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. HES APC Diagnoses

# COMMAND ----------

data_apc_icd = spark.sql(f"""
SELECT
  PERSON_ID_DEID as person_id_deid,
  ADMIDATE as date,
  DIAG_4_01, DIAG_4_02, DIAG_4_03, DIAG_4_04, DIAG_4_05, 
  DIAG_4_06, DIAG_4_07, DIAG_4_08, DIAG_4_09, DIAG_4_10, 
  DIAG_4_11, DIAG_4_12, DIAG_4_13, DIAG_4_14, DIAG_4_15, 
  DIAG_4_16, DIAG_4_17, DIAG_4_18, DIAG_4_19, DIAG_4_20
FROM
  {hes_apc_table}
WHERE
  ProductionDate == "{HES_ProductionDate}"
""")

assert data_apc_icd.count() !=0, "Table is empty - may indicate issue with production_date"

data_apc_icd = melt(data_apc_icd, 
           id_vars=['person_id_deid', 'date'], 
           value_vars=['DIAG_4_01', 'DIAG_4_02', 'DIAG_4_03', 'DIAG_4_04', 'DIAG_4_05', 
                       'DIAG_4_06', 'DIAG_4_07', 'DIAG_4_08', 'DIAG_4_09', 'DIAG_4_10', 
                       'DIAG_4_11', 'DIAG_4_12', 'DIAG_4_13', 'DIAG_4_14', 'DIAG_4_15', 
                       'DIAG_4_16', 'DIAG_4_17', 'DIAG_4_18', 'DIAG_4_19', 'DIAG_4_20']
          ) \
  .drop('variable') \
  .withColumnRenamed("value","code") \
  .na.drop() # drop all NAs

assert data_apc_icd.count() != 0, "Table is empty"
assert data_apc_icd.where(col("person_id_deid").isNull()).count() == 0, "person_id_deid has nulls"
assert data_apc_icd.where(col("date").isNull()).count() == 0, "date has nulls"
assert data_apc_icd.where(col("code").isNull()).count() == 0, "code has nulls"
print("Passed tests")

data_apc_icd.createOrReplaceGlobalTempView('ccu013_caliber_tmp_data_apc_icd')

# COMMAND ----------

pts_apc_icd = spark.sql("""
SELECT 
  a.person_id_deid,
  b.phenotype,
  a.date as date,
  a.CODE as code,
  'ICD' as terminology
FROM
   global_temp.ccu013_caliber_tmp_data_apc_icd as a
INNER JOIN
  dars_nic_391419_j3w9t_collab.ccu013_caliber_master_codelist as b
ON
  a.CODE = b.code
WHERE
  b.terminology = 'ICD'
  """)

assert pts_apc_icd.count() != 0, "Table is empty"
assert pts_apc_icd.where(col("person_id_deid").isNull()).count() == 0, "person_id_deid has nulls"
assert pts_apc_icd.where(col("phenotype").isNull()).count() == 0, "phenotype has nulls"
assert pts_apc_icd.where(col("date").isNull()).count() == 0, "date has nulls"
assert pts_apc_icd.where(col("code").isNull()).count() == 0, "code has nulls"
assert pts_apc_icd.where(col("terminology").isNull()).count() == 0, "terminology has nulls"
print("Passed tests")

pts_apc_icd.createOrReplaceGlobalTempView('ccu013_caliber_tmp_pts_apc_icd')

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. HES APC with OPCS4 codes

# COMMAND ----------

data_apc_opcs = spark.sql(f"""
SELECT
  PERSON_ID_DEID as person_id_deid,
  ADMIDATE as date,
  OPERTN_4_01, OPERTN_4_02, OPERTN_4_03, OPERTN_4_04, OPERTN_4_05,
  OPERTN_4_06, OPERTN_4_07, OPERTN_4_08, OPERTN_4_09, OPERTN_4_10,
  OPERTN_4_11, OPERTN_4_12, OPERTN_4_13, OPERTN_4_14, OPERTN_4_15,
  OPERTN_4_16, OPERTN_4_17, OPERTN_4_18, OPERTN_4_19, OPERTN_4_20,
  OPERTN_4_21, OPERTN_4_22, OPERTN_4_23, OPERTN_4_24
FROM
  {hes_apc_table}
WHERE
  ProductionDate == "{HES_ProductionDate}"
""")

assert data_apc_opcs.count() !=0, "Table is empty - may indicate issue with production_date"

data_apc_opcs = melt(data_apc_opcs, 
                     id_vars=['person_id_deid', 'date'], 
                     value_vars=[
                    'OPERTN_4_01', 'OPERTN_4_02', 'OPERTN_4_03', 'OPERTN_4_04', 'OPERTN_4_05',
                    'OPERTN_4_06', 'OPERTN_4_07', 'OPERTN_4_08', 'OPERTN_4_09', 'OPERTN_4_10',
                    'OPERTN_4_11', 'OPERTN_4_12', 'OPERTN_4_13', 'OPERTN_4_14', 'OPERTN_4_15',
                    'OPERTN_4_16', 'OPERTN_4_17', 'OPERTN_4_18', 'OPERTN_4_19', 'OPERTN_4_20',
                    'OPERTN_4_21', 'OPERTN_4_22', 'OPERTN_4_23', 'OPERTN_4_24'
                  ]) \
  .drop('variable') \
  .withColumnRenamed("value","code") \
  .na.drop()

assert data_apc_opcs.count() != 0, "Table is empty"
assert data_apc_opcs.where(col("person_id_deid").isNull()).count() == 0, "person_id_deid has nulls"
assert data_apc_opcs.where(col("date").isNull()).count() == 0, "date has nulls"
assert data_apc_opcs.where(col("code").isNull()).count() == 0, "code has nulls"
print("Passed tests")

data_apc_opcs.createOrReplaceGlobalTempView('ccu013_caliber_tmp_data_apc_opcs')

# COMMAND ----------

pts_apc_opcs = spark.sql("""
SELECT 
  a.person_id_deid,
  b.phenotype,
  a.date as date,
  a.CODE as code,
  'OPCS' as terminology
FROM
   global_temp.ccu013_caliber_tmp_data_apc_opcs as a
INNER JOIN
  dars_nic_391419_j3w9t_collab.ccu013_caliber_master_codelist as b
ON
  a.CODE = b.code
WHERE
  b.terminology = 'OPCS'
  """)

assert pts_apc_opcs.count() != 0, "Table is empty"
assert pts_apc_opcs.where(col("person_id_deid").isNull()).count() == 0, "person_id_deid has nulls"
assert pts_apc_opcs.where(col("phenotype").isNull()).count() == 0, "phenotype has nulls"
assert pts_apc_opcs.where(col("date").isNull()).count() == 0, "date has nulls"
assert pts_apc_opcs.where(col("code").isNull()).count() == 0, "code has nulls"
assert pts_apc_opcs.where(col("terminology").isNull()).count() == 0, "terminology has nulls"
print("Passed tests")

pts_apc_opcs.createOrReplaceGlobalTempView('ccu013_caliber_tmp_pts_apc_opcs')

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Unite & aggregate each dataset's phenotypes
# MAGIC
# MAGIC Plan:
# MAGIC * Union all each source
# MAGIC * Group by ID, select MIN date
# MAGIC * Narrow -> wide
# MAGIC * Sum duplicate entires - i.e. replace >1 with 1
# MAGIC   * These arise where the same diagnosis code is used multiple times on a given day for a given patient  
# MAGIC   * This does NOT represent a burden of illness but coding/administrative details  
# MAGIC   * Therefore if we are to use `n_occurences`/prevalence as a feature (instead of just binary) we need to remove these

# COMMAND ----------

patients = spark.sql("""
SELECT * FROM global_temp.ccu013_caliber_tmp_pts_gdppr
UNION ALL
SELECT * FROM global_temp.ccu013_caliber_tmp_pts_apc_icd
UNION ALL
SELECT * FROM global_temp.ccu013_caliber_tmp_pts_apc_opcs
""") \
.dropDuplicates()

assert patients.count() != 0, "Table is empty"
assert patients.select('terminology').distinct().count() == 3, "Doesn't contain 3 distinct terminologies, should be just SNOMEDCT, ICD, OPCS"
assert patients.select('phenotype').distinct().count() >=  270, "Data contains less than 270 distinct phenotypes"
print("Passed checks")

patients.createOrReplaceGlobalTempView(output_table)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   COUNT(*) as mentions, 
# MAGIC   COUNT(DISTINCT person_id_deid) as unique_pts, 
# MAGIC   COUNT(DISTINCT phenotype) as phenotypes,
# MAGIC   COUNT(DISTINCT terminology) as terminologies
# MAGIC FROM
# MAGIC   global_temp.ccu037_caliber

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   COUNT(*) as mentions, 
# MAGIC   COUNT(DISTINCT person_id_deid) as unique_pts, 
# MAGIC   COUNT(DISTINCT phenotype) as phenotypes,
# MAGIC   COUNT(DISTINCT terminology) as terminologies
# MAGIC FROM
# MAGIC   global_temp.ccu013_caliber_patients

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT * FROM global_temp.ccu037_caliber

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Write table

# COMMAND ----------

drop_table(output_table)
create_table(output_table) 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT phenotype FROM dars_nic_391419_j3w9t_collab.ccu037_caliber
# MAGIC WHERE phenotype LIKE '%malignancy%'
# MAGIC OR phenotype LIKE "%polycythaemia_vera%"
# MAGIC OR phenotype LIKE "%hodgkin_lymphoma%"
# MAGIC OR phenotype LIKE "%myelodysplastic_syndromes%"
# MAGIC OR phenotype LIKE "%multiple_myeloma_and_malignant_plasma_cell_neoplasms%"
# MAGIC OR phenotype LIKE "%leukaemia%"
# MAGIC OR phenotype LIKE "%non-hodgkin_lymphoma%"
# MAGIC OR phenotype LIKE "%monoclonal_gammopathy_of_undetermined_significance_mgus%"

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. Limit phenotype by type and date 

# COMMAND ----------

# selecting phenotype predictors from protocol
spark.sql(f""" 
CREATE OR REPLACE GLOBAL TEMP VIEW predictors1 AS
SELECT person_id_deid, phenotype, date
FROM dars_nic_391419_j3w9t_collab.ccu037_caliber
WHERE PHENOTYPE LIKE "%copd%"
   OR PHENOTYPE LIKE "%hypertension%"
   OR PHENOTYPE LIKE "%obesity%"
   OR PHENOTYPE LIKE "%diabetes%"
   OR PHENOTYPE LIKE "%fracture_of_hip%"
   OR PHENOTYPE LIKE "%fracture_of_wrist%"
   OR PHENOTYPE LIKE "%osteoporosis%"
   OR PHENOTYPE LIKE "%autoimmune_liver_disease%"
   OR PHENOTYPE LIKE "%dementia%"
   OR PHENOTYPE LIKE "%alcohol_problems%"
   OR PHENOTYPE LIKE "%alcoholic_liver_disease%"
""")

# COMMAND ----------

# grouping cancers
spark.sql(f""" 
CREATE OR REPLACE GLOBAL TEMP VIEW predictors2 AS
SELECT person_id_deid, "cancer" AS phenotype, date
FROM dars_nic_391419_j3w9t_collab.ccu037_caliber
WHERE phenotype LIKE '%malignancy%'
   OR phenotype LIKE "%polycythaemia_vera%"
   OR phenotype LIKE "%hodgkin_lymphoma%"
   OR phenotype LIKE "%myelodysplastic_syndromes%"
   OR phenotype LIKE "%multiple_myeloma_and_malignant_plasma_cell_neoplasms%"
   OR phenotype LIKE "%leukaemia%"
   OR phenotype LIKE "%non-hodgkin_lymphoma%"
   OR phenotype LIKE "%monoclonal_gammopathy_of_undetermined_significance_mgus%"
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM global_temp.predictors2

# COMMAND ----------

spark.sql(f""" 
CREATE OR REPLACE GLOBAL TEMP VIEW predictors AS
SELECT * FROM
global_temp.predictors1
UNION ALL
SELECT * FROM
global_temp.predictors2
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT phenotype FROM global_temp.predictors

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT(phenotype), COUNT(phenotype) AS count FROM global_temp.predictors GROUP BY phenotype
# MAGIC ORDER BY count DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT(phenotype), COUNT(phenotype) AS count FROM global_temp.predictors GROUP BY phenotype
# MAGIC ORDER BY count DESC

# COMMAND ----------

# all post covid cvd events
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_variables_phenotypes_list_final3 AS
    SELECT person_id_deid, phenotype AS predictor FROM
    (SELECT a.person_id_deid, a.phenotype, a.date, b.date_first_covid_event
    FROM global_temp.predictors AS a
    LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID)
    WHERE date < date_first_covid_event
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT(predictor), COUNT(predictor) AS count FROM global_temp.ccu037_variables_phenotypes_list_final2 GROUP BY predictor
# MAGIC ORDER BY count DESC

# COMMAND ----------

drop_table("ccu037_variables_phenotypes_list_final3")
create_table("ccu037_variables_phenotypes_list_final3")

# COMMAND ----------

phenotypes = spark.sql(f"""
SELECT person_id_deid, predictor, 1 as value FROM GLOBAL_TEMP.predictors_filtered """)

# COMMAND ----------

phenotypes = phenotypes \
  .to_koalas() \
  .pivot(index='person_id_deid', 
         columns='predictor', 
         values='value') \
  .fillna(0) \
  .reset_index() \
  .to_spark()
# Reset index to breakout ids to separate col

# COMMAND ----------

display(phenotypes)
