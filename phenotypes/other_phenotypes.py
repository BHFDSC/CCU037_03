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

import databricks.koalas as ks
import pandas as pd

from functools import reduce
from operator import add
from pyspark.sql.functions import lit, col

def row_sum_across(*cols):
    return reduce(add, cols, lit(0))

# COMMAND ----------

# MAGIC %run /Workspaces/dars_nic_391419_j3w9t_collab/CCU013/COVID-19-SEVERITY-PHENOTYPING/CCU013_00_helper_functions

# COMMAND ----------

project_prefix = "ccu037_02"
ProductionDate = "2022-06-29 00:00:00.000000"

# COMMAND ----------

hes_apc_table = "dars_nic_391419_j3w9t_collab.hes_apc_all_years_archive"
gdppr_table = "dars_nic_391419_j3w9t_collab.gdppr_dars_nic_391419_j3w9t_archive"
medications_table = "dars_nic_391419_j3w9t.primary_care_meds_dars_nic_391419_j3w9t"

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM 
# MAGIC dars_nic_391419_j3w9t_collab.hes_apc_all_years_archive LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM 
# MAGIC dars_nic_391419_j3w9t_collab.gdppr_dars_nic_391419_j3w9t_archive LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM 
# MAGIC dars_nic_391419_j3w9t_collab.gdppr_dars_nic_391419_j3w9t_archive LIMIT 10

# COMMAND ----------

data_apc_icd = spark.sql(f"""
SELECT
  PERSON_ID_DEID as NHS_NUMBER_DEID,
  ADMIDATE as date,
  DIAG_4_01, DIAG_4_02, DIAG_4_03, DIAG_4_04, DIAG_4_05, 
  DIAG_4_06, DIAG_4_07, DIAG_4_08, DIAG_4_09, DIAG_4_10, 
  DIAG_4_11, DIAG_4_12, DIAG_4_13, DIAG_4_14, DIAG_4_15, 
  DIAG_4_16, DIAG_4_17, DIAG_4_18, DIAG_4_19, DIAG_4_20
FROM
  {hes_apc_table}
WHERE
  ProductionDate == "{ProductionDate}"
""")

assert data_apc_icd.count() !=0, "Table is empty - may indicate issue with production_date"

data_apc_icd = melt(data_apc_icd, 
           id_vars=['NHS_NUMBER_DEID', 'date'], 
           value_vars=['DIAG_4_01', 'DIAG_4_02', 'DIAG_4_03', 'DIAG_4_04', 'DIAG_4_05', 
                       'DIAG_4_06', 'DIAG_4_07', 'DIAG_4_08', 'DIAG_4_09', 'DIAG_4_10', 
                       'DIAG_4_11', 'DIAG_4_12', 'DIAG_4_13', 'DIAG_4_14', 'DIAG_4_15', 
                       'DIAG_4_16', 'DIAG_4_17', 'DIAG_4_18', 'DIAG_4_19', 'DIAG_4_20']
          ) \
  .drop('variable') \
  .withColumnRenamed("value","code") \
  .na.drop() # drop all NAs

assert data_apc_icd.count() != 0, "Table is empty"
assert data_apc_icd.where(col("NHS_NUMBER_DEID").isNull()).count() == 0, "person_id_deid has nulls"
assert data_apc_icd.where(col("date").isNull()).count() == 0, "date has nulls"
assert data_apc_icd.where(col("code").isNull()).count() == 0, "code has nulls"
print("Passed tests")

data_apc_icd.createOrReplaceGlobalTempView('hes_icd')

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE GLOBAL TEMP VIEW gdppr_snomed AS
# MAGIC SELECT NHS_NUMBER_DEID, DATE AS date, CODE AS code
# MAGIC FROM dars_nic_391419_j3w9t_collab.gdppr_dars_nic_391419_j3w9t_archive WHERE ProductionDate = ProductionDate AND NHS_NUMBER_DEID IS NOT NULL

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW gdppr_meds AS
# MAGIC SELECT Person_ID_DEID AS NHS_NUMBER_DEID, ProcessingPeriodDate AS date, PrescribedBNFCode AS code FROM dars_nic_391419_j3w9t.primary_care_meds_dars_nic_391419_j3w9t WHERE Person_ID_DEID IS NOT NULL

# COMMAND ----------

# MAGIC %md
# MAGIC # 1.Mental illness
# MAGIC ### Anytime before

# COMMAND ----------

## Schizophrenia:
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW {project_prefix}_schizophrenia_codelist AS SELECT * FROM VALUES
  ("F20","Schizophrenia","ICD10","schizophrenia"),
  ("58214004","Schizophrenia (disorder)","SNOMED","schizophrenia"),
  ("111484002","Undifferentiated schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("1204417003","Early onset schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("12939007","Chronic disorganized schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("16990005","Subchronic schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("191527001","Simple schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("191542003","Catatonic schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("191577003","Cenesthopathic schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("26025008","Residual schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("27387000","Subchronic disorganized schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("58214004","Schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("29599000","Chronic undifferentiated schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("31658008","Chronic paranoid schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("35252006","Disorganized schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("416340002","Late onset schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("42868002","Subchronic catatonic schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("4926007","Schizophrenia in remission (disorder)","SNOMED","schizophrenia"), 
  ("64905009","Paranoid schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("68995007","Chronic catatonic schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("71103003","Chronic residual schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("76566000","Subchronic residual schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("111482003","Subchronic schizophrenia with acute exacerbations (disorder)","SNOMED","schizophrenia"), 
  ("111483008","Catatonic schizophrenia in remission (disorder)","SNOMED","schizophrenia"), 
  ("14291003","Subchronic disorganized schizophrenia with acute exacerbations (disorder)","SNOMED","schizophrenia"), 
  ("191531007","Acute exacerbation of chronic schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("31373002","Disorganized schizophrenia in remission (disorder)","SNOMED","schizophrenia"), 
  ("39610001","Undifferentiated schizophrenia in remission (disorder)","SNOMED","schizophrenia"), 
  ("51133006","Residual schizophrenia in remission (disorder)","SNOMED","schizophrenia"), 
  ("63181006","Paranoid schizophrenia in remission (disorder)","SNOMED","schizophrenia"), 
  ("79866005","Subchronic paranoid schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("85861002","Subchronic undifferentiated schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("191547009","Acute exacerbation of subchronic catatonic schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("191548004","Acute exacerbation of chronic catatonic schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("191554003","Acute exacerbation of subchronic paranoid schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("191555002","Acute exacerbation of chronic paranoid schizophrenia (disorder)","SNOMED","schizophrenia"), 
  ("278853003","Acute schizophrenia-like psychotic disorder (disorder)","SNOMED","schizophrenia"), 
  ("30336007","Chronic residual schizophrenia with acute exacerbations (disorder)","SNOMED","schizophrenia"), 
  ("35218008","Chronic disorganized schizophrenia with acute exacerbation (disorder)","SNOMED","schizophrenia"), 
  ("7025000","Subchronic undifferentiated schizophrenia with acute exacerbations (disorder)","SNOMED","schizophrenia"), 
  ("70814008","Subchronic residual schizophrenia with acute exacerbations (disorder)","SNOMED","schizophrenia"), 
  ("79204003","Chronic undifferentiated schizophrenia with acute exacerbations (disorder)","SNOMED","schizophrenia"), 
  ("191526005","Schizophrenic disorders (disorder)","SNOMED","schizophrenia"),
  ("22264002","Catatonic rigidity (finding)","SNOMED","schizophrenia"),
  ("247917007","Catatonia (finding)","SNOMED","schizophrenia"),
  ("36158005","Schizophreniform disorder with good prognostic features (disorder)","SNOMED","schizophrenia"),
  ("36734001","Catatonic posturing (finding)","SNOMED","schizophrenia"),
  ("441833000","Lethal catatonia (disorder)","SNOMED","schizophrenia"),
  ("46153004","Catatonic stupor (finding)","SNOMED","schizophrenia"),
  ("55736003","Schizophreniform disorder without good prognostic features (disorder)","SNOMED","schizophrenia"),
  ("68434003","Catatonic excitement (finding)","SNOMED","schizophrenia"),
  ("88975006","Schizophreniform disorder (disorder)","SNOMED","schizophrenia"),
  ("161468000","History of schizophrenia (situation)","SNOMED","schizophrenia"),
  ("83746006","Chronic schizophrenia (disorder)","SNOMED","schizophrenia")
  AS {project_prefix}_schizophrenia_codelist (code, term, system, codelist) """)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW schizophrenia AS
# MAGIC SELECT NHS_NUMBER_DEID, "schizophrenia" as predictor, date FROM global_temp.hes_icd 
# MAGIC WHERE CODE IN (SELECT code FROM global_temp.ccu037_02_schizophrenia_codelist)
# MAGIC UNION ALL
# MAGIC SELECT NHS_NUMBER_DEID, "schizophrenia" as predictor, date FROM global_temp.gdppr_snomed
# MAGIC WHERE CODE IN (SELECT code FROM global_temp.ccu037_02_schizophrenia_codelist)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE GLOBAL TEMP VIEW schizophrenia1 AS
# MAGIC SELECT DISTINCT NHS_NUMBER_DEID, predictor
# MAGIC FROM
# MAGIC (SELECT b.NHS_NUMBER_DEID, a.predictor, a.date, b.date_first_covid_event
# MAGIC FROM
# MAGIC (SELECT
# MAGIC   NHS_NUMBER_DEID,
# MAGIC   min(date) as date, predictor
# MAGIC FROM
# MAGIC   global_temp.schizophrenia
# MAGIC GROUP BY
# MAGIC   NHS_NUMBER_DEID, predictor) a
# MAGIC LEFT JOIN 
# MAGIC   dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria b
# MAGIC ON
# MAGIC   a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID)
# MAGIC WHERE date < date_first_covid_event

# COMMAND ----------

# spark.sql(f"""
# CREATE OR REPLACE GLOBAL TEMP VIEW schizophrenia1 AS
#     SELECT NHS_NUMBER_DEID, predictor FROM
#     (SELECT a.NHS_NUMBER_DEID, a.date, a.predictor, b.date_first_covid_event
#     FROM global_temp.schizophrenia AS a
#     LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
#     ON a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID) 
#     WHERE date < date_first_covid_event  """)

# COMMAND ----------

## Bipolar disorder:
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW {project_prefix}_bipolardisorder_codelist AS SELECT * FROM VALUES
  ("F31","Bipolar affective disorder","ICD10","bipolardisorder"),
  ("13746004","Bipolar disorder (disorder)","SNOMED","bipolardisorder"),
  ("133091000119105","Rapid cycling bipolar I disorder (disorder)","SNOMED","bipolardisorder"),
  ("13313007","Mild bipolar disorder (disorder)","SNOMED","bipolardisorder"),
  ("13746004","Bipolar disorder (disorder)","SNOMED","bipolardisorder"),
  ("16238741000119105","Bipolar disorder caused by drug (disorder)","SNOMED","bipolardisorder"),
  ("371596008","Bipolar I disorder (disorder)","SNOMED","bipolardisorder"),
  ("371599001","Severe bipolar I disorder (disorder)","SNOMED","bipolardisorder"),
  ("371600003","Severe bipolar disorder (disorder)","SNOMED","bipolardisorder"),
  ("4441000","Severe bipolar disorder with psychotic features (disorder)","SNOMED","bipolardisorder"),
  ("49468007","Depressed bipolar I disorder (disorder)","SNOMED","bipolardisorder"),
  ("79584002","Moderate bipolar disorder (disorder)","SNOMED","bipolardisorder"),
  ("16506000","Mixed bipolar I disorder (disorder)","SNOMED","bipolardisorder"),
  ("191636007","Mixed bipolar affective disorder (disorder)","SNOMED","bipolardisorder"),
  ("26530004","Severe bipolar disorder with psychotic features, mood-incongruent (disorder)","SNOMED","bipolardisorder"),
  ("371604007","Severe bipolar II disorder (disorder)","SNOMED","bipolardisorder"),
  ("51637008","Chronic bipolar I disorder, most recent episode depressed (disorder)","SNOMED","bipolardisorder"),
  ("723903001","Bipolar type I disorder currently in full remission (disorder)","SNOMED","bipolardisorder"),
  ("767631007","Bipolar disorder, most recent episode depression (disorder)","SNOMED","bipolardisorder"),
  ("767636002","Bipolar I disorder, most recent episode depression (disorder)","SNOMED","bipolardisorder"),
  ("83225003","Bipolar II disorder (disorder)","SNOMED","bipolardisorder"),
  ("85248005","Bipolar disorder in remission (disorder)","SNOMED","bipolardisorder"),
  ("191638008","Mixed bipolar affective disorder, mild (disorder)","SNOMED","bipolardisorder"),
  ("191639000","Mixed bipolar affective disorder, moderate (disorder)","SNOMED","bipolardisorder"),
  ("192362008","Bipolar affective disorder, current episode mixed (disorder)","SNOMED","bipolardisorder"),
  ("271000119101","Severe mixed bipolar I disorder (disorder)","SNOMED","bipolardisorder"),
  ("35481005","Mixed bipolar I disorder in remission (disorder)","SNOMED","bipolardisorder"),
  ("40926005","Moderate mixed bipolar I disorder (disorder)","SNOMED","bipolardisorder"),
  ("41836007","Bipolar disorder in full remission (disorder)","SNOMED","bipolardisorder"),
  ("43769008","Mild mixed bipolar I disorder (disorder)","SNOMED","bipolardisorder"),
  ("45479006","Manic bipolar I disorder in remission (disorder)","SNOMED","bipolardisorder"),
  ("75752004","Bipolar I disorder, most recent episode depressed with melancholic features (disorder)","SNOMED","bipolardisorder"),
  ("10981006","Severe mixed bipolar I disorder with psychotic features (disorder)","SNOMED","bipolardisorder"),
  ("111485001","Mixed bipolar I disorder in full remission (disorder)","SNOMED","bipolardisorder"),
  ("191641004","Mixed bipolar affective disorder, severe, with psychosis (disorder)","SNOMED","bipolardisorder"),
  ("191643001","Mixed bipolar affective disorder, in full remission (disorder)","SNOMED","bipolardisorder"),
  ("22121000","Depressed bipolar I disorder in full remission (disorder)","SNOMED","bipolardisorder"),
  ("30935000","Manic bipolar I disorder in full remission (disorder)","SNOMED","bipolardisorder"),
  ("53607008","Depressed bipolar I disorder in remission (disorder)","SNOMED","bipolardisorder"),
  ("5703000","Bipolar disorder in partial remission (disorder)","SNOMED","bipolardisorder"),
  ("68569003","Manic bipolar I disorder (disorder)","SNOMED","bipolardisorder"),
  ("789061003","Rapid cycling bipolar II disorder (disorder)","SNOMED","bipolardisorder"),
  ("1196001","Chronic bipolar II disorder, most recent episode major depressive (disorder)","SNOMED","bipolardisorder"),
  ("16295005","Bipolar II disorder, most recent episode major depressive (disorder)","SNOMED","bipolardisorder"),
  ("31446002","Bipolar I disorder, most recent episode hypomanic (disorder)","SNOMED","bipolardisorder"),
  ("3530005","Bipolar I disorder, single manic episode, in full remission (disorder)","SNOMED","bipolardisorder"),
  ("36583000","Mixed bipolar I disorder in partial remission (disorder)","SNOMED","bipolardisorder"),
  ("46229002","Severe mixed bipolar I disorder without psychotic features (disorder)","SNOMED","bipolardisorder"),
  ("49512000","Depressed bipolar I disorder in partial remission (disorder)","SNOMED","bipolardisorder"),
  ("63249007","Manic bipolar I disorder in partial remission (disorder)","SNOMED","bipolardisorder"),
  ("75360000","Bipolar I disorder, single manic episode, in remission (disorder)","SNOMED","bipolardisorder"),
  ("767633005","Bipolar affective disorder, most recent episode mixed (disorder)","SNOMED","bipolardisorder"),
  ("191618007","Bipolar affective disorder, current episode manic (disorder)","SNOMED","bipolardisorder"),
  ("191620005","Bipolar affective disorder, currently manic, mild (disorder)","SNOMED","bipolardisorder"),
  ("191621009","Bipolar affective disorder, currently manic, moderate (disorder)","SNOMED","bipolardisorder"),
  ("191629006","Bipolar affective disorder, currently depressed, mild (disorder)","SNOMED","bipolardisorder"),
  ("191630001","Bipolar affective disorder, currently depressed, moderate (disorder)","SNOMED","bipolardisorder"),
  ("35846004","Moderate bipolar II disorder, most recent episode major depressive (disorder)","SNOMED","bipolardisorder"),
  ("48937005","Bipolar II disorder, most recent episode hypomanic (disorder)","SNOMED","bipolardisorder"),
  ("70546001","Severe bipolar disorder with psychotic features, mood-congruent [disorder]","SNOMED","bipolardisorder"),
  ("71294008","Mild bipolar II disorder, most recent episode major depressive (disorder)","SNOMED","bipolardisorder"),
  ("723905008","Bipolar type II disorder currently in full remission (disorder)","SNOMED","bipolardisorder"),
  ("10875004","Severe mixed bipolar I disorder with psychotic features, mood-incongruent (disorder)","SNOMED","bipolardisorder"),
  ("23741000119105","Severe manic bipolar I disorder (disorder)","SNOMED","bipolardisorder"),
  ("261000119107","Severe depressed bipolar I disorder (disorder)","SNOMED","bipolardisorder"),
  ("53049002","Severe bipolar disorder without psychotic features (disorder)","SNOMED","bipolardisorder"),
  ("66631006","Moderate depressed bipolar I disorder (disorder)","SNOMED","bipolardisorder"),
  ("71984005","Mild manic bipolar I disorder (disorder)","SNOMED","bipolardisorder"),
  ("74686005","Mild depressed bipolar I disorder (disorder)","SNOMED","bipolardisorder"),
  ("767632000","Bipolar disorder, most recent episode manic (disorder)","SNOMED","bipolardisorder"),
  ("82998009","Moderate manic bipolar I disorder (disorder)","SNOMED","bipolardisorder"),
  ("9340000","Bipolar I disorder, single manic episode (disorder)","SNOMED","bipolardisorder"),
  ("12969000","Severe bipolar II disorder, most recent episode major depressive, in full remission (disorder)","SNOMED","bipolardisorder"),
  ("19300006","Severe bipolar II disorder, most recent episode major depressive with psychotic features, mood-congruent (disorder)","SNOMED","bipolardisorder"),
  ("20960007","Severe bipolar II disorder, most recent episode major depressive with psychotic features, mood-incongruent (disorder)","SNOMED","bipolardisorder"),
  ("22407005","Bipolar II disorder, most recent episode major depressive with catatonic features (disorder)","SNOMED","bipolardisorder"),
  ("29929003","Bipolar I disorder, most recent episode depressed with atypical features (disorder)","SNOMED","bipolardisorder"),
  ("30520009","Severe bipolar II disorder, most recent episode major depressive with psychotic features (disorder)","SNOMED","bipolardisorder"),
  ("34315001","Bipolar II disorder, most recent episode major depressive with melancholic features (disorder)","SNOMED","bipolardisorder"),
  ("64731001","Severe mixed bipolar I disorder with psychotic features, mood-congruent (disorder)","SNOMED","bipolardisorder"),
  ("65042007","Bipolar I disorder, most recent episode mixed with postpartum onset (disorder)","SNOMED","bipolardisorder"),
  ("73471000","Bipolar I disorder, most recent episode mixed with catatonic features (disorder)","SNOMED","bipolardisorder"),
  ("162004","Severe manic bipolar I disorder without psychotic features (disorder)","SNOMED","bipolardisorder"),
  ("191623007","Bipolar affective disorder, currently manic, severe, with psychosis (disorder)","SNOMED","bipolardisorder"),
  ("28663008","Severe manic bipolar I disorder with psychotic features (disorder)","SNOMED","bipolardisorder"),
  ("35722002","Severe bipolar II disorder, most recent episode major depressive, in remission (disorder)","SNOMED","bipolardisorder"),
  ("43568002","Bipolar II disorder, most recent episode major depressive with atypical features (disorder)","SNOMED","bipolardisorder"),
  ("59617007","Severe depressed bipolar I disorder with psychotic features (disorder)","SNOMED","bipolardisorder"),
  ("61403008","Severe depressed bipolar I disorder without psychotic features (disorder)","SNOMED","bipolardisorder"),
  ("67002003","Severe bipolar II disorder, most recent episode major depressive, in partial remission (disorder)","SNOMED","bipolardisorder"),
  ("767635003","Bipolar I disorder, most recent episode manic (disorder)","SNOMED","bipolardisorder"),
   ("81319007","Severe bipolar II disorder, most recent episode major depressive without psychotic features (disorder)","SNOMED","bipolardisorder"),
  ("13581000","Severe bipolar I disorder, single manic episode with psychotic features, mood-congruent (disorder)","SNOMED","bipolardisorder"),
  ("17782008","Bipolar I disorder, most recent episode manic with catatonic features (disorder)","SNOMED","bipolardisorder"),
  ("21900002","Bipolar I disorder, most recent episode depressed with catatonic features (disorder)","SNOMED","bipolardisorder"),
  ("26203008","Severe depressed bipolar I disorder with psychotic features, mood-incongruent (disorder)","SNOMED","bipolardisorder"),
  ("33380008","Severe manic bipolar I disorder with psychotic features, mood-incongruent (disorder)","SNOMED","bipolardisorder"),
  ("54761006","Severe depressed bipolar I disorder with psychotic features, mood-congruent (disorder)","SNOMED","bipolardisorder"),
  ("765176007","Psychosis and severe depression co-occurrent and due to bipolar affective disorder (disorder)","SNOMED","bipolardisorder"),
  ("78269000","Bipolar I disorder, single manic episode, in partial remission (disorder)","SNOMED","bipolardisorder"),
  ("78640000","Severe manic bipolar I disorder with psychotic features, mood-congruent (disorder)","SNOMED","bipolardisorder"),
  ("87950005","Bipolar I disorder, single manic episode with catatonic features (disorder)","SNOMED","bipolardisorder"),
  ("1499003","Bipolar I disorder, single manic episode with postpartum onset (disorder)","SNOMED","bipolardisorder"),
  ("191590005","Recurrent manic episodes (disorder)","SNOMED","bipolardisorder"),
  ("191592002","Recurrent manic episodes, mild (disorder)","SNOMED","bipolardisorder"),
  ("191593007","Recurrent manic episodes, moderate (disorder)","SNOMED","bipolardisorder"),
  ("191597008","Recurrent manic episodes, in full remission (disorder)","SNOMED","bipolardisorder"),
  ("231444002","Organic bipolar disorder (disorder)","SNOMED","bipolardisorder"),
  ("38368003","Schizoaffective disorder, bipolar type (disorder)","SNOMED","bipolardisorder"),
  ("55516002","Bipolar I disorder, most recent episode manic with postpartum onset (disorder)","SNOMED","bipolardisorder"),
  ("86058007","Severe bipolar I disorder, single manic episode with psychotic features, mood-incongruent (disorder)","SNOMED","bipolardisorder"),
  ("87203005","Bipolar I disorder, most recent episode depressed with postpartum onset (disorder)","SNOMED","bipolardisorder"),
  ("191625000","Bipolar affective disorder, currently manic, in full remission (disorder)","SNOMED","bipolardisorder"),
  ("191627008","Bipolar affective disorder, current episode depression (disorder)","SNOMED","bipolardisorder"),
  ("191634005","Bipolar affective disorder, currently depressed, in full remission (disorder)","SNOMED","bipolardisorder"),
  ("30687003","Bipolar II disorder, most recent episode major depressive with postpartum onset (disorder)","SNOMED","bipolardisorder")
  AS {project_prefix}_bipolardisorder_codelist (code, term, system, codelist) """)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW bipolardisorder AS
# MAGIC SELECT NHS_NUMBER_DEID, "bipolardisorder" as predictor, date FROM global_temp.hes_icd 
# MAGIC WHERE CODE IN (SELECT code FROM global_temp.ccu037_02_bipolardisorder_codelist)
# MAGIC UNION ALL
# MAGIC SELECT NHS_NUMBER_DEID, "bipolardisorder" as predictor, date FROM global_temp.gdppr_snomed
# MAGIC WHERE CODE IN (SELECT code FROM global_temp.ccu037_02_bipolardisorder_codelist)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE GLOBAL TEMP VIEW bipolardisorder1 AS
# MAGIC SELECT DISTINCT NHS_NUMBER_DEID, predictor
# MAGIC FROM
# MAGIC (SELECT b.NHS_NUMBER_DEID, a.predictor, a.date, b.date_first_covid_event
# MAGIC FROM
# MAGIC (SELECT
# MAGIC   NHS_NUMBER_DEID,
# MAGIC   min(date) as date, predictor
# MAGIC FROM
# MAGIC   global_temp.bipolardisorder
# MAGIC GROUP BY
# MAGIC   NHS_NUMBER_DEID, predictor) a
# MAGIC LEFT JOIN 
# MAGIC   dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria b
# MAGIC ON
# MAGIC   a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID)
# MAGIC WHERE date < date_first_covid_event

# COMMAND ----------

# spark.sql(f"""
# CREATE OR REPLACE GLOBAL TEMP VIEW bipolardisorder1 AS
#     SELECT NHS_NUMBER_DEID, predictor FROM
#     (SELECT a.NHS_NUMBER_DEID, a.date, a.predictor, b.date_first_covid_event
#     FROM global_temp.bipolardisorder AS a
#     LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
#     ON a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID) 
#     WHERE date < date_first_covid_event  """)

# COMMAND ----------

## Moderate/severe depression:
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW {project_prefix}_depression_codelist AS SELECT * FROM VALUES
  ("F32.1","Moderate depressive episode","ICD10","depression"),
  ("F32.2","Severe depressive episode without psychotic symptoms","ICD10","depression"),
  ("F32.3","Severe depressive episode with psychotic symptoms","ICD10","depression"),      
  ("F33.1","Major depressive disorder, recurrent, moderate","ICD10","depression"),       
  ("F33.2","Recurrent depressive disorder, current episode severe without psychotic symptoms","ICD10","depression"),      
  ("F33.3","Recurrent depressive disorder, current episode severe with psychotic symptoms","ICD10","depression"),      

('15639000','Moderate major depression, single episode (disorder)','SNOMED','depression'),
('310496002','Moderate depression (disorder)','SNOMED','depression'),
('832007','Moderate major depression (disorder)','SNOMED','depression'),
('191611001','Recurrent major depressive episodes, moderate (disorder)','SNOMED','depression'),
('18818009','Moderate recurrent major depression (disorder)','SNOMED','depression'),
('16264901000119109','Recurrent moderate major depressive disorder co-occurrent with anxiety (disorder)','SNOMED','depression'),
('16266831000119100','Moderate major depressive disorder co-occurrent with anxiety single episode (disorder)','SNOMED','depression'),
('66344007','Recurrent major depression (disorder)','SNOMED','depression'),
('300706003','Endogenous depression (disorder)','SNOMED','depression'),
('274948002','Endogenous depression - recurrent (disorder)','SNOMED','depression'),
('68019004','Recurrent major depression in remission (disorder)','SNOMED','depression'),
('46244001','Recurrent major depression in full remission (disorder)','SNOMED','depression'),
('40379007','Mild recurrent major depression (disorder)','SNOMED','depression'),
('33135002','Recurrent major depression in partial remission (disorder)','SNOMED','depression'),
('281000119103','Severe recurrent major depression (disorder)','SNOMED','depression'),
('720455008','Minimal major depression (disorder)','SNOMED','depression'),
('719593009','Moderately severe depression (disorder)','SNOMED','depression'),
('450714000','Severe major depression (disorder)','SNOMED','depression'),
('310497006','Severe depression (disorder)','SNOMED','depression'),
('720452006','Moderately severe recurrent major depression (disorder)','SNOMED','depression'),
('83458005','Agitated depression (disorder)','SNOMED','depression'),
('87512008','Mild major depression (disorder)','SNOMED','depression'),
('42810003','Major depression in remission (disorder)','SNOMED','depression'),
('320751009','Major depression, melancholic type (disorder)','SNOMED','depression'),
('719592004','Moderately severe major depression (disorder)','SNOMED','depression'),
('36474008','Severe recurrent major depression without psychotic features (disorder)','SNOMED','depression'),
('33078009','Severe recurrent major depression with psychotic features, mood-congruent (disorder)','SNOMED','depression'),
('28475009','Severe recurrent major depression with psychotic features (disorder)','SNOMED','depression'),
('15193003','Severe recurrent major depression with psychotic features, mood-incongruent (disorder)','SNOMED','depression'),
('251000119105','Severe major depression, single episode (disorder)','SNOMED','depression'),
('30605009','Major depression in partial remission (disorder)','SNOMED','depression'),
('75084000','Severe major depression without psychotic features (disorder)','SNOMED','depression'),
('73867007','Severe major depression with psychotic features (disorder)','SNOMED','depression'),
('726772006','Major depression with psychotic features (disorder)','SNOMED','depression'),
('720454007','Minimal major depression single episode (disorder)','SNOMED','depression'),
('720453001','Moderately severe major depression single episode (disorder)','SNOMED','depression'),
('63412003','Major depression in full remission (disorder)','SNOMED','depression'),
('76441001','Severe major depression, single episode, without psychotic features (disorder)','SNOMED','depression'),
('70747007','Major depression single episode, in partial remission (disorder)','SNOMED','depression'),
('60099002','Severe major depression with psychotic features, mood-incongruent (disorder)','SNOMED','depression'),
('430852001','Severe major depression, single episode, with psychotic features (disorder)','SNOMED','depression'),
('19527009','Single episode of major depression in full remission (disorder)','SNOMED','depression'),
('20250007','Severe major depression, single episode, with psychotic features, mood-incongruent (disorder)','SNOMED','depression'),
('77911002','Severe major depression, single episode, with psychotic features, mood-congruent (disorder)','SNOMED','depression'),
('310497006','Severe depression (disorder)','SNOMED','depression'),
('33736005','Severe major depression with psychotic features, mood-congruent (disorder)','SNOMED','depression'),
('231499006','Endogenous depression first episode (disorder)','SNOMED','depression')
AS {project_prefix}_depression_codelist (code, term, system, codelist) """)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW depression AS
# MAGIC SELECT NHS_NUMBER_DEID, "depression" as predictor, date FROM global_temp.hes_icd 
# MAGIC WHERE CODE IN (SELECT code FROM global_temp.ccu037_02_depression_codelist)
# MAGIC UNION ALL
# MAGIC SELECT NHS_NUMBER_DEID, "depression" as predictor, date FROM global_temp.gdppr_snomed
# MAGIC WHERE CODE IN (SELECT code FROM global_temp.ccu037_02_depression_codelist)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE GLOBAL TEMP VIEW depression1 AS
# MAGIC SELECT DISTINCT NHS_NUMBER_DEID, predictor
# MAGIC FROM
# MAGIC (SELECT b.NHS_NUMBER_DEID, a.predictor, a.date, b.date_first_covid_event
# MAGIC FROM
# MAGIC (SELECT
# MAGIC   NHS_NUMBER_DEID,
# MAGIC   min(date) as date, predictor
# MAGIC FROM
# MAGIC   global_temp.depression
# MAGIC GROUP BY
# MAGIC   NHS_NUMBER_DEID, predictor) a
# MAGIC LEFT JOIN 
# MAGIC   dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria b
# MAGIC ON
# MAGIC   a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID)
# MAGIC WHERE date < date_first_covid_event

# COMMAND ----------

# spark.sql(f"""
# CREATE OR REPLACE GLOBAL TEMP VIEW depression1 AS
#     SELECT NHS_NUMBER_DEID, predictor FROM
#     (SELECT a.NHS_NUMBER_DEID, a.date, a.predictor, b.date_first_covid_event
#     FROM global_temp.depression AS a
#     LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
#     ON a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID) 
#     WHERE date < date_first_covid_event  """)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Antipsychotic use
# MAGIC ### 1 year prior

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW antipsycoticBNF AS
# MAGIC SELECT * FROM VALUES 
# MAGIC ('Amisulpride','0402010A0'),
# MAGIC ('Aripiprazole','0402010AD'),
# MAGIC ('Clozapine','0402010C0'),
# MAGIC ('Lurasidone','0402010AI'),
# MAGIC ('Olanzapine','040201060'),
# MAGIC ('Paliperidone','0402010AE'),
# MAGIC ('Quetiapine','0402010AB'),
# MAGIC ('Risperidone','040201030'),
# MAGIC ('Zotepine','0402010AC')
# MAGIC AS tab(antipsycotic,BNFcode);

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW antipsychotic AS
# MAGIC SELECT NHS_NUMBER_DEID, "antipsychotic" as predictor, date FROM global_temp.gdppr_meds WHERE left(code, 9) IN (SELECT BNFcode FROM global_temp.antipsycoticBNF)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE GLOBAL TEMP VIEW antipsychotic1 AS
# MAGIC SELECT DISTINCT NHS_NUMBER_DEID, predictor
# MAGIC FROM
# MAGIC (SELECT b.NHS_NUMBER_DEID, a.predictor, a.date, b.date_first_covid_event
# MAGIC FROM
# MAGIC (SELECT
# MAGIC   NHS_NUMBER_DEID,
# MAGIC   min(date) as date, predictor
# MAGIC FROM
# MAGIC   global_temp.antipsychotic
# MAGIC GROUP BY
# MAGIC   NHS_NUMBER_DEID, predictor) a
# MAGIC LEFT JOIN 
# MAGIC   dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria b
# MAGIC ON
# MAGIC   a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID)
# MAGIC WHERE date < date_first_covid_event AND date > (date_first_covid_event - INTERVAL 365 DAY)

# COMMAND ----------

# spark.sql(f"""
# CREATE OR REPLACE GLOBAL TEMP VIEW antipsychotic1 AS
#     SELECT NHS_NUMBER_DEID, predictor FROM
#     (SELECT a.NHS_NUMBER_DEID, a.date, a.predictor, b.date_first_covid_event
#     FROM global_temp.antipsychotic AS a
#     LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
#     ON a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID) 
#     WHERE date < date_first_covid_event AND date > (date_first_covid_event - INTERVAL 365 DAY) """)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3.Migraines
# MAGIC ### Anytime before

# COMMAND ----------

## Migraine:
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW {project_prefix}_migraine_codelist AS SELECT * FROM VALUES
("G43","Migraine","ICD10","migraine"),

('145611000119107','Allergic migraine (disorder)','SNOMED','migraine'),
('193030005','Migraine variants (disorder)','SNOMED','migraine'),
('193039006','Complicated migraine (disorder)','SNOMED','migraine'),
('26150009','Lower half migraine (disorder)','SNOMED','migraine'),
('37796009','Migraine (disorder)','SNOMED','migraine'),
('423894005','Refractory migraine (disorder)','SNOMED','migraine'),
('425365009','Refractory migraine variants (disorder)','SNOMED','migraine'),
('427419006','Transformed migraine (disorder)','SNOMED','migraine'),
('49605003','Ophthalmoplegic migraine (disorder)','SNOMED','migraine'),
('75879005','Abdominal migraine (disorder)','SNOMED','migraine'),
('424699007','Migraine variants, not intractable (disorder)','SNOMED','migraine'),
('425936006','Intractable ophthalmic migraine (disorder)','SNOMED','migraine'),
('4473006','Migraine with aura (disorder)','SNOMED','migraine'),
('56097005','Migraine without aura (disorder)','SNOMED','migraine'),
('59292006','Hemiplegic migraine (disorder)','SNOMED','migraine'),
('79267007','Retinal migraine (disorder)','SNOMED','migraine'),
('83351003','Basilar migraine (disorder)','SNOMED','migraine'),
('95653008','Acute confusional migraine (disorder)','SNOMED','migraine'),
('95655001','Ophthalmic migraine (disorder)','SNOMED','migraine'),
('95656000','Familial hemiplegic migraine (disorder)','SNOMED','migraine'),
('124171000119105','Chronic intractable migraine without aura (disorder)','SNOMED','migraine'),
('230462002','Migraine with typical aura (disorder)','SNOMED','migraine'),
('230464001','Non-familial hemiplegic migraine (disorder)','SNOMED','migraine'),
('230465000','Migraine aura without headache (disorder)','SNOMED','migraine'),
('423279000','Refractory migraine without aura (disorder)','SNOMED','migraine'),
('423683008','Refractory migraine with aura (disorder)','SNOMED','migraine'),
('425007008','Migraine without aura, not refractory (disorder)','SNOMED','migraine'),
('445322004','Migraine variant with headache (disorder)','SNOMED','migraine'),
('699314009','Migraine with persistent visual aura (disorder)','SNOMED','migraine'),
('711545001','Migraine caused by estrogen contraceptive (disorder)','SNOMED','migraine'),
('124001000119104','Status migrainosus co-occurrent and due to migraine without aura (disorder)','SNOMED','migraine'),
('161481007','History of migraine (situation)','SNOMED','migraine'),
('38823002','Aural headache (finding)','SNOMED','migraine') 
AS {project_prefix}_migraine_codelist (code, term, system, codelist) """)


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM 
# MAGIC global_temp.ccu037_02_migraine_codelist

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW migraine AS
# MAGIC SELECT NHS_NUMBER_DEID, "migraine" as predictor, date FROM global_temp.hes_icd 
# MAGIC WHERE CODE IN (SELECT code FROM global_temp.ccu037_02_migraine_codelist)
# MAGIC UNION ALL
# MAGIC SELECT NHS_NUMBER_DEID, "migraine" as predictor, date FROM global_temp.gdppr_snomed
# MAGIC WHERE CODE IN (SELECT code FROM global_temp.ccu037_02_migraine_codelist)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM global_temp.migraine

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE GLOBAL TEMP VIEW migraine1 AS
# MAGIC SELECT DISTINCT NHS_NUMBER_DEID, predictor
# MAGIC FROM
# MAGIC (SELECT b.NHS_NUMBER_DEID, a.predictor, a.date, b.date_first_covid_event
# MAGIC FROM
# MAGIC (SELECT
# MAGIC   NHS_NUMBER_DEID,
# MAGIC   min(date) as date, predictor
# MAGIC FROM
# MAGIC   global_temp.migraine
# MAGIC GROUP BY
# MAGIC   NHS_NUMBER_DEID, predictor) a
# MAGIC LEFT JOIN 
# MAGIC   dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria b
# MAGIC ON
# MAGIC   a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID)
# MAGIC WHERE date < date_first_covid_event 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM global_temp.migraine1

# COMMAND ----------

# spark.sql(f"""
# CREATE OR REPLACE GLOBAL TEMP VIEW migraine1 AS
#     SELECT NHS_NUMBER_DEID, predictor FROM
#     (SELECT a.NHS_NUMBER_DEID, a.date, a.predictor, b.date_first_covid_event
#     FROM global_temp.depression AS a
#     LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
#     ON a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID) 
#     WHERE date < date_first_covid_event  """)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Erectile dysfunction
# MAGIC ### 1 year prior

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW erectiledysfunctionBNF AS
# MAGIC SELECT * FROM VALUES 
# MAGIC ('Alprostadil','0704050B0'),
# MAGIC ('Avanafil','0704050AC'),
# MAGIC ('Sildenafil (Erectile Dysfunction)','0704050Z0'),
# MAGIC ('Tadalafil','0704050R0'),
# MAGIC ('Vardenafil','0704050AA'),
# MAGIC ('Papaverine hydrochloride','0704050J0'),
# MAGIC ('Phentolamine/aviptadil','0704050AD')
# MAGIC AS tab(erectiledysfunction,BNFcode);

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW erectiledysfunction AS
# MAGIC SELECT NHS_NUMBER_DEID, "erectiledysfunction" as predictor, date FROM global_temp.gdppr_meds WHERE left(code, 9) IN (SELECT BNFcode FROM global_temp.erectiledysfunctionBNF)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE GLOBAL TEMP VIEW erectiledysfunction1 AS
# MAGIC SELECT DISTINCT NHS_NUMBER_DEID, predictor
# MAGIC FROM
# MAGIC (SELECT b.NHS_NUMBER_DEID, a.predictor, a.date, b.date_first_covid_event
# MAGIC FROM
# MAGIC (SELECT
# MAGIC   NHS_NUMBER_DEID,
# MAGIC   min(date) as date, predictor
# MAGIC FROM
# MAGIC   global_temp.erectiledysfunction
# MAGIC GROUP BY
# MAGIC   NHS_NUMBER_DEID, predictor) a
# MAGIC LEFT JOIN 
# MAGIC   dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria b
# MAGIC ON
# MAGIC   a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID)
# MAGIC WHERE date < date_first_covid_event AND date > (date_first_covid_event - INTERVAL 365 DAY)

# COMMAND ----------

# spark.sql(f"""
# CREATE OR REPLACE GLOBAL TEMP VIEW erectiledysfunction1 AS
#     SELECT NHS_NUMBER_DEID, predictor FROM
#     (SELECT a.NHS_NUMBER_DEID, a.date, a.predictor, b.date_first_covid_event
#     FROM global_temp.erectiledysfunction AS a
#     LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
#     ON a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID) 
#     WHERE date < date_first_covid_event AND date > (date_first_covid_event - INTERVAL 365 DAY) """)

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Steroid tablets
# MAGIC ### 1 year prior

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW steroidBNF AS
# MAGIC SELECT * FROM VALUES 
# MAGIC ('Prednisolone',' 0603020T0'),
# MAGIC ('Betamethasone',' 0603020B0'),
# MAGIC ('Betamethasone sodium phosphate',' 0603020C0'),
# MAGIC ('Cortisone acetate',' 0603020F0'),
# MAGIC ('Dexamethasone',' 0603020G0'),
# MAGIC ('Dexamethasone phosphate',' 0603020AA'),
# MAGIC ('Dexamethasone sodium phosphate',' 0603020H0'),
# MAGIC ('Deflazacort',' 0603020I0'),
# MAGIC ('Hydrocortisone',' 0603020J0'),
# MAGIC ('Hydrocortisone sodium phosphate',' 0603020L0'),
# MAGIC ('Hydrocortisone sodium succinate',' 0603020M0'),
# MAGIC ('Methylprednisolone',' 0603020S0'),
# MAGIC ('Methylprednisolone aceponate',' 0603020AC'),
# MAGIC ('Methylprednisolone sodium succinate',' 0603020K0'),
# MAGIC ('Triamcinolone acetonide',' 0603020Z0'),
# MAGIC ('Prednisone','0603020X0')  -- suposedly not included in the list of QRISK3 Corticosteroid list (bmj article)
# MAGIC AS tab(steroid,BNFcode);

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW steroid AS
# MAGIC SELECT NHS_NUMBER_DEID, "steroid" as predictor, date FROM global_temp.gdppr_meds WHERE left(code, 9) IN (SELECT BNFcode FROM global_temp.steroidBNF)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE GLOBAL TEMP VIEW steroid1 AS
# MAGIC SELECT DISTINCT NHS_NUMBER_DEID, predictor
# MAGIC FROM
# MAGIC (SELECT b.NHS_NUMBER_DEID, a.predictor, a.date, b.date_first_covid_event
# MAGIC FROM
# MAGIC (SELECT
# MAGIC   NHS_NUMBER_DEID,
# MAGIC   min(date) as date, predictor
# MAGIC FROM
# MAGIC   global_temp.steroid
# MAGIC GROUP BY
# MAGIC   NHS_NUMBER_DEID, predictor) a
# MAGIC LEFT JOIN 
# MAGIC   dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria b
# MAGIC ON
# MAGIC   a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID)
# MAGIC WHERE date < date_first_covid_event 
# MAGIC --AND date > (date_first_covid_event - INTERVAL 365 DAY)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM global_temp.steroid1

# COMMAND ----------

# spark.sql(f"""
# CREATE OR REPLACE GLOBAL TEMP VIEW steroid1 AS
#     SELECT NHS_NUMBER_DEID, predictor FROM
#     (SELECT a.NHS_NUMBER_DEID, a.date, a.predictor, b.date_first_covid_event
#     FROM global_temp.steroid AS a
#     LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
#     ON a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID) 
#     WHERE date < date_first_covid_event AND date > (date_first_covid_event - INTERVAL 365 DAY) """)

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. CKD
# MAGIC ### Anytime prior

# COMMAND ----------

#CKD - Chronic kidney disease (aka #hasbled - renal disease)
spark.sql(f"""
  CREATE OR REPLACE GLOBAL TEMP VIEW {project_prefix}_CKD_codelist AS
  SELECT DISTINCT code as code, term as term, terminology as system, "CKD" as codelist
  FROM bhf_cvd_covid_uk_byod.bhf_covid_uk_phenotypes_20210127 
  WHERE name == 'CKD'
  """)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW CKD AS
# MAGIC SELECT NHS_NUMBER_DEID, "CKD" as predictor, date FROM global_temp.hes_icd 
# MAGIC WHERE CODE IN (SELECT code FROM global_temp.ccu037_02_CKD_codelist)
# MAGIC UNION ALL
# MAGIC SELECT NHS_NUMBER_DEID, "CKD" as predictor, date FROM global_temp.gdppr_snomed
# MAGIC WHERE CODE IN (SELECT code FROM global_temp.ccu037_02_CKD_codelist)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE GLOBAL TEMP VIEW CKD1 AS
# MAGIC SELECT DISTINCT NHS_NUMBER_DEID, predictor
# MAGIC FROM
# MAGIC (SELECT b.NHS_NUMBER_DEID, a.predictor, a.date, b.date_first_covid_event
# MAGIC FROM
# MAGIC (SELECT
# MAGIC   NHS_NUMBER_DEID,
# MAGIC   min(date) as date, predictor
# MAGIC FROM
# MAGIC   global_temp.CKD
# MAGIC GROUP BY
# MAGIC   NHS_NUMBER_DEID, predictor) a
# MAGIC LEFT JOIN 
# MAGIC   dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria b
# MAGIC ON
# MAGIC   a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID)
# MAGIC WHERE date < date_first_covid_event 

# COMMAND ----------

# spark.sql(f"""
# CREATE OR REPLACE GLOBAL TEMP VIEW CKD1 AS
#     SELECT NHS_NUMBER_DEID, predictor FROM
#     (SELECT a.NHS_NUMBER_DEID, a.date, a.predictor, b.date_first_covid_event
#     FROM global_temp.CKD AS a
#     LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
#     ON a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID) 
#     WHERE date < date_first_covid_event  """)

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. AF
# MAGIC ### Anytime prior

# COMMAND ----------

#AF - Arterial Fibrilation
spark.sql(f"""
  CREATE OR REPLACE GLOBAL TEMP VIEW {project_prefix}_AF_codelist AS
  SELECT DISTINCT code as code, term as term, terminology as system, "AF" as codelist
  FROM bhf_cvd_covid_uk_byod.bhf_covid_uk_phenotypes_20210127 
  WHERE name == 'AF'
  """)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW AF AS
# MAGIC SELECT NHS_NUMBER_DEID, "AF" as predictor, date FROM global_temp.hes_icd 
# MAGIC WHERE CODE IN (SELECT code FROM global_temp.ccu037_02_AF_codelist)
# MAGIC UNION ALL
# MAGIC SELECT NHS_NUMBER_DEID, "AF" as predictor, date FROM global_temp.gdppr_snomed
# MAGIC WHERE CODE IN (SELECT code FROM global_temp.ccu037_02_AF_codelist)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE GLOBAL TEMP VIEW AF1 AS
# MAGIC SELECT DISTINCT NHS_NUMBER_DEID, predictor
# MAGIC FROM
# MAGIC (SELECT b.NHS_NUMBER_DEID, a.predictor, a.date, b.date_first_covid_event
# MAGIC FROM
# MAGIC (SELECT
# MAGIC   NHS_NUMBER_DEID,
# MAGIC   min(date) as date, predictor
# MAGIC FROM
# MAGIC   global_temp.AF
# MAGIC GROUP BY
# MAGIC   NHS_NUMBER_DEID, predictor) a
# MAGIC LEFT JOIN 
# MAGIC   dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria b
# MAGIC ON
# MAGIC   a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID)
# MAGIC WHERE date < date_first_covid_event 

# COMMAND ----------

# spark.sql(f"""
# CREATE OR REPLACE GLOBAL TEMP VIEW AF1 AS
#     SELECT NHS_NUMBER_DEID, predictor FROM
#     (SELECT a.NHS_NUMBER_DEID, a.date, a.predictor, b.date_first_covid_event
#     FROM global_temp.AF AS a
#     LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
#     ON a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID) 
#     WHERE date < date_first_covid_event  """)

# COMMAND ----------

# MAGIC %md
# MAGIC # 8. RA
# MAGIC ### Anytime prior

# COMMAND ----------

## RA:
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW {project_prefix}_RA_codelist AS SELECT * FROM VALUES
("M05","Diagnosis of rheumatoid arthritis","ICD10","RA"),
("M06.2","Rheumatoid bursitis","ICD10","RA"),
("M06.8","Other specified rheumatoid arthritis","ICD10","RA"),
("M06.9","Rheumatoid arthritis, unspecified","ICD10","RA"),
("M05.0","Felty syndrome","ICD10","RA"),
("J99.0","Caplan syndrome","ICD10","RA"),
("M05.1","Caplan syndrome","ICD10","RA"),
("M06.1","Adult-onset Still disease ","ICD10","RA"),
("M06.4","inflammatory polyarthropathy not otherwise specified","ICD10","RA"),
("69896004","Rheumatoid arthritis (disorder)","SNOMED","RA"),
('201791009','Flare of rheumatoid arthritis (disorder)','SNOMED','SLE'),
('143441000119108','Rheumatoid arthritis in remission (disorder)','SNOMED','SLE'),
('201768005','Rheumatoid arthritis of acromioclavicular joint (disorder)','SNOMED','SLE'),
('287008006','Rheumatoid arthritis of ankle and/or foot (disorder)','SNOMED','SLE'),
('201779000','Rheumatoid arthritis of ankle (disorder)','SNOMED','SLE'),
('201769002','Rheumatoid arthritis of elbow (disorder)','SNOMED','SLE'),
('287007001','Rheumatoid arthritis of hand joint (disorder)','SNOMED','SLE'),
('201774005','Rheumatoid arthritis of distal interphalangeal joint of finger (disorder)','SNOMED','SLE'),
('1073711000119105','Rheumatoid arthritis of left hand (disorder)','SNOMED','SLE'),
('201772009','Rheumatoid arthritis of metacarpophalangeal joint (disorder)','SNOMED','SLE'),
('201773004','Rheumatoid arthritis of proximal interphalangeal joint of finger (disorder)','SNOMED','SLE'),
('1073791000119101','Rheumatoid arthritis of right hand (disorder)','SNOMED','SLE'),
('15687321000119109','Rheumatoid arthritis of bilateral hands (disorder)','SNOMED','SLE'),
('201775006','Rheumatoid arthritis of hip (disorder)','SNOMED','SLE'),
('1073721000119103','Rheumatoid arthritis of left hip (disorder)','SNOMED','SLE'),
('1073801000119100','Rheumatoid arthritis of right hip (disorder)','SNOMED','SLE'),
('781206002','Rheumatoid arthritis of joint of spine (disorder)','SNOMED','SLE'),
('201764007','Rheumatoid arthritis of cervical spine (disorder)','SNOMED','SLE'),
('201777003','Rheumatoid arthritis of knee (disorder)','SNOMED','SLE'),
('1073731000119100','Rheumatoid arthritis of left knee (disorder)','SNOMED','SLE'),
('15687201000119107','Rheumatoid arthritis of bilateral knees (disorder)','SNOMED','SLE'),
('1073811000119102','Rheumatoid arthritis of right knee (disorder)','SNOMED','SLE'),
('287006005','Rheumatoid arthritis of multiple joints (disorder)','SNOMED','SLE'),
('201776007','Rheumatoid arthritis of sacroiliac joint (disorder)','SNOMED','SLE'),
('201766009','Rheumatoid arthritis of shoulder (disorder)','SNOMED','SLE'),
('1073741000119109','Rheumatoid arthritis of left shoulder (disorder)','SNOMED','SLE'),
('1073821000119109','Rheumatoid arthritis of right shoulder (disorder)','SNOMED','SLE'),
('201767000','Rheumatoid arthritis of sternoclavicular joint (disorder)','SNOMED','SLE'),
('427770001','Rheumatoid arthritis of temporomandibular joint (disorder)','SNOMED','SLE'),
('201778008','Rheumatoid arthritis of tibiofibular joint (disorder)','SNOMED','SLE'),
('201771002','Rheumatoid arthritis of wrist (disorder)','SNOMED','SLE'),
('201770001','Rheumatoid arthritis of distal radioulnar joint (disorder)','SNOMED','SLE'),
('1073751000119106','Rheumatoid arthritis of left wrist (disorder)','SNOMED','RA'),
('1073831000119107','Rheumatoid arthritis of right wrist (disorder)','SNOMED','RA'),
('28880005','Rheumatoid arthritis with carditis (disorder)','SNOMED','RA'),
('735599007','Rheumatoid arthritis with erosion of joint (disorder)','SNOMED','RA'),
('402432002','Rheumatoid arthritis with neutrophilic dermatitis (disorder)','SNOMED','RA'),
('129563009','Rheumatoid arthritis with osteoperiostitis (disorder)','SNOMED','RA'),
('7607008','Rheumatoid arthritis with pericarditis (disorder)','SNOMED','RA'),
('398640008','Rheumatoid arthritis with pneumoconiosis (disorder)','SNOMED','RA'),
('1162677006','Rheumatoid arthritis with rheumatoid lung disease (disorder)','SNOMED','RA'),
('59165007','Rheumatoid arthritis with scleritis (disorder)','SNOMED','RA'),
('400054000','Rheumatoid arthritis with vasculitis (disorder)','SNOMED','RA'),
('239792003','Seronegative rheumatoid arthritis (disorder)','SNOMED','RA'),
('239791005','Seropositive rheumatoid arthritis (disorder)','SNOMED','RA'),
('86219005','Uveitis-rheumatoid arthritis syndrome (disorder)','SNOMED','RA')    
AS {project_prefix}_RA_codelist (code, term, system, codelist) """)


# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW RA AS
# MAGIC SELECT NHS_NUMBER_DEID, "RA" as predictor, date FROM global_temp.hes_icd 
# MAGIC WHERE CODE IN (SELECT code FROM global_temp.ccu037_02_RA_codelist)
# MAGIC UNION ALL
# MAGIC SELECT NHS_NUMBER_DEID, "RA" as predictor, date FROM global_temp.gdppr_snomed
# MAGIC WHERE CODE IN (SELECT code FROM global_temp.ccu037_02_RA_codelist)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE GLOBAL TEMP VIEW RA1 AS
# MAGIC SELECT DISTINCT NHS_NUMBER_DEID, predictor
# MAGIC FROM
# MAGIC (SELECT b.NHS_NUMBER_DEID, a.predictor, a.date, b.date_first_covid_event
# MAGIC FROM
# MAGIC (SELECT
# MAGIC   NHS_NUMBER_DEID,
# MAGIC   min(date) as date, predictor
# MAGIC FROM
# MAGIC   global_temp.RA
# MAGIC GROUP BY
# MAGIC   NHS_NUMBER_DEID, predictor) a
# MAGIC LEFT JOIN 
# MAGIC   dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria b
# MAGIC ON
# MAGIC   a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID)
# MAGIC WHERE date < date_first_covid_event 

# COMMAND ----------

# spark.sql(f"""
# CREATE OR REPLACE GLOBAL TEMP VIEW RA1 AS
#     SELECT NHS_NUMBER_DEID, predictor FROM
#     (SELECT a.NHS_NUMBER_DEID, a.date, a.predictor, b.date_first_covid_event
#     FROM global_temp.RA AS a
#     LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
#     ON a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID) 
#     WHERE date < date_first_covid_event  """)

# COMMAND ----------

# MAGIC %md
# MAGIC # 9. SLE
# MAGIC ### Anytime prior

# COMMAND ----------

## SLE:
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW {project_prefix}_SLE_codelist AS SELECT * FROM VALUES
("M32","Systemic lupus erythematosus (SLE)","ICD10","SLE"),    
("M32.11","Libman-Sacks disease","ICD10","SLE"),

('1144942002','Neuropsychiatric disorder due to systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('201436003','Drug-induced systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('230307005','Chorea in systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('239889005','Bullous systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('403486000','Acute systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('403487009','Fulminating systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('403488004','Systemic lupus erythematosus of childhood (disorder)','SNOMED','SLE'),
('55464009','Systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('698694005','Systemic lupus erythematosus in remission (disorder)','SNOMED','SLE'),
('773333003','Autosomal systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('196138005','Lung disease with systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('239886003','Limited lupus erythematosus (disorder)','SNOMED','SLE'),
('403502003','Disseminated discoid lupus erythematosus (disorder)','SNOMED','SLE'),
('403510002','Lupus erythematosus-associated urticarial vasculitis (disorder)','SNOMED','SLE'),
('403511003','Lupus erythematosus-associated necrotizing vasculitis (disorder)','SNOMED','SLE'),
('68815009','Systemic lupus erythematosus glomerulonephritis syndrome (disorder)','SNOMED','SLE'),
('72181000119109','Endocarditis due to systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('724767000','Chorea co-occurrent and due to systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('95332009','Rash of systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('95644001','Systemic lupus erythematosus encephalitis (disorder)','SNOMED','SLE'),
('1144973003','Disorder of kidney due to systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('295101000119105','Nephropathy co-occurrent and due to systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('295121000119101','Nephrosis co-occurrent and due to systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('308751000119106','Glomerular disease due to systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('309762007','Systemic lupus erythematosus with pericarditis (disorder)','SNOMED','SLE'),
('1144921004','Disorder of heart due to systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('1144923001','Disorder of immune function due to systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('1144927000','Disorder of gastrointestinal tract due to systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('1144970000','Disorder of joint due to systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('732960002','Secondary autoimmune hemolytic anemia co-occurrent and due to systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('193178008','Polyneuropathy in disseminated lupus erythematosus (disorder)','SNOMED','SLE'),
('239944008','Lupus vasculitis (disorder)','SNOMED','SLE'),
('25380002','Pericarditis co-occurrent and due to systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('307755009','Renal tubulo-interstitial disorder in systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('36471000','Dilated cardiomyopathy due to systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('417303004','Retinal vasculitis due to systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('11013005','Systemic lupus erythematosus glomerulonephritis syndrome, World Health Organization class VI (disorder)','SNOMED','SLE'),
('295111000119108','Nephrotic syndrome co-occurrent and due to systemic lupus erythematosus (disorder)','SNOMED','SLE'),
('36402006','Systemic lupus erythematosus glomerulonephritis syndrome, World Health Organization class IV (disorder)','SNOMED','SLE'),
('4676006','Systemic lupus erythematosus glomerulonephritis syndrome, World Health Organization class II (disorder)','SNOMED','SLE'),
('52042003','Systemic lupus erythematosus glomerulonephritis syndrome, World Health Organization class V (disorder)','SNOMED','SLE'),
('73286009','Systemic lupus erythematosus glomerulonephritis syndrome, World Health Organization class I (disorder)','SNOMED','SLE'),
('193248005','Myopathy due to disseminated lupus erythematosus (disorder)','SNOMED','SLE'),
('76521009','Systemic lupus erythematosus glomerulonephritis syndrome, World Health Organization class III (disorder)','SNOMED','SLE'),
('54912002','Drug-induced lupus erythematosus caused by hydralazine (disorder)','SNOMED','SLE'),
('233730002','Lupus pneumonia (disorder)','SNOMED','SLE'),
('77753005','Lupus disease of the lung (disorder)','SNOMED','SLE'),
('239887007','Systemic lupus erythematosus with organ/system involvement (disorder)','SNOMED','SLE'),
('239890001','Systemic lupus erythematosus with multisystem involvement (disorder)','SNOMED','SLE'),
('397856003','Systemic lupus erythematosus-related syndrome (disorder)','SNOMED','SLE'),
('402865003','Systemic lupus erythematosus-associated antiphospholipid syndrome (disorder)','SNOMED','SLE')
    
AS {project_prefix}_sle_codelist (code, term, system, codelist) """)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW SLE AS
# MAGIC SELECT NHS_NUMBER_DEID, "SLE" as predictor, date FROM global_temp.hes_icd 
# MAGIC WHERE CODE IN (SELECT code FROM global_temp.ccu037_02_SLE_codelist)
# MAGIC UNION ALL
# MAGIC SELECT NHS_NUMBER_DEID, "SLE" as predictor, date FROM global_temp.gdppr_snomed
# MAGIC WHERE CODE IN (SELECT code FROM global_temp.ccu037_02_SLE_codelist)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_02_SLE AS
# MAGIC SELECT DISTINCT NHS_NUMBER_DEID, predictor
# MAGIC FROM
# MAGIC (SELECT b.NHS_NUMBER_DEID, a.predictor, a.date, b.date_first_covid_event
# MAGIC FROM
# MAGIC (SELECT
# MAGIC   NHS_NUMBER_DEID,
# MAGIC   min(date) as date, predictor
# MAGIC FROM
# MAGIC   global_temp.SLE
# MAGIC GROUP BY
# MAGIC   NHS_NUMBER_DEID, predictor) a
# MAGIC LEFT JOIN 
# MAGIC   dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria b
# MAGIC ON
# MAGIC   a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID)
# MAGIC WHERE date < date_first_covid_event 

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from global_temp.ccu037_02_sle

# COMMAND ----------

drop_table("ccu037_02_sle")
create_table("ccu037_02_sle")

# COMMAND ----------

# spark.sql(f"""
# CREATE OR REPLACE GLOBAL TEMP VIEW SLE1 AS
#     SELECT NHS_NUMBER_DEID, predictor FROM
#     (SELECT a.NHS_NUMBER_DEID, a.date, a.predictor, b.date_first_covid_event
#     FROM global_temp.SLE AS a
#     LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
#     ON a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID) 
#     WHERE date < date_first_covid_event  """)

# COMMAND ----------

# MAGIC %md #save SLE using another fx (does not work either)

# COMMAND ----------

# MAGIC %run /Workspaces/dars_nic_391419_j3w9t_collab/CCU037_01/Create_skinny_table_using_013_code/CCU037_00_helper_functions

# COMMAND ----------

#load tables for export into pyspark
output_table_name = "global_temp." + "ccu037_02_sle"
outable_table_df = spark.table(output_table_name)

# COMMAND ----------

#create and export tables
export_table_name = "ccu037_02_sle_test"
create_table_pyspark(outable_table_df, export_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC # 10. Smoking
# MAGIC ### Anytime prior

# COMMAND ----------

#smoking (ever smokers)

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW {project_prefix}_smoking_codelist AS SELECT * FROM VALUES
("160603005","Light cigarette smoker (1-9 cigs/day) (finding)","SNOMED","smoking"),
("160606002","Very heavy cigarette smoker (40+ cigs/day) (finding)","SNOMED","smoking"),
("160613002","Admitted tobacco consumption possibly untrue (finding)","SNOMED","smoking"),
("160619003","Rolls own cigarettes (finding)","SNOMED","smoking"),
("230056004","Cigarette consumption (observable entity)","SNOMED","smoking"),
("230057008","Cigar consumption (observable entity)","SNOMED","smoking"),
("230058003","Pipe tobacco consumption (observable entity)","SNOMED","smoking"),
("230060001","Light cigarette smoker (finding)","SNOMED","smoking"),
("230062009","Moderate cigarette smoker (finding)","SNOMED","smoking"),
("230065006","Chain smoker (finding)","SNOMED","smoking"),
("266918002","Tobacco smoking consumption (observable entity)","SNOMED","smoking"),
("446172000","Failed attempt to stop smoking (finding)","SNOMED","smoking"),
("449868002","Smokes tobacco daily (finding)","SNOMED","smoking"),
("56578002","Moderate smoker (20 or less per day) (finding)","SNOMED","smoking"),
("56771006","Heavy smoker (over 20 per day) (finding)","SNOMED","smoking"),
("59978006","Cigar smoker (finding)","SNOMED","smoking"),
("65568007","Cigarette smoker (finding)","SNOMED","smoking"),
("134406006","Smoking reduced (finding)","SNOMED","smoking"),
("160604004","Moderate cigarette smoker (10-19 cigs/day) (finding)","SNOMED","smoking"),
("160605003","Heavy cigarette smoker (20-39 cigs/day) (finding)","SNOMED","smoking"),
("160612007","Keeps trying to stop smoking (finding)","SNOMED","smoking"),
("160616005","Trying to give up smoking (finding)","SNOMED","smoking"),
("203191000000107","Wants to stop smoking (finding)","SNOMED","smoking"),
("225934006","Smokes in bed (finding)","SNOMED","smoking"),
("230059006","Occasional cigarette smoker (finding)","SNOMED","smoking"),
("230063004","Heavy cigarette smoker (finding)","SNOMED","smoking"),
("230064005","Very heavy cigarette smoker (finding)","SNOMED","smoking"),
("266920004","Trivial cigarette smoker (less than one cigarette/day) (finding)","SNOMED","smoking"),
("266929003","Smoking started (finding)","SNOMED","smoking"),
("308438006","Smoking restarted (finding)","SNOMED","smoking"),
("394871007","Thinking about stopping smoking (finding)","SNOMED","smoking"),
("394872000","Ready to stop smoking (finding)","SNOMED","smoking"),
("394873005","Not interested in stopping smoking (finding)","SNOMED","smoking"),
("401159003","Reason for restarting smoking (observable entity)","SNOMED","smoking"),
("413173009","Minutes from waking to first tobacco consumption (observable entity)","SNOMED","smoking"),
("428041000124106","Occasional tobacco smoker (finding)","SNOMED","smoking"),
("77176002","Smoker (finding)","SNOMED","smoking"),
("82302008","Pipe smoker (finding)","SNOMED","smoking"),
("836001000000109","Waterpipe tobacco consumption (observable entity)","SNOMED","smoking"),
("160603005","Light cigarette smoker (1-9 cigs/day) (finding)","SNOMED","smoking"),
("160612007","Keeps trying to stop smoking (finding)","SNOMED","smoking"),
("160613002","Admitted tobacco consumption possibly untrue (finding)","SNOMED","smoking"),
("160616005","Trying to give up smoking (finding)","SNOMED","smoking"),
("160619003","Rolls own cigarettes (finding)","SNOMED","smoking"),
("160625004","Date ceased smoking (observable entity)","SNOMED","smoking"),
("225934006","Smokes in bed (finding)","SNOMED","smoking"),
("230056004","Cigarette consumption (observable entity)","SNOMED","smoking"),
("230057008","Cigar consumption (observable entity)","SNOMED","smoking"),
("230059006","Occasional cigarette smoker (finding)","SNOMED","smoking"),
("230060001","Light cigarette smoker (finding)","SNOMED","smoking"),
("230062009","Moderate cigarette smoker (finding)","SNOMED","smoking"),
("230063004","Heavy cigarette smoker (finding)","SNOMED","smoking"),
("230064005","Very heavy cigarette smoker (finding)","SNOMED","smoking"),
("266920004","Trivial cigarette smoker (less than one cigarette/day) (finding)","SNOMED","smoking"),
("266929003","Smoking started (finding)","SNOMED","smoking"),
("394872000","Ready to stop smoking (finding)","SNOMED","smoking"),
("401159003","Reason for restarting smoking (observable entity)","SNOMED","smoking"),
("449868002","Smokes tobacco daily (finding)","SNOMED","smoking"),
("65568007","Cigarette smoker (finding)","SNOMED","smoking"),
("134406006","Smoking reduced (finding)","SNOMED","smoking"),
("160604004","Moderate cigarette smoker (10-19 cigs/day) (finding)","SNOMED","smoking"),
("160605003","Heavy cigarette smoker (20-39 cigs/day) (finding)","SNOMED","smoking"),
("160606002","Very heavy cigarette smoker (40+ cigs/day) (finding)","SNOMED","smoking"),
("203191000000107","Wants to stop smoking (finding)","SNOMED","smoking"),
("308438006","Smoking restarted (finding)","SNOMED","smoking"),
("394871007","Thinking about stopping smoking (finding)","SNOMED","smoking"),
("394873005","Not interested in stopping smoking (finding)","SNOMED","smoking"),
("401201003","Cigarette pack-years (observable entity)","SNOMED","smoking"),
("413173009","Minutes from waking to first tobacco consumption (observable entity)","SNOMED","smoking"),
("428041000124106","Occasional tobacco smoker (finding)","SNOMED","smoking"),
("446172000","Failed attempt to stop smoking (finding)","SNOMED","smoking"),
("56578002","Moderate smoker (20 or less per day) (finding)","SNOMED","smoking"),
("56771006","Heavy smoker (over 20 per day) (finding)","SNOMED","smoking"),
("59978006","Cigar smoker (finding)","SNOMED","smoking"),
("77176002","Smoker (finding)","SNOMED","smoking"),
("82302008","Pipe smoker (finding)","SNOMED","smoking"),
("836001000000109","Waterpipe tobacco consumption (observable entity)","SNOMED","smoking"),
("53896009","Tolerant ex-smoker (finding)","SNOMED","smoking"),
("1092041000000100","Ex-very heavy smoker (40+/day) (finding)","SNOMED","smoking"),
("1092091000000100","Ex-moderate smoker (10-19/day) (finding)","SNOMED","smoking"),
("160620009","Ex-pipe smoker (finding)","SNOMED","smoking"),
("160621008","Ex-cigar smoker (finding)","SNOMED","smoking"),
("228486009","Time since stopped smoking (observable entity)","SNOMED","smoking"),
("266921000","Ex-trivial cigarette smoker (<1/day) (finding)","SNOMED","smoking"),
("266922007","Ex-light cigarette smoker (1-9/day) (finding)","SNOMED","smoking"),
("266923002","Ex-moderate cigarette smoker (10-19/day) (finding)","SNOMED","smoking"),
("266928006","Ex-cigarette smoker amount unknown (finding)","SNOMED","smoking"),
("281018007","Ex-cigarette smoker (finding)","SNOMED","smoking"),
("735128000","Ex-smoker for less than 1 year (finding)","SNOMED","smoking"),
("8517006","Ex-smoker (finding)","SNOMED","smoking"),
("1092031000000100","Ex-smoker amount unknown (finding)","SNOMED","smoking"),
("1092071000000100","Ex-heavy smoker (20-39/day) (finding)","SNOMED","smoking"),
("1092111000000100","Ex-light smoker (1-9/day) (finding)","SNOMED","smoking"),
("1092131000000100","Ex-trivial smoker (<1/day) (finding)","SNOMED","smoking"),
("160617001","Stopped smoking (finding)","SNOMED","smoking"),
("160625004","Date ceased smoking (observable entity)","SNOMED","smoking"),
("266924008","Ex-heavy cigarette smoker (20-39/day) (finding)","SNOMED","smoking"),
("266925009","Ex-very heavy cigarette smoker (40+/day) (finding)","SNOMED","smoking"),
("360890004","Intolerant ex-smoker (finding)","SNOMED","smoking"),
("360900008","Aggressive ex-smoker (finding)","SNOMED","smoking"),
("48031000119106","Ex-smoker for more than 1 year (finding)","SNOMED","smoking"),
("492191000000103","Ex roll-up cigarette smoker (finding)","SNOMED","smoking"),
("53896009","Tolerant ex-smoker (finding)","SNOMED","smoking"),
("735112005","Date ceased using moist tobacco (observable entity)","SNOMED","smoking"),
("1092041000000100","Ex-very heavy smoker (40+/day) (finding)","SNOMED","smoking"),
("1092071000000100","Ex-heavy smoker (20-39/day) (finding)","SNOMED","smoking"),
("1092111000000100","Ex-light smoker (1-9/day) (finding)","SNOMED","smoking"),
("228486009","Time since stopped smoking (observable entity)","SNOMED","smoking"),
("266921000","Ex-trivial cigarette smoker (<1/day) (finding)","SNOMED","smoking"),
("266923002","Ex-moderate cigarette smoker (10-19/day) (finding)","SNOMED","smoking"),
("266928006","Ex-cigarette smoker amount unknown (finding)","SNOMED","smoking"),
("360900008","Aggressive ex-smoker (finding)","SNOMED","smoking"),
("492191000000103","Ex roll-up cigarette smoker (finding)","SNOMED","smoking"),
("735112005","Date ceased using moist tobacco (observable entity)","SNOMED","smoking"),
("735128000","Ex-smoker for less than 1 year (finding)","SNOMED","smoking"),
("1092031000000100","Ex-smoker amount unknown (finding)","SNOMED","smoking"),
("1092091000000100","Ex-moderate smoker (10-19/day) (finding)","SNOMED","smoking"),
("1092131000000100","Ex-trivial smoker (<1/day) (finding)","SNOMED","smoking"),
("160617001","Stopped smoking (finding)","SNOMED","smoking"),
("160620009","Ex-pipe smoker (finding)","SNOMED","smoking"),
("160621008","Ex-cigar smoker (finding)","SNOMED","smoking"),
("230058003","Pipe tobacco consumption (observable entity)","SNOMED","smoking"),
("230065006","Chain smoker (finding)","SNOMED","smoking"),
("266918002","Tobacco smoking consumption (observable entity)","SNOMED","smoking"),
("266922007","Ex-light cigarette smoker (1-9/day) (finding)","SNOMED","smoking"),
("266924008","Ex-heavy cigarette smoker (20-39/day) (finding)","SNOMED","smoking"),
("266925009","Ex-very heavy cigarette smoker (40+/day) (finding)","SNOMED","smoking"),
("281018007","Ex-cigarette smoker (finding)","SNOMED","smoking"),
("360890004","Intolerant ex-smoker (finding)","SNOMED","smoking"),
("48031000119106","Ex-smoker for more than 1 year (finding)","SNOMED","smoking"),
("8517006","Ex-smoker (finding)","SNOMED","smoking")
AS {project_prefix}_smoking_codelist (code, term, system, codelist)
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW smoking AS
# MAGIC SELECT NHS_NUMBER_DEID, "smoking" as predictor, date FROM global_temp.gdppr_snomed
# MAGIC WHERE CODE IN (SELECT code FROM global_temp.ccu037_02_smoking_codelist)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE GLOBAL TEMP VIEW smoking1 AS
# MAGIC SELECT DISTINCT NHS_NUMBER_DEID, predictor
# MAGIC FROM
# MAGIC (SELECT b.NHS_NUMBER_DEID, a.predictor, a.date, b.date_first_covid_event
# MAGIC FROM
# MAGIC (SELECT
# MAGIC   NHS_NUMBER_DEID,
# MAGIC   min(date) as date, predictor
# MAGIC FROM
# MAGIC   global_temp.smoking
# MAGIC GROUP BY
# MAGIC   NHS_NUMBER_DEID, predictor) a
# MAGIC LEFT JOIN 
# MAGIC   dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria b
# MAGIC ON
# MAGIC   a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID)
# MAGIC WHERE date < date_first_covid_event 

# COMMAND ----------

# spark.sql(f"""
# CREATE OR REPLACE GLOBAL TEMP VIEW smoking1 AS
#     SELECT NHS_NUMBER_DEID, predictor FROM
#     (SELECT a.NHS_NUMBER_DEID, a.date, a.predictor, b.date_first_covid_event
#     FROM global_temp.smoking AS a
#     LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
#     ON a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID) 
#     WHERE date < date_first_covid_event  """)

# COMMAND ----------

# MAGIC %md
# MAGIC # 11. Join tables

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_other_qrisk_phenotypes AS
SELECT * FROM global_temp.schizophrenia1
UNION ALL
SELECT * FROM global_temp.bipolardisorder1
UNION ALL
SELECT * FROM global_temp.depression1
UNION ALL
SELECT * FROM global_temp.antipsychotic1
UNION ALL
SELECT * FROM global_temp.migraine1
UNION ALL
SELECT * FROM global_temp.erectiledysfunction1
UNION ALL
SELECT * FROM global_temp.steroid1
UNION ALL
SELECT * FROM global_temp.CKD1
UNION ALL
SELECT * FROM global_temp.AF1
UNION ALL
SELECT * FROM global_temp.RA1
UNION ALL
SELECT * FROM global_temp.smoking1
""")

# COMMAND ----------

drop_table("ccu037_other_qrisk_phenotypes")
create_table("ccu037_other_qrisk_phenotypes")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT(predictor), COUNT(predictor) AS count FROM dars_nic_391419_j3w9t_collab.ccu037_other_qrisk_phenotypes GROUP BY predictor
# MAGIC ORDER BY count DESC
