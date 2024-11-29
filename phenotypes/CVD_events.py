# Databricks notebook source
# MAGIC %md
# MAGIC # Creating a table with post-covid infection CVD events

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

# MAGIC %md
# MAGIC # Filter phenotypes table to just include CVD categories

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM dars_nic_391419_j3w9t_collab.ccu037_caliber

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM dars_nic_391419_j3w9t_collab.ccu013_caliber_category_mapping

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW phenotype_categories AS
   SELECT a.person_id_deid, a.phenotype, a.date, a.code, a.terminology, b.category
   FROM dars_nic_391419_j3w9t_collab.ccu037_caliber AS a
   LEFT JOIN dars_nic_391419_j3w9t_collab.ccu013_caliber_category_mapping as b
   ON a.phenotype = b.phenotype
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM global_temp.phenotype_categories

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT phenotype FROM global_temp.phenotype_categories WHERE category LIKE "%cancers%" AND  phenotype NOT LIKE "%malignancy%"

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW CVD_phenotypes AS
   SELECT person_id_deid, phenotype,date,code,terminology,category
   FROM global_temp.phenotype_categories WHERE category LIKE '%diseases_of_the_circulatory_system%'
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM global_temp.CVD_phenotypes

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT phenotype FROM global_temp.CVD_phenotypes

# COMMAND ----------

# MAGIC %md
# MAGIC # Select all post covid CVD events

# COMMAND ----------

# all post covid cvd events
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_post_covid_cvd_events_list AS
    SELECT person_id_deid, phenotype, date, code, terminology FROM
    (SELECT a.person_id_deid, a.phenotype, a.date, a.code, a.terminology, a.category, b.date_first_covid_event
    FROM global_temp.CVD_phenotypes AS a
    LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID)
    WHERE date_first_covid_event < date
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(distinct person_id_deid) FROM global_temp.ccu037_post_covid_cvd_events_list

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT code, phenotype FROM global_temp.ccu037_post_covid_cvd_events_list ORDER BY code

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT person_id_deid) FROM global_temp.ccu037_post_covid_cvd_events_list

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT(phenotype), COUNT(phenotype) AS count FROM global_temp.ccu037_post_covid_cvd_events_list GROUP BY phenotype
# MAGIC ORDER BY count DESC

# COMMAND ----------

drop_table("ccu037_post_covid_cvd_events_list")
create_table("ccu037_post_covid_cvd_events_list")

# COMMAND ----------

# creating a pivoted table with post covid CVD events list
cvd_events = spark.sql(f"""
SELECT person_id_deid, phenotype, 1 as value FROM global_temp.ccu037_post_covid_cvd_events_list
""")

# COMMAND ----------

cvd_events = cvd_events \
  .to_koalas() \
  .pivot(index='person_id_deid', 
         columns='phenotype', 
         values='value') \
  .fillna(0) \
  .reset_index() \
  .to_spark()

# COMMAND ----------

display(cvd_events)

# COMMAND ----------

cvd_events.createOrReplaceGlobalTempView("ccu037_post_covid_cvd_events_binary")
drop_table("ccu037_post_covid_cvd_events_binary")
create_table("ccu037_post_covid_cvd_events_binary")

# COMMAND ----------

# MAGIC %md
# MAGIC # Creating a CVD events table for any history, CVD 1 year before index date, and 90, 180, 1 year and 2 years post-covid

# COMMAND ----------

# cvd event anytime before infection
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_pre_covid_cvd_events AS
    SELECT person_id_deid, "pre_covid_CVD_event" as predictor FROM
    (SELECT a.person_id_deid, a.phenotype, a.date, a.code, a.terminology, a.category, b.date_first_covid_event
    FROM global_temp.CVD_phenotypes AS a
    LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID)
    WHERE date < date_first_covid_event
    AND phenotype NOT LIKE "%hypertension%" 
    AND phenotype NOT LIKE "%primary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%secondary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%raynauds_syndrome%" 
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT person_id_deid) FROM global_temp.ccu037_pre_covid_cvd_events

# COMMAND ----------



# COMMAND ----------

# cvd events 1 year before covid infection
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_1_year_pre_covid_cvd_events AS
    SELECT person_id_deid, "1_year_pre_covid_CVD_event" as predictor FROM
    (SELECT a.person_id_deid, a.phenotype, a.date, a.code, a.terminology, a.category, b.date_first_covid_event
    FROM global_temp.CVD_phenotypes AS a
    LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID)
    WHERE (date_first_covid_event - INTERVAL 365 DAY) < date AND date < date_first_covid_event 
    AND phenotype NOT LIKE "%hypertension%" 
    AND phenotype NOT LIKE "%primary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%secondary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%raynauds_syndrome%" 
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT person_id_deid) FROM global_temp.ccu037_1_year_pre_covid_cvd_events

# COMMAND ----------

# cvd events 90 days after infection
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_90_day_post_covid_cvd_events AS
    SELECT person_id_deid, "90_day_CVD_event" as predictor, date FROM
    (SELECT a.person_id_deid, a.phenotype, a.date, a.code, a.terminology, a.category, b.date_first_covid_event
    FROM global_temp.CVD_phenotypes AS a
    LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID)
    WHERE date_first_covid_event < date AND date < (date_first_covid_event + INTERVAL 90 DAY) 
    AND phenotype NOT LIKE "%hypertension%" 
    AND phenotype NOT LIKE "%primary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%secondary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%raynauds_syndrome%" 
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT person_id_deid) FROM global_temp.ccu037_90_day_post_covid_cvd_events

# COMMAND ----------

# cvd events 180 days after infection
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_180_day_post_covid_cvd_events AS
    SELECT person_id_deid, "180_day_CVD_event" as predictor, date FROM
    (SELECT a.person_id_deid, a.phenotype, a.date, a.code, a.terminology, a.category, b.date_first_covid_event
    FROM global_temp.CVD_phenotypes AS a
    LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID)
    WHERE date_first_covid_event < date AND date < (date_first_covid_event + INTERVAL 180 DAY)
    AND phenotype NOT LIKE "%hypertension%" 
    AND phenotype NOT LIKE "%primary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%secondary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%raynauds_syndrome%"
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT person_id_deid) FROM global_temp.ccu037_180_day_post_covid_cvd_events

# COMMAND ----------

# cvd events 1 year after infection
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_1_year_post_covid_cvd_events AS
    SELECT person_id_deid, "1_year_CVD_event" as predictor, date FROM
    (SELECT a.person_id_deid, a.phenotype, a.date, a.code, a.terminology, a.category, b.date_first_covid_event
    FROM global_temp.CVD_phenotypes AS a
    LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID)
    WHERE date_first_covid_event < date AND date < (date_first_covid_event + INTERVAL 365 DAY)
    AND phenotype NOT LIKE "%hypertension%" 
    AND phenotype NOT LIKE "%primary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%secondary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%raynauds_syndrome%"
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT person_id_deid) FROM global_temp.ccu037_1_year_post_covid_cvd_events

# COMMAND ----------

# cvd events 2 years after infection
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_2_years_post_covid_cvd_events AS
    SELECT person_id_deid, "2_years_CVD_event" as predictor FROM
    (SELECT a.person_id_deid, a.phenotype, a.date, a.code, a.terminology, a.category, b.date_first_covid_event
    FROM global_temp.CVD_phenotypes AS a
    LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID)
    WHERE date_first_covid_event < date AND date < (date_first_covid_event + INTERVAL 730 DAY)
    AND phenotype NOT LIKE "%hypertension%" 
    AND phenotype NOT LIKE "%primary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%secondary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%raynauds_syndrome%"
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT person_id_deid) FROM global_temp.ccu037_2_years_post_covid_cvd_events

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_all_cvd_events AS
SELECT * FROM global_temp.ccu037_pre_covid_cvd_events
UNION ALL
SELECT * FROM global_temp.ccu037_1_year_pre_covid_cvd_events
UNION ALL
SELECT * FROM global_temp.ccu037_90_day_post_covid_cvd_events
UNION ALL
SELECT * FROM global_temp.ccu037_180_day_post_covid_cvd_events
UNION ALL
SELECT * FROM global_temp.ccu037_1_year_post_covid_cvd_events
UNION ALL
SELECT * FROM global_temp.ccu037_2_years_post_covid_cvd_events
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM GLOBAL_TEMP.ccu037_CVD_all_events_list

# COMMAND ----------

drop_table("ccu037_all_cvd_events")
create_table("ccu037_all_cvd_events")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pivot

# COMMAND ----------

CVD = spark.sql(f"""
  SELECT person_id_deid, predictor, 1 as value FROM GLOBAL_TEMP.ccu037_CVD_all_events_list """)

# COMMAND ----------

CVD = CVD \
  .to_koalas() \
  .pivot(index='person_id_deid', 
         columns='predictor', 
         values='value') \
  .fillna(0) \
  .reset_index() \
  .to_spark()
# Reset index to breakout ids to separate col

# COMMAND ----------

display(CVD)

# COMMAND ----------

CVD.createOrReplaceGlobalTempView("ccu037_CVD_events_table")
drop_table("ccu037_CVD_events_table")
create_table("ccu037_CVD_events_table")

# COMMAND ----------

# MAGIC %md
# MAGIC # Post-covid CVD events with dates

# COMMAND ----------

# cvd events within 2 years post covid infection
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW all_post_covid_cvd_events AS
    SELECT person_id_deid, phenotype, date AS cvd_date, date_first_covid_event as covid_date FROM
    (SELECT a.person_id_deid, a.phenotype, a.date, a.code, a.terminology, a.category, b.date_first_covid_event
    FROM global_temp.CVD_phenotypes AS a
    LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID)
    WHERE date_first_covid_event < date AND date < (date_first_covid_event + interval 730 DAY)
    AND phenotype NOT LIKE "%hypertension%" 
    AND phenotype NOT LIKE "%primary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%secondary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%raynauds_syndrome%" 
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT person_id_deid) FROM global_temp.all_post_covid_cvd_events

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_02_earliest_post_covid_cvd_event AS
SELECT person_id_deid, phenotype, min(cvd_date) as min_cvd_date, covid_date
FROM global_temp.all_post_covid_cvd_events
GROUP BY person_id_deid, PHENOTYPE, COVID_DATE
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT PERSON_ID_DEID) FROM global_temp.ccu037_02_post_covid_cvd_events

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM global_temp.ccu037_02_post_covid_cvd_events

# COMMAND ----------

drop_table("ccu037_02_earliest_post_covid_cvd_event")
create_table("ccu037_02_earliest_post_covid_cvd_event")

# COMMAND ----------

# MAGIC %md
# MAGIC # CVD events for each date

# COMMAND ----------

CVD = spark.sql(f"""
  SELECT person_id_deid, predictor, 1 as value FROM dars_nic_391419_j3w9t_collab.ccu037_all_cvd_events """)

# COMMAND ----------

CVD = CVD \
  .to_koalas() \
  .pivot(index='person_id_deid', 
         columns='predictor', 
         values='value') \
  .fillna(0) \
  .reset_index() \
  .to_spark()

# COMMAND ----------

CVD.createOrReplaceGlobalTempView("CVD_events_table")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM global_temp.CVD_events_table

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW post_covid_cvd_events AS
SELECT person_id_deid, 90_day_CVD_event as post_covid_cvd_event_90_day, 180_day_CVD_event AS post_covid_cvd_event_180_day, 1_year_CVD_event AS post_covid_cvd_event_1_year, 2_years_cvd_event AS post_covid_cvd_event_2_years
FROM global_temp.CVD_events_table """)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM global_temp.post_covid_cvd_events

# COMMAND ----------



# COMMAND ----------

# cvd events within 90 days post covid infection
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW 90_day AS
    SELECT person_id_deid, date AS cvd_date, date_first_covid_event as covid_date FROM
    (SELECT a.person_id_deid, a.phenotype, a.date, a.code, a.terminology, a.category, b.date_first_covid_event
    FROM global_temp.CVD_phenotypes AS a
    LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID)
    WHERE date_first_covid_event < date AND date < (date_first_covid_event + interval 90 DAY)
    AND phenotype NOT LIKE "%hypertension%" 
    AND phenotype NOT LIKE "%primary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%secondary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%raynauds_syndrome%" 
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW earliest_90_day AS
SELECT person_id_deid, min(cvd_date) as date_cvd_90d
FROM global_temp.90_day
GROUP BY person_id_deid
""")

# COMMAND ----------

# cvd events within 180 days post covid infection
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW 180_day AS
    SELECT person_id_deid, date AS cvd_date, date_first_covid_event as covid_date FROM
    (SELECT a.person_id_deid, a.phenotype, a.date, a.code, a.terminology, a.category, b.date_first_covid_event
    FROM global_temp.CVD_phenotypes AS a
    LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID)
    WHERE date_first_covid_event < date AND date < (date_first_covid_event + interval 180 DAY)
    AND phenotype NOT LIKE "%hypertension%" 
    AND phenotype NOT LIKE "%primary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%secondary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%raynauds_syndrome%" 
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW earliest_180_day AS
SELECT person_id_deid, min(cvd_date) as date_cvd_180d
FROM global_temp.180_day
GROUP BY person_id_deid
""")

# COMMAND ----------

# cvd events within 1 year post covid infection
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW 1_year AS
    SELECT person_id_deid, date AS cvd_date, date_first_covid_event as covid_date FROM
    (SELECT a.person_id_deid, a.phenotype, a.date, a.code, a.terminology, a.category, b.date_first_covid_event
    FROM global_temp.CVD_phenotypes AS a
    LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID)
    WHERE date_first_covid_event < date AND date < (date_first_covid_event + interval 365 DAY)
    AND phenotype NOT LIKE "%hypertension%" 
    AND phenotype NOT LIKE "%primary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%secondary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%raynauds_syndrome%" 
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW earliest_1_year AS
SELECT person_id_deid, min(cvd_date) as date_cvd_1y
FROM global_temp.1_year
GROUP BY person_id_deid
""")

# COMMAND ----------

# cvd events within 2 years post covid infection
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW 2_years AS
    SELECT person_id_deid, date AS cvd_date, date_first_covid_event as covid_date FROM
    (SELECT a.person_id_deid, a.phenotype, a.date, a.code, a.terminology, a.category, b.date_first_covid_event
    FROM global_temp.CVD_phenotypes AS a
    LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID)
    WHERE date_first_covid_event < date AND date < (date_first_covid_event + interval 730 DAY)
    AND phenotype NOT LIKE "%hypertension%" 
    AND phenotype NOT LIKE "%primary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%secondary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%raynauds_syndrome%" 
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW earliest_2_years AS
SELECT person_id_deid, min(cvd_date) as date_cvd_2y
FROM global_temp.2_years
GROUP BY person_id_deid
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM global_temp.full_cohort

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT a.NHS_NUMBER_DEID, a.post_covid_event_90_day, a.post_covid_cvd_event_180_day,
# MAGIC a.post_covid_event_1_year, a.post_covid_cvd_event_2_years, b.date_cvd_90d, c.date_cvd_180d, d.date_cvd_1y, e.date_cvd_2y
# MAGIC FROM dars_nic_391419_j3w9t_collab.ccu037_02_final_cohort a
# MAGIC LEFT JOIN global_temp.earliest_90_day b
# MAGIC ON a.NHS_NUMBER_DEID = b.person_id_deid
# MAGIC LEFT JOIN global_temp.earliest_180_day c
# MAGIC ON a.NHS_NUMBER_DEID = c.person_id_deid
# MAGIC LEFT JOIN global_temp.earliest_1_year d
# MAGIC ON a.NHS_NUMBER_DEID = d.person_id_deid
# MAGIC LEFT JOIN global_temp.earliest_2_years e
# MAGIC ON a.NHS_NUMBER_DEID= e.person_id_deid

# COMMAND ----------

spark.sql(f""" CREATE OR REPLACE GLOBAL TEMP VIEW CV_AND_DATES AS
SELECT a.NHS_NUMBER_DEID, a.post_covid_event_90_day, a.post_covid_cvd_event_180_day,
a.post_covid_event_1_year, a.post_covid_cvd_event_2_years, b.date_cvd_90d, c.date_cvd_180d, d.date_cvd_1y, e.date_cvd_2y
FROM dars_nic_391419_j3w9t_collab.ccu037_02_final_cohort a
LEFT JOIN global_temp.earliest_90_day b
ON a.NHS_NUMBER_DEID = b.person_id_deid
LEFT JOIN global_temp.earliest_180_day c
ON a.NHS_NUMBER_DEID = c.person_id_deid
LEFT JOIN global_temp.earliest_1_year d
ON a.NHS_NUMBER_DEID = d.person_id_deid
LEFT JOIN global_temp.earliest_2_years e
ON a.NHS_NUMBER_DEID= e.person_id_deid
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_cvd_events_and_dates AS
# MAGIC SELECT NHS_NUMBER_DEID, post_covid_event_90_day as post_covid_cvd_event_90d, post_covid_cvd_event_180_day as post_covid_cvd_event_180d,
# MAGIC post_covid_event_1_year as post_covid_cvd_event_1y, post_covid_cvd_event_2_years as post_covid_cvd_event_2y, date_cvd_90d, date_cvd_180d, date_cvd_1y, date_cvd_2y, date_first_covid_event as covid_date
# MAGIC FROM (SELECT a.NHS_NUMBER_DEID, a.post_covid_event_90_day, a.post_covid_cvd_event_180_day, a.post_covid_event_1_year, a.post_covid_cvd_event_2_years, a.date_cvd_90d, a.date_cvd_180d, a.date_cvd_1y, a.date_cvd_2y, b.date_first_covid_event
# MAGIC FROM global_temp.CV_and_dates AS a
# MAGIC LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
# MAGIC ON a.NHS_NUMBER_DEID = b.NHS_NUMBER_DEID)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM GLOBAL_TEMP.ccu037_cvd_events_and_dates 

# COMMAND ----------

drop_table("ccu037_cvd_events_and_dates")
create_table("ccu037_cvd_events_and_dates")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM dars_nic_391419_j3w9t_collab.ccu037_cvd_events_and_dates

# COMMAND ----------

# MAGIC %md
# MAGIC ### Adding in 30 day and VTE outcome

# COMMAND ----------

# cvd events 2 years after infection
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW cvd_30d AS
    SELECT person_id_deid, "post_covid_cvd_event_30d" as predictor FROM
    (SELECT a.person_id_deid, a.phenotype, a.date, a.code, a.terminology, a.category, b.date_first_covid_event
    FROM global_temp.CVD_phenotypes AS a
    LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID)
    WHERE date_first_covid_event < date AND date < (date_first_covid_event + INTERVAL 30 DAY)
    AND phenotype NOT LIKE "%hypertension%" 
    AND phenotype NOT LIKE "%primary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%secondary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%raynauds_syndrome%"
""")

# COMMAND ----------

# cvd events 2 years after infection
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW VTE AS
    SELECT person_id_deid, "VTE_outcome" as predictor FROM
    (SELECT a.person_id_deid, a.phenotype, a.date, a.code, a.terminology, a.category, b.date_first_covid_event
    FROM global_temp.CVD_phenotypes AS a
    LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID)
    WHERE date_first_covid_event < date 
    AND phenotype LIKE "%venous_thromboembolic_disease_excl_pe%" 
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW VTE_AND_CVD AS
SELECT a.NHS_NUMBER_DEID AS person_id_deid, b.predictor
FROM dars_nic_391419_j3w9t_collab.ccu037_cohort_base_nhs_numbers as a
LEFT JOIN
(SELECT * FROM GLOBAL_TEMP.CVD_30D
UNION ALL
SELECT * FROM GLOBAL_TEMP.VTE) as b
ON a.NHS_NUMBER_DEID = b.person_id_deid
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM global_temp.VTE_AND_CVD

# COMMAND ----------

CVD = spark.sql(f"""
SELECT person_id_deid, predictor, 1 as value FROM GLOBAL_TEMP.VTE_AND_CVD """)

# COMMAND ----------

CVD = CVD \
  .to_koalas() \
  .pivot(index='person_id_deid', 
         columns='predictor', 
         values='value') \
  .fillna(0) \
  .reset_index() \
  .to_spark()

# COMMAND ----------

CVD.createOrReplaceGlobalTempView("VTE_AND_30_DAY")

# COMMAND ----------

display(CVD)

# COMMAND ----------

# cvd events within 2 years post covid infection
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW dates_30d AS
    SELECT person_id_deid, date AS cvd_date, date_first_covid_event as covid_date FROM
    (SELECT a.person_id_deid, a.phenotype, a.date, a.code, a.terminology, a.category, b.date_first_covid_event
    FROM global_temp.CVD_phenotypes AS a
    LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID)
    WHERE date_first_covid_event < date AND date < (date_first_covid_event + interval 30 DAY)
    AND phenotype NOT LIKE "%hypertension%" 
    AND phenotype NOT LIKE "%primary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%secondary_pulmonary_hypertension%" 
    AND phenotype NOT LIKE "%raynauds_syndrome%" 
""")

# COMMAND ----------

# VTE evnt post covid infection
spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW dates_VTE AS
    SELECT person_id_deid, date AS cvd_date, date_first_covid_event as covid_date FROM
    (SELECT a.person_id_deid, a.phenotype, a.date, a.code, a.terminology, a.category, b.date_first_covid_event
    FROM global_temp.CVD_phenotypes AS a
 LEFT JOIN dars_nic_391419_j3w9t_collab.ccu037_cohort_all_inclusion_criteria as b
    ON a.person_id_deid = b.NHS_NUMBER_DEID)
    WHERE date_first_covid_event < date 
    AND phenotype LIKE "%venous_thromboembolic_disease_excl_pe%" 
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW earliest_30d AS
SELECT person_id_deid, min(cvd_date) as date_cvd_30d
FROM global_temp.dates_30d
GROUP BY person_id_deid
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE GLOBAL TEMP VIEW earliest_VTE AS
SELECT person_id_deid, min(cvd_date) as date_VTE
FROM global_temp.dates_VTE
GROUP BY person_id_deid
""")

# COMMAND ----------

spark.sql
SELECT * FROM dars_nic_391419_j3w9t_collab.ccu037_cvd_events_and_dates

# COMMAND ----------

spark.sql(f""" CREATE OR REPLACE GLOBAL TEMP VIEW ccu037_cvd_outcomes_with_dates_3 AS
SELECT a.NHS_NUMBER_DEID, d.post_covid_cvd_event_30d, a.post_covid_cvd_event_90d, a.post_covid_cvd_event_180d,
a.post_covid_cvd_event_1y, a.post_covid_cvd_event_2y, d.VTE_outcome, b.date_cvd_30d, a.date_cvd_90d, a.date_cvd_180d, a.date_cvd_1y, a.date_cvd_2y, a.covid_date, c.date_VTE
FROM dars_nic_391419_j3w9t_collab.ccu037_cvd_events_and_dates a
LEFT JOIN global_temp.earliest_30d b
ON a.NHS_NUMBER_DEID = b.person_id_deid
LEFT JOIN global_temp.earliest_VTE c
ON a.NHS_NUMBER_DEID = c.person_id_deid
LEFT JOIN global_temp.VTE_AND_30_DAY d
ON a.NHS_NUMBER_DEID = d.person_id_deid
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM global_temp.ccu037_cvd_outcomes_with_dates_3 LIMIT 10

# COMMAND ----------

drop_table("ccu037_cvd_outcomes_with_dates_3")
create_table("ccu037_cvd_outcomes_with_dates_3")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Creating CVD predictors table
# MAGIC SELECT DISTINCT predictor FROM dars_nic_391419_j3w9t_collab.ccu037_all_cvd_events

# COMMAND ----------

spark
