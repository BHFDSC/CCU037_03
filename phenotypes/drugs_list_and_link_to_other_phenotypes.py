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

# MAGIC %sql
# MAGIC SELECT * FROM bhf_cvd_covid_uk_byod.bhf_covid_uk_phenotypes_20210127

# COMMAND ----------

# MAGIC %md # History of cardiovascular  disease 
# MAGIC
# MAGIC ### In the prior year to index date (ie. covid diagnosis)
# MAGIC ### Ever (eny time prior index date)
# MAGIC
# MAGIC Use same list that CVD event

# COMMAND ----------

# MAGIC %md # Predictiors that we did not have before in our ccu0037_01 folder

# COMMAND ----------

# MAGIC %md ## DRUGS IN GENERAL
# MAGIC
# MAGIC Within 'https://db.core.data.digital.nhs.uk/#notebook/4873002/command/4873007', there is the section DRUGS.
# MAGIC They obtained the llists from the table 'ccu002_03_primary_care_meds_dars_nic_391419_j3w9t' with is a freezed table from 'primary_care_meds_dars_nic_391419_j3w9t'
# MAGIC
# MAGIC They freeze the table in the following library:
# MAGIC https://db.core.data.digital.nhs.uk/#notebook/2904418/command/2904438

# COMMAND ----------

# MAGIC %md ##Anti-diabetic therapies 

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW antidiabeticBNF AS
# MAGIC SELECT * FROM VALUES 
# MAGIC ('0601023A0','Acarbose'),('0601023AS','Albiglutide'),('0601023AK','Alogliptin'),('0601023AJ','Alogliptin/metformin'),('0601023AM','Canagliflozin'),('0601023AP','Canagliflozin/metformin'),('0601021E0','Chlorpropamide'),('0601023AG','Dapagliflozin'),('0601023AL','Dapagliflozin/metformin'),('0601023AQ','Dulaglutide'),('0601023AN','Empagliflozin'),('0601023AY','Empagliflozin/linagliptin'),('0601023AR','Empagliflozin/metformin'),('0601023AX','Ertugliflozin'),('0601023Y0','Exenatide'),('0601021H0','Glibenclamide'),('0601021M0','Gliclazide'),('0601021A0','Glimepiride'),('0601021P0','Glipizide'),('0601023I0','Guar gum'),('0601023AU','Ins degludec/liraglutide'),('0601023AE','Linagliptin'),('0601023AF','Linagliptin/metformin'),('0601023AB','Liraglutide'),('0601023AI','Lixisenatide'),('0601022B0','Metformin hydrochloride'),('0601023W0','Metformin hydrochloride/pioglitazone'),('0601023V0','Metformin hydrochloride/rosiglitazone'),('0601023AD','Metformin hydrochloride/sitagliptin'),('0601023Z0','Metformin hydrochloride/vildagliptin'),('0601023U0','Nateglinide'),('0601023B0','Pioglitazone hydrochloride'),('0601023R0','Repaglinide'),('0601023S0','Rosiglitazone'),('0601023AC','Saxagliptin'),('0601023AV','Saxagliptin/dapagliflozin'),('0601023AH','Saxagliptin/metformin'),('0601023AW','Semaglutide'),('0601023X0','Sitagliptin'),('0601021X0','Tolbutamide'),('0601023AA','Vildagliptin')
# MAGIC AS tab(BNFcode, dantidiabetic_name);

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT DISTINCT PrescribedBNFCode AS code, PrescribedBNFName AS term, 'BNF' AS system, 'antidiabetic_drugs' AS codelist
# MAGIC FROM dars_nic_391419_j3w9t_collab.ccu002_03_primary_care_meds_dars_nic_391419_j3w9t
# MAGIC WHERE left(PrescribedBNFCode, 9) IN (SELECT BNFcode FROM global_temp.antidiabeticBNF)

# COMMAND ----------

# MAGIC %sql -- SEE there are records
# MAGIC SELECT * FROM dars_nic_391419_j3w9t.primary_care_meds_dars_nic_391419_j3w9t WHERE (Person_ID_DEID IS NOT NULL AND left(PrescribedBNFCode, 9) IN (SELECT BNFcode FROM global_temp.antidiabeticBNF) )

# COMMAND ----------

# MAGIC %md ##Anti-hypertensive drug/s 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bhf_cvd_covid_uk_byod.bhf_covid_uk_phenotypes_20210127 where name = 'hypertension_drugs'

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE
# MAGIC OR REPLACE GLOBAL TEMPORARY VIEW hypertension_drugs AS 
# MAGIC SELECT * FROM bhf_cvd_covid_uk_byod.bhf_covid_uk_phenotypes_20210127 where name = 'hypertension_drugs'

# COMMAND ----------

# MAGIC %sql  -- SEE there are records using dmd
# MAGIC SELECT * FROM dars_nic_391419_j3w9t.primary_care_meds_dars_nic_391419_j3w9t WHERE (Person_ID_DEID IS NOT NULL AND PrescribeddmdCode IN (SELECT code FROM global_temp.hypertension_drugs) )

# COMMAND ----------

# MAGIC %sql  -- SEE if we found BNF codes with dmd
# MAGIC SELECT PrescribeddmdCode, PrescribedBNFCode FROM dars_nic_391419_j3w9t.primary_care_meds_dars_nic_391419_j3w9t WHERE (PrescribeddmdCode IN (SELECT code FROM global_temp.hypertension_drugs) )

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM dars_nic_391419_j3w9t.primary_care_meds_dars_nic_391419_j3w9t WHERE (Person_ID_DEID IS NOT NULL AND PrescribeddmdCode IN (SELECT code FROM global_temp.hypertension_drugs) )

# COMMAND ----------

# MAGIC %md ###Manual list of anti-hypertensive drugs:
# MAGIC
# MAGIC ('Clonidine hydrochloride','0205020E0'),('Guanfacine hydrochloride','0205020G0'),('Methyldopa','0205020H0'),('Moxonidine','0205020M0') -- Centrally-acting antihypertensive drugs
# MAGIC
# MAGIC ('Ambrisentan','0205010X0'),('Bosentan','0205010U0'),('Diazoxide','0205010E0'),('Hydralazine hydrochloride','0205010J0'),('Iloprost','0205010V0'),('Macitentan','0205010AA'),('Minoxidil','0205010N0'),('Riociguat','0205010AB'),('Sildenafil (Vasodilator Antihypertensive)','0205010Y0'),('Sitaxentan sodium','0205010W0'),('Tadalafil (Vasodilator Antihypertensive)','0205010Z0'),('Vericiguat','0205010AC') -- Vasodilator antihypertensive drugs

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

# MAGIC %sql -- SEE there are records
# MAGIC SELECT * FROM dars_nic_391419_j3w9t.primary_care_meds_dars_nic_391419_j3w9t WHERE (Person_ID_DEID IS NOT NULL AND left(PrescribedBNFCode, 9) IN (SELECT BNFcode FROM global_temp.antihypertensiveBNF))

# COMMAND ----------

# MAGIC %md ##Anti-thrombotic or anti-coagulant treatment
# MAGIC
# MAGIC -- Antithrombotic drugs in routine use include antiplatelet drugs and anticoagulants 
# MAGIC
# MAGIC ie., to include anti-coagulants and antiplatelet drugs

# COMMAND ----------

# MAGIC %sql
# MAGIC -- FROM https://db.core.data.digital.nhs.uk/#notebook/4873002/command/4873007
# MAGIC
# MAGIC -- ANTI-COAGULANTS
# MAGIC
# MAGIC SELECT DISTINCT PrescribedBNFCode AS code, PrescribedBNFName AS term, 'BNF' AS system, 'anticoagulant_drugs' AS codelist
# MAGIC FROM dars_nic_391419_j3w9t_collab.ccu002_03_primary_care_meds_dars_nic_391419_j3w9t
# MAGIC WHERE (left(PrescribedBNFCode, 6) = '020802'
# MAGIC        AND NOT (left(PrescribedBNFCode, 8) = '0208020I' OR left(PrescribedBNFCode, 8) = '0208020W'))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ANTIPLATELETS
# MAGIC
# MAGIC SELECT DISTINCT PrescribedBNFCode AS code, PrescribedBNFName AS term, 'BNF' AS system, 'antiplatelet_drugs' AS codelist
# MAGIC FROM dars_nic_391419_j3w9t_collab.ccu002_03_primary_care_meds_dars_nic_391419_j3w9t
# MAGIC WHERE (left(PrescribedBNFCode, 7) = '0209000')

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- SEE there are records with ANTI-COAGULANTS
# MAGIC SELECT * FROM dars_nic_391419_j3w9t.primary_care_meds_dars_nic_391419_j3w9t WHERE 
# MAGIC (Person_ID_DEID IS NOT NULL AND left(PrescribedBNFCode, 6) = '020802'
# MAGIC        AND NOT (left(PrescribedBNFCode, 8) = '0208020I' OR left(PrescribedBNFCode, 8) = '0208020W'))

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- SEE there are records with ANTIPLATELETS
# MAGIC SELECT DISTINCT * FROM dars_nic_391419_j3w9t_collab.ccu002_03_primary_care_meds_dars_nic_391419_j3w9t
# MAGIC WHERE (Person_ID_DEID IS NOT NULL AND left(PrescribedBNFCode, 7) = '0209000')

# COMMAND ----------

# MAGIC %md ##use of statin/s

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT PrescribedBNFCode AS code, PrescribedBNFName AS term, 'BNF' AS system, 'statin_drugs' AS codelist FROM dars_nic_391419_j3w9t.primary_care_meds_dars_nic_391419_j3w9t WHERE (left(PrescribedBNFCode, 7) =  "0212000"   AND NOT (left(PrescribedBNFCode, 9) = '0212000U0')) -- Nicotinic acid (is a nutrient, not an statin)

# COMMAND ----------

# MAGIC %sql -- See there are records
# MAGIC SELECT * FROM dars_nic_391419_j3w9t.primary_care_meds_dars_nic_391419_j3w9t WHERE (Person_ID_DEID IS NOT NULL AND left(PrescribedBNFCode, 7) =  "0212000"   AND NOT (left(PrescribedBNFCode, 9) = '0212000U0')) -- Nicotinic acid (is a nutrient, not an statin)

# COMMAND ----------

# MAGIC %md ##Hyperlipidaemia (recorded diagnosis) 
# MAGIC Proxy: we can get hypercoholesterolemia (instead of high lipids it says to us high coholesterol)
# MAGIC
# MAGIC Following 'https://db.core.data.digital.nhs.uk/#notebook/4873002/command/4873007' structure:

# COMMAND ----------

# MAGIC %md # Predictiors that we already have and were to find its code:

# COMMAND ----------

# MAGIC %md ## SMOKING
# MAGIC
# MAGIC ### see library: 
# MAGIC https://db.core.data.digital.nhs.uk/#notebook/7138108/command/7286155 

# COMMAND ----------

# MAGIC %md ##Chronic kidney disease 
# MAGIC
# MAGIC Following 'https://db.core.data.digital.nhs.uk/#notebook/4873002/command/4873007' structure:

# COMMAND ----------

# MAGIC %md ##Dementia
# MAGIC
# MAGIC ### see again library: 
# MAGIC https://db.core.data.digital.nhs.uk/#notebook/7138108/command/7286155 

# COMMAND ----------

# MAGIC %md ##COPD
# MAGIC
# MAGIC ### see again library: 
# MAGIC https://db.core.data.digital.nhs.uk/#notebook/7138108/command/7286155 
# MAGIC
# MAGIC ### OR OPTION B below from:
# MAGIC https://db.core.data.digital.nhs.uk/#notebook/4873002/command/4873019

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace global temporary view ccu002_03_codelist_COPD as
# MAGIC select *
# MAGIC from bhf_cvd_covid_uk_byod.bhf_covid_uk_phenotypes_20210127
# MAGIC where name like '%COPD%'
# MAGIC and 
# MAGIC (terminology like 'SNOMED'
# MAGIC or terminology like 'ICD10')

# COMMAND ----------

# MAGIC %md ##Liver disease
# MAGIC
# MAGIC ### see again library: 
# MAGIC https://db.core.data.digital.nhs.uk/#notebook/7138108/command/7286155 

# COMMAND ----------

# MAGIC %md ##Cancer (instead of Solid tumor)
# MAGIC
# MAGIC ### see again library: 
# MAGIC https://db.core.data.digital.nhs.uk/#notebook/7138108/command/7286155 

# COMMAND ----------

# MAGIC %md ##Leukemia
# MAGIC
# MAGIC #Not include, it's from the charlson index and we cover it within the Cancer variable.

# COMMAND ----------

# MAGIC %md ##Lymphoma 
# MAGIC
# MAGIC #Not include, it's from the charlson index
