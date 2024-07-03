# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:41:16 2024

@author: cpalmisano
"""

import pandas as pd
import sqlalchemy as sa
import numpy as np
from sqlalchemy.orm import sessionmaker 
import os
import pyodbc
import gc
import inspect
import datetime 
from datetime import date, timedelta
import logging
import sys
import time
from smtplib import SMTP
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path 
import configparser
from logging.handlers import RotatingFileHandler
import time
 
sys.path.insert(1, '// PATH /Logging/') 

current_time = time.time()
start_time = time.monotonic()
today = date.today()

# #!!!
####  ------------------------------- Wrappers -------------------------------------
#### @wraps for making sure the function inherits its name and properties
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f'{func.__name__} took {end - start:.6f} seconds to complete')
        return result
    return wrapper
  
####  ------------------------------- Wrappers -------------------------------------
    
@timeit
def close_all_connections():
    """
    This function will close all open connections
    """
    for obj in gc.get_objects():
        if inspect.ismodule(obj):
            for name in dir(obj):
                if name == "engine":
                    engine = getattr(obj, name)
                    if isinstance(engine, sa.engine.Engine):
                        engine.dispose()
                elif name == "conn":
                    conn = getattr(obj, name)
                    if conn is not None and isinstance(conn, pyodbc.Connection):
                        try:
                            conn.close()
                        except pyodbc.ProgrammingError:
                            pass
                elif name == "con":
                    con = getattr(obj, name)
                    if con is not None and isinstance(con, pyodbc.Connection):
                        try:
                            con.close()
                        except pyodbc.ProgrammingError:
                            pass
                elif name == "cursor":
                    cursor = getattr(obj, name)
                    if cursor is not None and isinstance(cursor, pyodbc.Cursor):
                        try:
                            cursor.close()
                        except pyodbc.ProgrammingError:
                            pass

####Connection 
@timeit
def connect_to_sql_server(server_name, database_name):
    # connect via pyodbc
    conn = pyodbc.connect(f'Driver={{SQL Server}};Server={server_name};Database={database_name};Trusted_Connection=yes;')
    
    # connect via sqlalchemy
    con = sa.create_engine(f'mssql://{server_name}/{database_name}?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server', fast_executemany=True)
    
    return conn, con

##***************************************All the setup********************************* UPDATE EMAIL HERE ******************#
##Get the current date and time once for the job instead of calling it on every loop.
maintainer = 'cpalmisano@EMAIL.com'
appName = Path('// PATH /Logging/').stem

# Create logger
logger = logging.getLogger(appName)
logger.setLevel(logging.DEBUG)

# Set up configuration
config = HfDE_config()
SQLProdDB = config['SQL']['Prod']

# Initialize logger
logger = initLogger(appName)

# Connection to server -> SERVER   db -> DATABASE
conn, con = connect_to_sql_server('SERVER', 'DATABASE')
cursor = conn.cursor()
conn.autocommit = True

Session = sessionmaker(bind=con)
session = Session()


# Job setup
print('#------------------ ETL Started-------------------#', today, current_time)
##########################------------------------------------------------------- UPDATE TITLES HERE ---------------------
job_type = 'Wellness Warnings'
job_id = start_sql_job_report(cursor, job_type, job_desc='Monthly Wellness Warnings')
create_job_details_log(cursor, job_id, job_detail_info='Wellness Warnings data upload')

#####################################--------------  Start of Code
#!!!

sql_query = '''
WITH cte_elig AS (    -------------Eligibility----v
    SELECT
        person_id
        ,MAX(STOP_DATE) AS recent_date
    FROM SCHEMA.EligMedical WITH (NOLOCK)
		WHERE GETDATE() BETWEEN START_DATE AND STOP_DATE
		GROUP BY PERSON_ID
),
cte_elig_plan AS (
	SELECT
		ep.person_id
		,CAST(ep.recent_date AS DATE) AS recent_date
		,em.PLAN_DESCRIPTION AS plan_name
	FROM cte_elig ep WITH (NOLOCK)
	  LEFT JOIN SCHEMA.EligMedical em ON em.PERSON_ID = ep.PERSON_ID AND em.STOP_DATE = CAST(ep.recent_date AS DATE)
		WHERE em.PLAN_DESCRIPTION IS NOT NULL
			AND GETDATE() BETWEEN [START_DATE] AND STOP_DATE
),
cte_elgtemp AS (
    SELECT
        DISTINCT es.person_id
		,CAST([start_date] AS DATE) AS [start_date]
        ,CAST([stop_date] AS DATE) AS [stop_date]
        ,ep.plan_name
    FROM SCHEMA.vwEligMedicalSimple es WITH (NOLOCK)
    LEFT JOIN cte_elig_plan ep WITH (NOLOCK) ON ep.person_id = es.person_id
    WHERE es.stop_date > GETDATE()         --------^
),
cte_primary_care AS (
    SELECT HfSpecialtyId
      FROM SCHEMA.HfSpecialtyId
		WHERE AttributionLevel = 1    --- 3, 41, 46, 48, 65, 74, 88, 130, 139
),
cte_primary_care_ids AS (
    SELECT sm.HfSpecialtyId 
			,sm.SpecialtyCode 
	    FROM SCHEMA.specialtymap sm
	 INNER JOIN cte_primary_care id ON id.HfSpecialtyId = sm.HfSpecialtyId
)
SELECT DISTINCT  -- Create the final result
    a.person_id
    ,ServiceStartDate
    ,BillingTaxId
    ,funds_enrollmentid
    ,[Status Reason]
    ,CAST(fp.funds_fein as int) AS fundstaxid
    ,fp.funds_location AS funds_name
    ,funds_dateactive
    ,et.[start_date]
    ,et.[stop_date]
    ,fp.funds_fivestar
    ,a.funds_facility
	,et.plan_name
    ,rl.Last_Visit
    ,rl.Next_Appt
    ,CASE WHEN BillingTaxId IS NOT NULL 
                AND BillingTaxId IN (
    				SELECT DISTINCT fp.funds_fein 
    				FROM [SCHEMA].[funds_providerorganization] fp
    				WHERE fp.funds_fivestar = 1   ---establishes a flag that if the claim was at a fivestar facility
    			) THEN 1
    			ELSE 0
    		END AS  FiveStar_Usage
    ,CAST(NULL AS DATE) AS [date]         -- add date column for tracking
    ,CAST(NULL AS VARCHAR) AS [JobId]      -- add jobid column for tracking                            
 FROM (				    -- Get distinct columns from Claims table
    SELECT DISTINCT 
        CAST(PERSON_ID AS varchar) AS claim_personID
        ,PaidDate
        ,ServiceStartDate
        ,ServiceEndDate
        ,ClaimEntryDate
        ,RenderingName
		,Claimnr
        ,CAST(BillingNPI AS INT) AS BillingNPI
        ,BillingName
        ,CAST(BillingTaxId AS INT) AS BillingTaxId
        ,CAST(GroupNr AS INT) AS GroupNr
        ,CAST(HfSpecialtyId AS INT) AS HfSpecialtyId
    FROM SCHEMA.ClaimsData WITH (NOLOCK)
		WHERE ServiceStartDate >= DATEADD(MONTH, -9, GETDATE())
			AND (PlaceOfService = '11' OR PlaceOfService = '22')
		) b
INNER JOIN (		
    SELECT DISTINCT      -- Get distinct columns from [DATABASE].SCHEMA.funds_enrollment table
        CAST(person_id AS varchar) AS person_id
        ,funds_sourcename
        ,CAST(funds_dateactive AS DATE) AS funds_dateactive
        ,CAST(funds_closedate AS DATE) AS funds_closedate
        ,funds_facilityname
		,funds_facility
        ,funds_enrollmentid
        ,statuscode as [Status Reason]
        ,funds_outreach
        ,CAST(funds_warningdate AS DATE) AS funds_warningdate
        ,funds_warningreason
    FROM SCHEMA.funds_enrollment WITH (NOLOCK)
    WHERE funds_program = 268300000		---wellness
         AND statuscode = 268300305		---Participating 
          AND statecode = 0				   ---Active
		   AND (datediff(day, funds_dateactive, getdate()) > 180) ---check to make sure they have been in the program for AT LEAST 6 months 
			)
			a ON a.person_id = b.claim_personID
 LEFT JOIN 
	(SELECT Subscr_id,
			CAST(Last_Visit AS DATE) AS Last_Visit
			,CAST([Next_Appt] AS DATE) AS Next_Appt
				FROM SCHEMA.Registry_List WITH (NOLOCK) 
	) rl 
		ON rl.Subscr_id = a.person_id
 LEFT JOIN (			
    SELECT DISTINCT   -- Matching funds_fein for Tax ID comparison later
        funds_fein
		,funds_providerorganizationid
        ,funds_fivestar
        ,CASE WHEN funds_parentorganizationname IS NOT NULL 
			THEN funds_parentorganizationname 
				ELSE funds_name 
					END AS funds_location
    FROM [SCHEMA].[funds_providerorganization]
			) fp  
				ON a.funds_facility = fp.funds_providerorganizationid
 inner JOIN cte_primary_care_ids hfid  -- Join with attribution level 1 
	ON hfid.HfSpecialtyId = b.HfSpecialtyId
 INNER JOIN cte_elgtemp et			 -- Join with cte_elgtemp
	ON CAST(et.person_id as varchar) = a.person_id
		WHERE a.Person_ID IS NOT NULL
			and ServiceStartDate >= DATEADD(MONTH, -9, GETDATE())
            order by person_id, servicestartdate desc;
'''

sql = pd.read_sql_query(sql_query, con)    

#########################################!!!

# Replace True with 1 and False with 0
sql['funds_fivestar'] = sql['funds_fivestar'].replace({True: 1, False: 0})

# # Create a flag based on matching 'ein' and the comparison field being 1
# sql['FiveStar_Usage'] = (sql['BillingTaxId'] == sql['fundstaxid']) & (sql['funds_fivestar'] == 1)

# # Replace True with 1 and False with 0 
# sql['FiveStar_Usage'] = sql['FiveStar_Usage'].replace({True: 1, False: 0})

# Add new columns with NULL values 
sql['date'] = pd.to_datetime('NaT')

# Assuming today is a datetime object (e.g., pd.to_datetime('today'))
today = pd.to_datetime('today').date()  # Extract the date part

sql['ServiceStartDate'] = pd.to_datetime(sql['ServiceStartDate'])

# Assuming today is a datetime object (e.g., pd.to_datetime('today'))
today = pd.to_datetime('today').normalize()  # Extract the date part

# Calculate days since ServiceStartDate
sql["DaysSinceServiceStart"] = (today - sql['ServiceStartDate']).dt.days.fillna(0)



@timeit
def process_data(data, temp_wellness=None):
    """
    Processes data to create binary flags based on case statements, replicating SQL logic.

    Args:
        data (pd.DataFrame): Input DataFrame containing necessary columns.
        temp_wellness (pd.DataFrame, optional): Optional temporary DataFrame containing
            additional claims data (if provided). Defaults to None.

    Returns:
        pd.DataFrame: Output DataFrame with binary flags (Warn_Switch, Warn_ActiveNoApt, Warn_NoUsage).
    """

    # Warn_Switch - Mismatched Tax IDs and 5-star usage
    data['Warn_Switch'] = (
        (data['fundstaxid'].notnull()) &
        (data['BillingTaxId'] != data['fundstaxid']) &
        (data['FiveStar_Usage'] == 1)
    ).astype(int)

    # Warn_ActiveNoApt - No start date and no upcoming appointment
    data['Warn_ActiveNoApt'] = (
        (data['ServiceStartDate'].isna()) &
        (data['Next_Appt'].isna())
    ).astype(int)

    # Filter for rows with non-missing fundstaxid
    data = data[data['fundstaxid'].notna()]
    
        # Warn_NoUsage (vectorized)
    today = pd.to_datetime('today')  #time set up for today
    data['Warn_NoUsage'] = (
        (data['Next_Appt'].isna() | (pd.to_datetime(data['Next_Appt']) <= today))
        & (data['DaysSinceServiceStart'] >= 270)  # >= for past 270 days
        & (data['fundstaxid'].notnull()) & (data['BillingTaxId'] != data['fundstaxid'])  # BillingTaxId should not match fundstaxid for no usage
    ).astype(int)
    
        # Usage (vectorized)
    data['Usage'] = (
        (data['DaysSinceServiceStart'] <= 270)  # >= for past 270 days
        & (data['BillingTaxId'] == data['fundstaxid'])  # BillingTaxId should not match fundstaxid for no usage
    ).astype(int)

    # Check for past claims (within conditional loop)
    if temp_wellness is not None and not temp_wellness.empty:
        past_claim_mask = temp_wellness.query(
            'person_id == @data["person_id"] and servicestartdate >= @data["ServiceStartDate"] - pd.Timedelta(days=270) and BillingTaxId == fundstaxid'
        ).groupby('person_id')['person_id'].transform('size') > 0
        data.loc[past_claim_mask, 'Warn_NoUsage'] = 0

    return data


## Create warnings in df
wellness = process_data(sql)

#!!!
df = wellness

##Group by person_id and select the first row, which will be the most recent visit or warning (or None if no visits or warnings exist)
df = df.sort_values(by=['person_id', 'ServiceStartDate'], ascending=False)

### Add 'Action' column with priority actions for warnings
### Some actions are not being used but code is here
df.loc[df['Warn_Switch'] == 1, 'Action'] = 'Switch Five Star'
df.loc[df['Usage'] == 1, 'Action'] = 'Usage'
df.loc[df['Warn_NoUsage'] == 1, 'Action'] = 'No Usage'
#df.loc[df['Warn_ActiveNoApt'] == 1, 'Action'] = 'Active No Apt'
#df.loc[ (df['Warn_NoUsage'] == 1) & (warnings['Warn_ActiveNoApt'] == 1), 'Action'] = 'No Usage'
#df.loc[(df['Warn_Switch'] == 1) & (warnings['Warn_NoUsage'] == 1) & (warnings['Warn_ActiveNoApt'] == 1), 'Action'] = 'Switch Five Star'
df.loc[(df['Warn_Switch'] == 1) & (df['Warn_NoUsage'] == 1), 'Action'] = 'Switch Five Star'

# Filter the dataframe based on the given criteria
filtered_df = df[(df['Action'].notnull())]

# If the action is 'Switch Five Star', filter based on the past 30 days
filtered_df = filtered_df[(filtered_df['Action'] != 'Switch Five Star') | (filtered_df['ServiceStartDate'] >= pd.Timestamp('today') - pd.DateOffset(days=30))]


## No USAGE
# Optimized filtering with negation and chaining
nu_fin = df[(df['Action'] == 'No Usage') & (~df['person_id'].isin(df[df['Action'] == 'Usage']['person_id']))]

## SWTICH
## sort by person_id (descending) and ServiceStartDate (descending)
# and select the most recent entry for each person_id using groupby and head(1)
sw_fin = (df[df['Action'] == 'Switch Five Star']
               .sort_values(by=['person_id', 'ServiceStartDate'], ascending=False)
               .groupby('person_id').head(1))


# Concatenate visits and filtered warnings back into a single DataFrame
warnings = pd.concat([sw_fin, nu_fin])

#!!!
### Set up for SQL & Dynamics 
well = warnings

# Set up new columns for Dynamics integration 
well['source'] = 'Claims'               ###============================
well['program'] = 'Wellness'            ###============================
well['Notes'] = ''                      ###============================
well['funds_source'] = pd.NA
well['funds_action'] = pd.NA
well['funds_casenumber'] = pd.NA
well['funds_dateofservice'] = pd.NA
well['funds_disease'] = pd.NA
well['funds_provider'] = pd.NA
well['funds_facility'] = pd.NA
well['funds_futureMomsRegistrationDate'] = pd.NA
well['funds_name'] = pd.NA
well['funds_person'] = pd.NA
well['funds_program'] = pd.NA
well['funds_rebateeligible'] = pd.NA
well['funds_score'] = pd.NA
well['funds_procedure_multiselect'] = pd.NA
well['funds_32bjissecondaryinsurance'] = pd.NA
well['funds_episodesid'] = pd.NA
well['errormessage'] = pd.NA
well['errorcreatedon'] = pd.NA
well['d365syncedon'] = pd.NA
#well['createdon'] = pd.NA  ##table populated


# Set up for Dynamics push
# Drop unneeded columns
well = well.drop(['ServiceStartDate',  'BillingTaxId', 'fundstaxid', 'funds_dateactive', 'start_date',
'stop_date', 'plan_name', 'Last_Visit', 'Next_Appt',   'FiveStar_Usage', 'Warn_Switch', 'Warn_ActiveNoApt', 'Warn_NoUsage'], axis=1)        
        
# Rename for proper field name
well = well.rename(columns={'funds_enrollmentid':'funds_engagement'})
        

# Using boolean indexing with negation 
def is_not_empty(value):
    return not pd.isna(value) and value != ''  # Check for null, NaN, and empty string

# Filter only values with an Action 
wellness_fin = well[well['Action'].apply(is_not_empty)]


############################## Dupe Check ##################################

#### Compare new episodes with history to not provide dupes
welleps = pd.read_sql("""
select person_id, program, funds_engagement, funds_episodesid, d365syncedon 
from SCHEMA.src_funds_episodes
                       """, con)
                       
# change to object for ease of use
welleps['person_id'] = welleps['person_id'].astype(object)

# Perform an inner join with indicator to mark matching rows
dup_eps = welleps.merge(wellness_fin, on=['person_id', 'funds_engagement'], how='inner', indicator=True)

# index fixing
dup_eps = dup_eps.reset_index(drop=True)
wellness_fin = wellness_fin.reset_index(drop=True)


import win32com.client as win32
import datetime

def send_episode_email(episode_found):
    """Sends an email notification based on episode presence.

    Args:
        episode_found (bool): True if episodes were found, False otherwise.
    """

    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.To = 'cpalmisano@EMAIL.com'  
    mail.Subject = 'Wellness Episodes Update'

    if episode_found:
        mail.Body = "Episodes found in dup_eps DataFrame."
    else:
        mail.Body = "No episodes found in dup_eps DataFrame."

    # mail.Send()  # Uncomment to send the email
    print(f"Email sent: {episode_found}")  # Print status for now


if 'episodes' in dup_eps.columns:
    # process episodes
    try:
        drop_indices = dup_eps[dup_eps['Action'] == 'No Usage']['funds_engagement']
        wellness_fin.drop(wellness_fin[wellness_fin['funds_engagement'].isin(drop_indices)].index, inplace=True)
        send_episode_email(True)  # Send email if episodes found
    except Exception as e:
        print("An error occurred processing episodes:", e)
        send_episode_email(False)  # Send email indicating error

else:
    print("No episodes found in dup_eps DataFrame.")
    send_episode_email(False)  # Send email indicating no episodes
        

#=======================  HISTORY ===================================#!!!

# Set up for wellness History table    
well_his = filtered_df

# Set up columns
well_his['Program'] = 'Wellness'
well_his['Source'] = 'Claims'

# Rename column
well_his = well_his.rename(columns={'funds_enrollmentid':'Engagement'})

# Drop unneeded columns
well_his = well_his.drop(['ServiceStartDate',
       'BillingTaxId', 'fundstaxid', 'funds_dateactive',
       'start_date', 'stop_date', 'plan_name', 'Last_Visit', 'Next_Appt',
       'date',  'FiveStar_Usage', 'Warn_Switch', 'Warn_ActiveNoApt',
       'Warn_NoUsage'], axis=1)        
        
# Filter only values with an Action 
well_history = well_his[well_his['Action'].apply(is_not_empty)]

well_history = well_history[well_history['Action'] != 'Usage']


#############################################!!!
##Clean up and touch up

#Add updated date
from datetime import datetime
today = datetime.today().date()
well_history['DateCreated'] = pd.to_datetime(today)


wellness_fin = wellness_fin.drop(['DaysSinceServiceStart', 'date', 'Status Reason', 'Usage', 'funds_fivestar', 'JobId'
                                      ], axis=1)

wellness_fin = wellness_fin.rename(columns={'Notes':'funds_note', 'Action':'action', 'funds_futureMomsRegistrationDate':'funds_futuremomsregistrationdate'})
                 

#############################################!!!

                
#!!!
### Insert data into SQL Server
print('Wellness Warnings Complete')  

# code to upload dataframe to SQL Server
try:
    conn.commit()    
    well_history.to_sql( 'Episode_history', con, schema='SCHEMA', index=False, chunksize=1000, if_exists='append')   
    print("History uploaded successfully!")
    
except Exception as e:
    print("Error uploading History:", e)
    
finally:
    print("Wellness Warnings History Loaded")
    
try:
    conn.commit()    
    wellness_fin.to_sql( 'src_funds_episodes', con, schema='SCHEMA', index=False, chunksize=1000, if_exists='append')   
    ##well_test.to_sql( 'src_funds_episodes', con, schema='d365', index=False, chunksize=1000, if_exists='append')   
    print("Wellness Warnings uploaded successfully!")
    
except Exception as e:
    print("Error uploading Warnings:", e)

finally:
    print("Wellness Warnings Loaded")    

##close all connections
try:
    close_all_connections()
    session.close()

except:
    print('session close failed')

finally:
    print('session closed')
    
close_out_job_log(cursor, job_id)

