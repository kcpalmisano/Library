# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 09:24:06 2022
@author: cpalmisano
"""

import pandas as pd
import numpy as np
import datetime
from datetime import date
import sqlalchemy
from sqlalchemy.orm import sessionmaker 


#Load Python Packages
import sqlalchemy as sa
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


sys.path.insert(1, '//PATH/Logging/') 

current_time = datetime.datetime.now()
start_time = time.monotonic()
today = date.today()

###  ------------------------------- Wrappers -------------------------------------
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

##################################################### ************************************UPDATE THIS FUNCTION *********************************************

def start_sql_job_report(cursor, job_type:str='SCHEMA.Provider792', source:str='792Provider', test_job:bool=False, job_desc:str='Weekly'):
    """
        Inserts into the main.jobs table a new job for the current process. Returns the job id of the newly created job
        NOTE: the initial start time will be set to the current time, but in order to have the job close out properly, then you need to 
        call the .close_out_job_log() method once your process has finished running. This will add an end time for the process, which is how we determine
        whether a job finished running or not
        
        Parameters
        ----------
            cursor: pyodbc.cursor
                Cursor object that connects to database to execute queries
            job_type:str
                name of job which will be added ot the job_type column in main.jobs
            source_folder: str 
                String describing the source for the job if relevant. Gets added to the job description. Empty by default
            test_job:bool
                Inserts True/false into the isTest column if this is a test run that is not writing to the "prod" schema. Defaults to False.
            job_desc:str
                This is actually more of a field describind the frequency of the job (i.e. weekly process, daily). Defaults to Ad Hoc currently

            
        @params optional: the job type defaulted to denote an invoice and the job description, defaulted to weekly
        Returns
        ---------
            job_id:int    
                the job id of the newly initiated job
    """
    sql = """SET NOCOUNT ON;
              INSERT INTO Jobs
              (JobType, JobStart, JobDescription, IsTest) 
              VALUES
              ('{0}','{1}', '{2}', '{3}');
              SELECT scope_identity();
          """.format(job_type, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), source +'_'+job_desc, test_job)
          
    job_id = int(cursor.execute(sql).fetchval())
    return job_id


@timeit
class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    # We'd like stdout and stderr to go to the log file we're rotating.
    # https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass
    
# *********** Data Eng Config  ************
@timeit
def HfDE_config():
    config = configparser.ConfigParser()
    globalConfigPath = Path('//PATH/data-eng-config.ini')
    localConfigPath = Path(os.getcwd()) / 'jobMonitorConfig.ini'
    configCandidates = [globalConfigPath, localConfigPath]
    config.read(configCandidates)
    return config


def getNextScheduledDate(row):
    '''
    Determine what the next date should be, math or calendar dending on frequency in Main.jobMonitor     
    '''
    
    #frequency = row.frequency
    prevDate = row.lastRunStart
    
    if row.frequency.lower() == 'daily':
        return prevDate + timedelta(days=1)

    elif row.frequency.lower() == 'weekday':
        if prevDate.weekday() == 6 or prevDate.weekday() < 4:   
            return prevDate + timedelta(days=1)
        elif prevDate.weekday() == 4:
            return prevDate + timedelta(days=3)
        elif prevDate.weekday() == 5:
            return prevDate + timedelta(days=2)        

    elif row.frequency.lower() == 'weekly':
        return prevDate + timedelta(days=7)

    elif row.frequency.lower() == 'monthly':
        if prevDate.month == 12:
            nextMonth = 1
            nextYear = prevDate.year + 1
        else:
            nextMonth = prevDate.month + 1
            nextYear = prevDate.year

        nextDate = str(nextYear) + '-' + str(nextMonth) + '-' + str(prevDate.day)
        return datetime.strptime(nextDate, '%Y-%m-%d')

    elif row.frequency.lower() == 'quarterly':
        if prevDate.month > 9:
            nextMonth = prevDate.month - 8
            nextYear = prevDate.year + 1
        else:
            nextMonth = prevDate.month + 3
            nextYear = prevDate.year

        nextDate = str(nextYear) + '-' + str(nextMonth) + '-' + str(prevDate.day)
        return datetime.strptime(nextDate, '%Y-%m-%d')  

    elif row.frequency.lower() == 'annual':
        return prevDate + timedelta(days=365)    

    else:
        return None


##################################################### ******************************UPDATE THIS FUNCTION *********************************************
def notify(row, config):
    '''   
    Report results of job monitoring to the people in Main.jobMonitor.notificationGroup.  
    '''
    #logging.info('Sending notifications')
    notificationSent = False
    msg = """From: From Person <EMAIL@GMAIL.com>
    To: To Person <EMAIL@GMAIL.com>
    MIME-Version: 1.0
    Content-type: text/html
    Subject: 792 File

    792 Provider File had an issue 

    <b>This is HTML message.</b>
    <h1>This is headline.</h1>
    """
    
    notificationBody = ''
    jobDescriptor = row.jobType + '-' + row.fileType
    reportHasError = row.hasError and not row.ignoreHasError
    reportMissedJob = row.missedJobFlag and not row.ignoreMissedJob
    reportNoJobEnd = row.noJobEndFlag and not row.ignoreNoJobEnd
    
    if reportHasError is True:
        newMessage = '{0}: has reported an error. In job ID {1}\r\n'.format(jobDescriptor, row.lastJob)
        notificationBody = notificationBody + newMessage
    
    if reportMissedJob is True:
        newMessage = '{0}: did not run on schedule. It was expected to run at {1}\r\n' \
            .format(jobDescriptor, datetime.strftime(row.nextJobStart, '%Y-%m-%d %H:%M:%S') )
        notificationBody = notificationBody + newMessage
    
    if reportNoJobEnd is True:
        newMessage = '{0}: has not completed. It started at {1} and was expected to complete within {2} minutes\r\n' \
            .format(jobDescriptor, datetime.strftime(row.lastRunStart, '%Y-%m-%d %H:%M:%S'), row.runGracePeriodMinutes)
        notificationBody = notificationBody + newMessage

    #smtplib does not support logging, all information is written out to stdout & stderr, see .out file if available.
    msg = MIMEMultipart()
    msg['From'] = config['SMTP']['Sender']
    msg['To'] = row['notificationGroup']
    msg['Subject'] = 'Test of Notifications: ' + row['jobType'] + '-' + row['fileType']
    msg.attach(MIMEText(notificationBody))
    logger.debug(msg)
    
    if notificationBody != '':
        logger.warn('Notification {0} sent to {1}'.format(notificationBody, row.notificationGroup))
        try:
            with SMTP(config['SMTP']['MailServer'], port=25) as smtp:
                smtp.set_debuglevel(2)
                #print(smtp.noop())
                notificationSent = True
        except Exception as error:
            print('SMTP Error! {0}'.format(error))
            notificationSent = False
    else:
        logger.info('No notification for {0}. Exclusions: {1}'.format(
            row.jobType, row.ignoreHasError and row.ignoreMissedJob and row.ignoreNoJobEnd))
    return notificationSent

logger_initialized = True  # Variable to track if the logger is already initialized ----------------------------------


def initLogger(appName):
    global logger_initialized

    if logger_initialized:
        return logging.getLogger(appName)  # Return the existing logger

    path_exception = None

    # Try to set log file path and fall back if you can't
    try:
        logFileName = Path('//PATH/Logging/' + os.path.basename(__file__)).with_suffix('.log')
    except Exception as error:
        logFileName = 'jobmonitor_app_path_exception.log'
        path_exception = error

    # Create log handlers
    logFileHandler = RotatingFileHandler(
        filename=logFileName,
        maxBytes=int(config['LOG']['fileSizeBytes']),
        backupCount=int(config['LOG']['fileCount']),
        encoding=config['LOG']['encoding']
    )

    # Set log formatter from config
    logFormat = config['LOG']['format'].format(
        levelname='%(levelname)s',
        name='%(name)s',
        asctime='%(asctime)s',
        message='%(message)s'
    )

    # Create application logger for our use. Set level then attach formatter and log file handler.
    logger = logging.getLogger(appName)
    logger.setLevel(int(config['LOG']['level']))
    logFileHandler.setFormatter(logging.Formatter(logFormat))

    # Exclude specific log messages from the logger
    excluded_logs = [
        'Row %r',
        'Message: %r',
        'Arguments: %r',
    ]
    for log in excluded_logs:
        logger.addFilter(lambda record, log=log: log not in record.getMessage())

    # Add the log file handler to the existing handlers of the logger
    logger.addHandler(logFileHandler)

    logger.info('**************Begin logging Job Monitoring*********************')

    # Send stdout, stderr to our existing logger with appropriate log level.
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    logger_initialized = True  # Set the flag to indicate that the logger is initialized

    if path_exception is not None:
        logger.error('Error finding logging path: ' + str(path_exception))

    return logger



def create_job_details_log(cursor, job_id:int, job_detail_info:str, test:bool = False):
    """
        Creates a jobdetails log with the job id and the description including the job type (summary, load file, summary details) and the filename
        
        Parameters:
        ----------
        cursor: pyodbc.cursor
            Cursor object that connects to database to execute queries
        job_id:int
        
        ##TO BE ADDED LATER (?)
        row_count_nr:int
            the row count number that can be passed in. It defaults to -1 if this is a field that was not passed in. 
        
        
        Returns
        --------
        1 if successful
        0 plus a print message if there was a failure
    """
    if test:
        test_prefix = 'TEST_'
    else:
        test_prefix = ''
    try:
        sql = """SET NOCOUNT ON;
                INSERT INTO JobDetails
                (jobId,JobDetailInfo)
                VALUES
                ({0},'{1}');
                """.format(job_id, test_prefix+job_detail_info)
        cursor.execute(sql)
    
        # print(str(job_id) + ": " + filename + " summary details created")
        return 1
    except Exception as e:
        print('Error occurred in creating job details: ' + str(e))
        return 0
    
   
def close_out_job_log(cursor, job_id:int):
    """ 
        Will close out a job by updating job end to the current time
        
        Parameters:
        -----------
        cursor: pyodbc.cursor
            Cursor object that connects to database to execute queries
        job_id:int 
            job id of the job that is being closed out
    """
    sql = """SET NOCOUNT ON;
            UPDATE Jobs
            SET JobEnd = '{0}'
            WHERE JobId = {1};
        """.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), job_id)
    cursor.execute(sql)
    
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
##***************************

####Connection 
@timeit
def connect_to_sql_server(server_name, database_name):
    # connect via pyodbc
    conn = pyodbc.connect(f'Driver={{SQL Server}};Server={server_name};Database={database_name};Trusted_Connection=yes;')
    
    # connect via sqlalchemy
    con = sa.create_engine(f'mssql://{server_name}/{database_name}?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server', fast_executemany=True)
    
    return conn, con


################################################################# --------------------- Script Specific Functions 

#Set Date domains to make files look neat & pretty
today = datetime.date.today()
dateset = today
mon = str(dateset.month).rjust(2,'0')
day = str(dateset.day).rjust(2,'0')
yr = str(dateset.year)[0:]


def execute_stored_procedure(conn_str, sp_name):
    try:
        con = pyodbc.connect(conn_str)
        cursor = con.cursor()
        cursor.execute(f"EXECUTE {sp_name}")
        con.commit()
        return True
    except:
        return False
    finally:
        if 'con' in locals() and con is not None:
            print('Can now close')

def main():
    # Connect to SQL Server
    conn_str = 'Driver={SQL Server};' \
               'Server=SERVER;' \
               'Database=DATABASE;' \
               'Trusted_Connection=yes;'
    
    # Execute stored procedure
    sp_name = 'SCHEMA.sp792_ProviderFlag'
    success = execute_stored_procedure(conn_str, sp_name)
    if success:
        print('Stored procedure executed successfully!')
    else:
        print('Stored procedure failed to execute.')

    
#***************************All the setup**************************#
# Get the current date and time once for the job instead of calling it on every loop.
maintainer = 'EMAIL@GMAIL.com'
appName = Path('//PATH/Logging/').stem

# Create logger
logger = logging.getLogger(appName)
logger.setLevel(logging.DEBUG)

# Set up configuration
config = HfDE_config()
SQLProdDB = config['SQL']['Prod']

# Initialize logger
logger = initLogger(appName)


#Connect to DB via SQL Server
conn = pyodbc.connect('Driver={SQL Server};'   
                      'Server=SERVER;'   ##SERVER NAME
                      'Database=DATABASE;'       ##DATABASE
                      'Trusted_Connection=yes;')
cursor = conn.cursor()
conn.autocommit = True


###Connect to SQL Server
con = sqlalchemy.create_engine('mssql://SERVER/DATABASE?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server', fast_executemany=True)    
Session = sessionmaker(bind=con)
session = Session()   


# Job setup
print('#------------------ ETL Started-------------------#', today, current_time)
################################################################# --------------------- UPDATE THESE 
job_type = '792 Weekly Provider File'
job_id = start_sql_job_report(cursor, job_type, job_desc='Weekly 792')
create_job_details_log(cursor, job_id, job_detail_info='792 ETL Run')

#######################-----------------------------------------  Start of Code

#!!!
#---------------------
###Old file load for On-Ramp
# df = pd.read_csv('//PATH//792_Flag//32BJ_Flag792_20230710.csv')

# ##### TEST FILE #####
# df = pd.read_csv('//PATH/Test_32BJ_Flag792.txt', sep=';' , index_col=False)
#---------------------

##Loading the weekly .txt file
df = pd.read_csv('//PATH/32BJ_Flag792.txt' , sep=';' , header=0, index_col=False)


#Move raw file to folder
try: 
    df.to_csv(r'\\PATH\RAW_32BJ_Flag792_'+yr+mon+day+'.csv', index=False)
    print('written WithOUT hardcode')
    
except OSError:
    df.to_csv(r'S:\PATH\32BJ_Flag792_'+yr+mon+day+'.csv', index=False)
    print('written WITH hardcode')

#-------------------------------  Data clean up 

# # ##Create the Location column from the EPIN 
df['LOCATION'] = df['PROV-ID'].str[-1]

df['PROV-ID'] = df['PROV-ID'].str[:-1]


# ###add a date column that is empty
df['Date_added'] = np.nan

df['Date_added'] = pd.to_datetime(df['Date_added'])   #convert to proper type
 

##### FlagID creation 
#change to string for ID creation
df[['NPI','TAX']] = df[['NPI', 'TAX']].astype(str)

#remove (.0) from NPI and TAX columns
df['NPI'] = df['NPI'].str[:-2]
df['TAX'] = df['TAX'].str[:-2]

#---------------------------
###create an ID column
col = ['NPI', 'TAX', 'PROV-ID', 'LOCATION']    #PROV-ID & LOCATION = EPIN
df['Flag792_ID'] = df[col].apply(lambda row: ''.join(row.values.astype(str)), axis=1)

#remove periods from ID column 
df['Flag792_ID'] = df['Flag792_ID'].str.replace('.', '')


###======================================================= New Empire BS fix
# load in crosswalk code table 
df792 = pd.read_sql('''
                    SELECT  * FROM SCHEMA.SpecialtyMap
                    ''', con)

###====== Fix for SPS specialty codes to be HFSpeciatlyID  Part 1 

# Select distinct SpecialtyCodes and HfSpecialtyIds from specialty_map
sm1 = df792[df792['FileType'] == 'SPS'][['SpecialtyCode', 'HfSpecialtyId']].copy()
sm1.rename(columns={'HfSpecialtyId1': 'HfSpecialtyId1'}, inplace=True)
# Merge with data based on SPEC1CODE
result = df.merge(sm1, left_on='SPEC1CODE', right_on='SpecialtyCode', how='left')

# Repeat for other SPEC columns
sm2 = df792[df792['FileType'] == 'SPS'][['SpecialtyCode', 'HfSpecialtyId']].copy()
sm2.rename(columns={'HfSpecialtyId2': 'HfSpecialtyId2'}, inplace=True)
result = result.merge(sm2, left_on='SPEC2CODE', right_on='SpecialtyCode', how='left')

##Drop columns
columns_to_drop = ['SPEC1CODE', 'SPEC2CODE',  'Unnamed: 23', 'SpecialtyCode_x','SpecialtyCode_y']
result = result.drop(columns_to_drop, axis=1)

#rename columns
result = result.rename(columns={'HfSpecialtyId_x':'SPEC1CODE' , 'HfSpecialtyId_y':'SPEC2CODE'})

####===== Part 2


sm3 = df792[df792['FileType'] == 'SPS'][['SpecialtyCode', 'HfSpecialtyId']].copy()
sm3.rename(columns={'HfSpecialtyId3': 'HfSpecialtyId3'}, inplace=True)
result = result.merge(sm2, left_on='SPEC3CODE', right_on='SpecialtyCode', how='left')

sm4 = df792[df792['FileType'] == 'SPS'][['SpecialtyCode', 'HfSpecialtyId']].copy()
sm4.rename(columns={'HfSpecialtyId4': 'HfSpecialtyId4'}, inplace=True)
result = result.merge(sm2, left_on='SPEC4CODE', right_on='SpecialtyCode', how='left')

##Drop columns
columns_to_drop = ['SPEC3CODE', 'SPEC4CODE', 'SpecialtyCode_x', 'SpecialtyCode_y']
result = result.drop(columns_to_drop, axis=1)

#rename columns
result_df = result.rename(columns={'HfSpecialtyId_x':'SPEC3CODE' , 'HfSpecialtyId_y':'SPEC4CODE', 'RLTDPADRSKEY':'PADRSKEY'})


# Create a new DataFrame with selected columns
final_result = result_df[['Flag792_ID', 'PROV-ID', 'NPI', 'TAX', 'LASTNAME', 'FIRSTNAME', 'MI', 'ORG NAME',
       'ADDRESS1', 'ADDRESS2', 'CITY', 'STATE', 'ZIP', 'ZIP4', 'PHONE',
       'DEGREE', 'GENDER', 'FAX', 'SPEC1CODE', 'SPEC2CODE', 'SPEC3CODE',
       'SPEC4CODE', 'PADRSKEY', 'POAKEY', 'LOCATION', 'Date_added']].copy()

######################################################################################################

#reorder columns to better suit database 
df_out = result_df[['Flag792_ID', 'Date_added', 'PROV-ID', 'LOCATION', 'NPI', 'TAX', 'LASTNAME', 'FIRSTNAME', 'MI','ORG NAME', 'ADDRESS1', 'ADDRESS2',
          'CITY', 'STATE', 'ZIP', 'ZIP4', 'PHONE', 'DEGREE', 'GENDER', 'FAX', 'SPEC1CODE', 'SPEC2CODE', 'SPEC3CODE', 'SPEC4CODE', 'POAKEY', 'PADRSKEY']]

 
### Add empty date columns
df_out['ModifiedDate'] = np.nan
df_out['ModifiedDate'] = pd.to_datetime(df_out['ModifiedDate'])   #convert to proper type

df_out['DateRemoved'] = np.nan
df_out['DateRemoved'] = pd.to_datetime(df_out['DateRemoved'])   #convert to proper type

df_out['Source_type'] = 'SPS'


columns_to_convert = ['LOCATION', 'TAX', 'ZIP']
for col in columns_to_convert:
    df_out[col] = pd.to_numeric(df_out[col], errors='coerce').fillna(0).astype(int)


df_out = df_out.drop_duplicates()

#---------------------------

### Insert data into SQL Server
print('Loading new 792 Table data to SQL archive table')  

conn.commit()                 #code to upload dataframe to SQL Server
df_out.to_sql( 'TABLE', con, schema='SCHEMA', index=False, chunksize=1000, if_exists='replace')   

print("New data loaded into TABLE Archive") 

#-------------------------------------------------------------
####Move file to tracking folder

try: 
    df_out.to_csv(r'\\PATH\792_Flag\32BJ_Flag792_'+yr+mon+day+'.csv', index=False)
    print('written WithOUT hardcode')
    
except OSError:
    df_out.to_csv(r'S:\PATH\792_Flag\32BJ_Flag792_'+yr+mon+day+'.csv', index=False)
    print('written WITH hardcode')

#----------------------------------------
#### Run stored procedure and close all connections
      

try: 
    if __name__ == '__main__':
        main()
except:
    print('SP Ran')       
   # notify(row, config)
    
# finally:    
#     close_all_connections()
    

#######################################################################_------------------------------------------------  End of Code
end_time = time.monotonic()

close_out_job_log(cursor, job_id)
print('#------------------ETL Completed-------------------#', today, 'Time Elapsed', timedelta(seconds=end_time - start_time))
close_all_connections()
session.close()
+
