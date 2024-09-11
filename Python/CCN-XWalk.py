################ INFO ##############
# Tin to CCN to Hosp System crosswalk
# tin to ccn
# https://www.dolthub.com/repositories/dolthub/standard-charge-files/query/main?active=Tables&q=SELECT+*%0AFROM+%60hospitals%60%0Awhere+enrollment_state+%3D+%27NY%27%0A
# ccn to syst
# https://www.ahrq.gov/chsp/data-resources/compendium-2022.html
 

# hospital linkage file (excel) 
####################################

import requests
import pandas as pd

############################## EVENTUAL API PULL ##############################
# # Define the API endpoint and the SQL query
# url = "https://www.dolthub.com/api/v1/query"

# query = "SELECT * FROM `hospitals` where enrollment_state IN ('NY', 'NJ', 'PA') "
  
# # Make a POST request to the API endpoint
# response = requests.post(url, json={"query": query})
# # Convert the response to JSON

###  API check ###
# print(response.status_code)
# print(response.text)
##################

# data = response.json()

### Extract the rows from the response
# rows = data['rows']

### Create a DataFrame from the rows
# df = pd.DataFrame(rows)

###############################################################################

import gc
import inspect
import pyodbc
import sqlalchemy

def close_all_connections():
    """
    This function will close all open connections
    """
    for obj in gc.get_objects():
        if inspect.ismodule(obj):
            for name in dir(obj):
                if name == "engine":
                    engine = getattr(obj, name)
                    if isinstance(engine, sqlalchemy.engine.Engine):
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
def connect_to_sql_server(server_name, database_name):
    # connect via pyodbc
    conn = pyodbc.connect(f'Driver={{SQL Server}};Server={server_name};Database={database_name};Trusted_Connection=yes;')
    
    # connect via sqlalchemy
    con = sqlalchemy.create_engine(f'mssql://{server_name}/{database_name}?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server', fast_executemany=True)
    
    return conn, con


###connection to server ->    db -> 
conn, con = connect_to_sql_server('SERVER', 'DATABASE')
cursor = conn.cursor()
conn.autocommit = True 


###DOLTHUB TABLE TO CSV FILE 
path = r" *PATH* /Data/TIN_CCN.csv"

data = pd.read_csv(path)

###pull CCN to Syst from AHRQ 2022 file
url = 'https://www.ahrq.gov/PATH/wysiwyg/chsp/compendium/chsp-hospital-linkage-2022-rev.xlsx'

ccn = pd.read_excel(url)

## merge data
merge = ccn.merge(data, how='left')

##Pull relevant columns only
provider = merge[['compendium_hospital_id', 'ccn', 'hospital_name', 'hospital_street',
       'hospital_city', 'hospital_state', 'hospital_zip', 'tin', 'health_sys_id', 'health_sys_name', 'npi', 'organization_name' ]]

###Testing table for output
sql = pd.read_sql('''select distinct * from dbo.TIER_TAX_ID_SYSTEM''', con) 

###merge table 
merge_df = provider.merge(sql, left_on='tin', right_on='prov_tax_id', how='left')

###################################################
#merge_df['tin'].count()  #357

#merge_df[merge_df['hospital_state'] =='PA']   #NY = 232, NJ = 106 , PA = 259
###################################################

try:    ## Table load
    conn.commit()    
    #merge_df.to_sql( 'CCN_Xwalk', con, schema='Temp', index=False, chunksize=1000, if_exists='append')   
    print("CCN Xwalk uploaded successfully!")
  
except Exception as e:
    print("Error uploading CCN Xwalk:", e)

finally:
    print("CCN to TIN data Loaded")  

      
##close all connections
try:
    close_all_connections()

except:
    print('session close failed')

finally:
    print('session closed') 
