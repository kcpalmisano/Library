
import pandas as pd
import numpy as np
import datetime
from datetime import date
from sqlalchemy.orm import sessionmaker 
import sqlalchemy as sa
import pyodbc
import gc
import inspect


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
def connect_to_sql_server(server_name, database_name):
    # connect via pyodbc
    conn = pyodbc.connect(f'Driver={{SQL Server}};Server={server_name};Database={database_name};Trusted_Connection=yes;')
    
    # connect via sqlalchemy
    con = sa.create_engine(f'mssql://{server_name}/{database_name}?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server', fast_executemany=True)
    
    return conn, con


#Connect to DB via SQL Server
conn = pyodbc.connect('Driver={SQL Server};'   
                      'Server=SERVER;'   ##SERVER NAME
                      'Database=DATABSE;'       ##DATABASE
                      'Trusted_Connection=yes;')
cursor = conn.cursor()
conn.autocommit = True


###Connect to SQL Server
con = sa.create_engine('mssql://SERVER/DATABASE?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server', fast_executemany=True)    
Session = sessionmaker(bind=con)
session = Session()   

##Loading the file  
df = pd.read_excel(r'C:/PATH/FILE.xlsx')

### Insert data into SQL Server
print('Loading new  table')  

conn.commit()                 #code to upload dataframe to SQL Server
df.to_sql( 'FILE_NAME', con, schema='Temp', index=False, chunksize=1000, if_exists='replace')   

print("New data loaded") 

close_all_connections()

