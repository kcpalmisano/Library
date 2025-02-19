
#basic load
import pandas as pd
import numpy as np
from datetime import datetime, date
import seaborn as sns
import matplotlib.pyplot as plt 

import sqlalchemy
from sqlalchemy.orm import sessionmaker 


#####Timing START
import time
from datetime import timedelta
current_time = datetime.now()
start_time = time.monotonic()
print(current_time)


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


###Connect to SQL Server
con = sqlalchemy.create_engine('mssql://SERVER/DATABASE?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server',
                                fast_executemany=True)    

Session = sessionmaker(bind=con)
session = Session()    

#Check for database connection
if (con == False):
    print("Connection to DATABASE Error")
else: 
    print("Connection to DATABASE Successful")

####################
#Data selection query. Structure the WHERE clause to best fit needs of data 

#####Timing 
end_time = time.monotonic()
print("Starting Data pull from SQL",timedelta(seconds=end_time - start_time), current_time)

data = pd.read_sql_query('''
                         select cu.PERSON_ID,
                         PatientDOB as DOB,
                         PatientGender,
                         ServiceStartDate,
                         HCPCS,
                         RevenueCode as revcode,
                         DiagnosisCodePrinciple as  DxCodeP,
                         DiagnosisCode1 as DxCode1,
                         HfClaimId,
                         rec.tjr as TJR,
                         PlaceOfService as PoS
                             from TABLE1 cu  
					left join TABLE2 rec
					on cu.person_id = rec.person_id
                          ''', con)

###############################
### General Clean up

data['DOB'] = pd.to_datetime(data['DOB'])  #change to datetime field as some are not 

##Clean up DOB to age
def age(birthdate):
    today = date.today()
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    return age

#apply the function 
data['DOB'] = data['DOB'].apply(age)

#Rename the field
data = data.rename({'DOB':'Patient_Age'}, axis=1)


#####Timing END
end_time = time.monotonic()
print("age complete", timedelta(seconds=end_time - start_time))


#----------------------------------------------------------------
### Categorical to numeric
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

### Label encoding for categorical columns 
def label_encode_columns(df, columns):
    """
    Purpose:
        Label encode a set of columns 
    
    Args:
        df (pandas DataFrame): The input DataFrame
        cols (list of str): The list of column names to label encode

    Returns:
        The transformed DataFrame with the specified columns label encoded (Male / Female == 1 / 2)
    """
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

## categorical columns
cols = data[['HCPCS', 'revcode', 'DxCodeP', 'DxCode1', 'PoS', 'PatientGender']]

### label encoding
label_encode_columns(data, cols)

## convert to datetime
data['ServiceStartDate'] = pd.to_datetime(data['ServiceStartDate'], errors='coerce')

data['StartDay_year'] = data['ServiceStartDate'].dt.year
data['StartDay_month'] = data['ServiceStartDate'].dt.month
data['StartDay_week'] = data['ServiceStartDate'].dt.isocalendar().week

# data_id = data_done[['PERSON_ID', 'HfClaimId', 'TJR']]


#drop uneeded columns
results = data.drop(columns=[ 'PERSON_ID', 'ServiceStartDate', 'HfClaimId'])

# Replace True with 1 and False with 0  ### SQL PULL ONLY ###
data['TJR'] = data['TJR'].replace({True: 1, None: 0})

results = results.replace(np.nan,0)

# results.columns
# results.dtypes
# results.nunique()

#CHANGE non int columns to int columns
results = results.astype({'TJR':'int', 'DxCode1':'int', 'PatientGender':'int', 'StartDay_week':'int', 'HCPCS':'int'
                          , 'revcode':'int', 'DxCodeP':'int', 'PoS':'int'})


#####Timing END
end_time = time.monotonic()
print("data manipulation complete", timedelta(seconds=end_time - start_time))

######################## SET UP THE PICKLE ###############################
import pickle

# load the model from the file
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)


# use the model to make predictions on the new data
y_pred = model.predict(X)

actual = results[['TJR']]

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(actual,y_pred)
print('Accuracy is', accuracy)

############################################  See predicted with actual and identifier

#change predictions into a dataframe
y_preddf = pd.DataFrame(y_pred)  

#rename the end prediction column
y_preddf = y_preddf.rename(columns={0: 'Predicted_TJR'})

#put predicted output with actual and identifier 
final = pd.concat([results,y_preddf], axis=1) 

fin = final.head(1000)

print(fin)

#############################################

#close all connections
try:
    close_all_connections()
    session.close()

except:
    print('session close failed')

finally:
    print('session closed')


#####Timing END
end_time = time.monotonic()
print("Model Run in ", timedelta(seconds=end_time - start_time))
