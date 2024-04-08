# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 09:39:01 2023

@author: cpalmisano
"""


#basic load
import pandas as pd
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt 
import pyodbc
import sqlalchemy
from sqlalchemy.orm import sessionmaker 


#ML load
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from sklearn import metrics

def buildROC(target_test,test_preds):
    fpr, tpr, threshold = metrics.roc_curve(target_test, test_preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

#####Timing START
import time
from datetime import timedelta
start_time = time.monotonic()

# #####################

#Connect to DB via SQL Server
conn = pyodbc.connect('Driver={SQL Server};'   
                      'Server=SERVER;'   ##SERVER NAME
                      'Database=DATABASE;'       ##DATABASE
                      'Trusted_Connection=yes;')
cursor = conn.cursor()

conn.autocommit = True


###Connect to SQL Server
con = sqlalchemy.create_engine('mssql://SERVER/DATABASE?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server',
                                fast_executemany=True)    

Session = sessionmaker(bind=con)
session = Session()    

#Check for database connection
if (con == False):
    print("Connection to Database Error")
else: 
    print("Connection to Database Successful")

####################
#Data selection query. Structure the WHERE clause to best fit needs of data 

#####Timing 
end_time = time.monotonic()
print("Starting Data pull from SQL",timedelta(seconds=end_time - start_time))

data1 = pd.read_sql_query('''
          SELECT top 1000 cft.PERSON_ID,
       PatientDOB AS DOB,
       PatientGender,
       ServiceStartDate,
       HCPCS,
       RevenueCode,
       DiagnosisCodePrincipal,
       DiagnosisCode1,
       HfClaimId,
       --osteo_hip_dummy,
       --osteo_knee_dummy,
       --esrd_dummy,
       CASE WHEN tjr.person_id IS NOT NULL THEN 1 ELSE 0 END AS TJR,
       RevenueCode AS revcode,
       DiagnosisCodePrincipal AS DxCodeP,
       DiagnosisCode1 AS DxCode1,
       BillingTaxId AS BTaxId,
       PlaceOfService AS PoS
FROM ClaimsFlat_test cft
LEFT JOIN TJR.Recruitment tjr ON cft.person_id = tjr.person_id
WHERE YEAR(ServiceStartDate) >= 2019;
                          ''', con)


###############################

#read in data
# data1 = pd.read_excel(r'C:\PATH\Data\TJR_data.xlsx')

#data1 = pd.read_excel(r'C:\PATH\tjr_justdata.xlsx')
####817715 rows

data = data1


df = data.head(25)

#-----------------------------------------------------------
#set up target dataframe
target_name = "TJR"
target_id = data[['HfClaimId', 'TJR']]    
target = data[['TJR']]

#-----------------------------------------------------

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
#-------------------------------------------------------------------

from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


data['TJR'].value_counts()

sns.countplot(x='TJR', data=data,palette='hls')
plt.show()
plt.savefig('count_plot')


count_no_sub = len(data[data['TJR']==0])
count_sub = len(data[data['TJR']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription", pct_of_sub*100)


dd = data.groupby('TJR').mean()

desc = data.groupby('TJR').describe().apply(lambda s: s.apply(lambda x: format(x, 'f')))  #max/min/etc.
descipt = data.describe().apply(lambda s: s.apply(lambda x: format(x, 'f')))


data.Patient_Age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('hist_age')

dh = data.head(25)

#----------------------------------------------------------------

data_clean = data


#numeric gender
data_gnd = pd.get_dummies(data_clean['PatientGender'])

#gnd = data_gnd.head(50)

#join binary gender back to dataframe 
data_done = pd.concat((data_clean, data_gnd), axis=1)
data_done = data_done.drop(columns=['PatientGender', 'M'])
data_done = data_done.rename(columns={'F':'Gender'})

## convert to datetime
data_done['ServiceStartDate'] = pd.to_datetime(data_done['ServiceStartDate'], errors='coerce')

data_done['StartDay_year'] = data_done['ServiceStartDate'].dt.year
data_done['StartDay_month'] = data_done['ServiceStartDate'].dt.month
data_done['StartDay_week'] = data_done['ServiceStartDate'].dt.isocalendar().week


# Replace True with 1 and False with 0   SQL PULL ONLY
data_done['TJR'] = data_done['TJR'].replace({True: 1, None: 0})



##########Imputer/ OHE/ as needed

results = data_done

res = results.head(25)

#results.columns
#results.nunique()

# #Dropping columns that are not needed, duplicated or problematic  
# results = results.drop(columns=[ 'PERSON_ID', 'ClaimNr', 'ServiceStartDate', 'ServiceEndDate','ProviderSpecialtyCode', 
#                                 'ICDVersion','FileType', 'HCPCS','DiagnosisCodePrinciple', 'DiagnosisCode1','HfClaimLineId',
#                                 'HealthCardId','OrganizationName', 'U'])

#Dropping columns that are not needed, duplicated or problematic  IN SQL PULL
results = results.drop(columns=[ 'PERSON_ID', 'ServiceStartDate', 'HCPCS','DiagnosisCodePrincipal',
                                'DiagnosisCode1', 'RevenueCode',  'HfClaimId'])

#####Timing END
end_time = time.monotonic()
print("data manipulation complete", timedelta(seconds=end_time - start_time))

#####################################
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# Function for comparing different approaches
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


#results.dtypes



# column_name = "RevenueCode"
# null_values = results['RevenueCode'].isnull().sum()
# total_values = results.shape[0]
# percentage_null = (null_values / total_values) * 100
# print("Percentage of missing values in the {} column: {:.2f}%".format(column_name, percentage_null))
# #98.96%

# column_name = "DrgType"
# null_values = results['DrgType'].isnull().sum()
# total_values = results.shape[0]
# percentage_null = (null_values / total_values) * 100
# print("Percentage of missing values in the {} column: {:.2f}%".format(column_name, percentage_null))
# #99.83%

# results = results.drop(columns=['DRG', 'DrgType'])

###########################################################

from imblearn.over_sampling import SMOTE

# ###See null values by column
#results.isna().sum()  

# ###Fill in missing values in column with median value
# col = 'Patient_Age'
# median = results[col].median()
# results[col].fillna(median, inplace=True)


#fill NULL values with 0 
results = results.replace(np.nan,0)

# results.dtypes


#CHANGE non int columns to int columns
#results = results.astype({'Patient_Age':'int', 'DxPrinciple':'int', 'DxCode1':'int', 'Gender':'int', 'StartDay_week':'int'})

#CHANGE non int columns to int columns   SQL PULL
results = results.astype({'TJR':'int', 'hcpcs':'int', 'revcode':'int',  'BTaxId':'int', 'PoS':'int', 'DxCodeP':'int', 'DxCode1':'int', 'Gender':'int', 'StartDay_week':'int'})

#################

X = results.loc[:, results.columns != 'TJR']
y = results.loc[:, results.columns == 'TJR']


# os = SMOTE(random_state=56)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=56)
# columns = X_train.columns
# os_data_X,os_data_y=os.fit_resample(X_train, y_train)
# os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
# os_data_y= pd.DataFrame(data=os_data_y,columns=['TJR'])

# # we can Check the numbers of our data
# print("length of oversampled data is ",len(os_data_X))
# print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['TJR']==0]))
# print("Number of subscription",len(os_data_y[os_data_y['TJR']==1]))
# print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['TJR']==0])/len(os_data_X))
# print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['TJR']==1])/len(os_data_X))

# corr_matrix = results.corr().abs()

# mask = np.triu(np.ones_like(corr_matrix))

# sns.heatmap(corr_matrix, cmap='ocean_r', yticklabels=True, annot=False, mask=mask)

# #####Timing END
# end_time = time.monotonic()
# print("SMOTE complete",timedelta(seconds=end_time - start_time))


################# RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# # Load your data and split it into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=56)

# # Train the random forest classifier
# clf = RandomForestClassifier(n_estimators=200, random_state=56)
# clf.fit(X_train, y_train)


# #### Feature Importance 
# # Get the feature importance values
# importance = clf.feature_importances_

# # Create a dataframe to store the feature importance values and feature names
# feature_importance = pd.DataFrame({"feature": X.columns, "importance": importance})

# # Sort the dataframe by the importance values
# feature_importance = feature_importance.sort_values(by="importance", ascending=False)

# # Print the order of important features
# print(feature_importance)

# # Plot the feature importance values as a bar chart
# plt.bar(range(feature_importance.shape[0]), feature_importance["importance"])
# plt.xticks(range(feature_importance.shape[0]), feature_importance["feature"], rotation=90)
# plt.xlabel("Feature")
# plt.ylabel("Importance")
# plt.title("Feature Importance")
# plt.show()

# results.columns

#Excel pull
#cols = [ 'Patient_Age' , 'DxPrinciple' ,  'DxCode1' ,  'HfClaimId' , 'BillingTaxId'  , 'StartDay_week'  ,  'hcpcs' , 'osteo_knee_dummy' ,  'RevenueCode' , 'StartDay_month'  , 'osteo_hip_dummy' , 'PlaceOfService' ,'Gender' , 'StartDay_year'  ,'esrd_dummy' ] 

#SQL PULL
# cols = [ 'Patient_Age' , 'DxCodeP' ,  'DxCode1' ,  
#         'BTaxId'  , 'StartDay_week'  ,  'hcpcs' , 'osteo_knee_dummy' ,
#         'revcode' , 'StartDay_month'  , 'osteo_hip_dummy' , 'PoS' ,
#         'Gender' , 'StartDay_year'  ,'esrd_dummy' ] 

#Univariate Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# Select the top 10 features based on chi-square test
selector = SelectKBest(chi2, k=10)
X_new = selector.fit_transform(X, y)

# Get the boolean mask of selected features
selected_features = selector.get_support()

# Print the names of the selected features
feature_names = list(X.columns)
selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_features[i]]
print(selected_feature_names)

##Suggested Columns from 
#cols = ['Patient_Age', 'DxCodeP', 'DxCode1', 'BTaxId', 'StartDay_week', 'hcpcs', 'osteo_knee_dummy', 'revcode', 'osteo_hip_dummy', 'PoS'] 
cols = ['Patient_Age', 'DxCodeP', 'DxCode1', 'BTaxId', 'StartDay_week', 'hcpcs',  'revcode',  'PoS'] 



X=results[cols]
y=results['TJR']


# X=os_data_X[cols]
# y=os_data_y['TJR']

#make y a list!
y = list(y)

###Data scaling
from sklearn.preprocessing import StandardScaler

# Create an instance of the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data
scaler.fit(X)

# Transform the data
X_scaled = scaler.transform(X)


###Train / test at a 70/30
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=56)

#####Timing END
end_time = time.monotonic()
print("data changes finished", timedelta(seconds=end_time - start_time))


#####Timing END
end_time = time.monotonic()
print("start models", timedelta(seconds=end_time - start_time))

################################################
import statsmodels.api as sm

# logit_model=sm.Logit(y,X)
# result=logit_model.fit()
# print(result.summary2())


##-----------------
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import mean_absolute_error


# logreg = LogisticRegression(random_state=56)
# logreg.fit(X_train, y_train)

# y_pred = logreg.predict(X_test)
# print("Accuracy:", logreg.score(X_test, y_test))


# from sklearn.metrics import confusion_matrix
# confusion_matrix = confusion_matrix(y_test, y_pred)
# print(confusion_matrix)

# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))

# buildROC(y_test, y_pred)

# # confusion matrix plotting
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred, labels=logreg.classes_) 

# # labelling
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)
# disp.plot()
# plt.show()



# logreg1 = LogisticRegression(penalty='elasticnet', 
#                               class_weight='balanced', 
#                               solver='saga', 
#                               max_iter=125,
#                               l1_ratio=(.5),
#                               random_state=56)
# logreg1.fit(X_train, y_train)

# y_pred1 = logreg1.predict(X_test)
# print("Accuracy:", logreg1.score(X_test, y_test))
# print("Mean Absolute Error: " + str(mean_absolute_error(y_pred1, y_test)))

# from sklearn.metrics import confusion_matrix
# confusion_matrix = confusion_matrix(y_test, y_pred1)
# print(confusion_matrix)

# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred1))
 

# buildROC(y_test, y_pred1)

# # confusion matrix plotting
# from sklearn.metrics import confusion_matrix
# cm1 = confusion_matrix(y_test, y_pred1, labels=logreg1.classes_) 

# # labelling
# disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=logreg1.classes_)
# disp1.plot()
# plt.show()



# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import Ridge

# # Define the hyperparameter grid to search
# param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}

# # Create a logistic regression model
# log_reg = Ridge()

# # Use GridSearchCV to perform the hyperparameter tuning
# grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(X_train, y_train)

# # Print the best hyperparameters found by GridSearchCV
# print("Best hyperparameters:", grid_search.best_params_)

# # Evaluate the performance of the logistic regression model with the best hyperparameters
# best_log_reg = grid_search.best_estimator_
# test_accuracy = best_log_reg.score(X_test, y_test)
# print("Test accuracy:", test_accuracy)



########################################

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import mean_absolute_error

# # Initialize the model
# rf = RandomForestClassifier( n_estimators = 200, random_state=56)

# # Train the model
# rf.fit(X_train, y_train)

# # Predict on new data
# yrf_pred = rf.predict(X_test)
# print("Accuracy:", rf.score(X_test, y_test))
# print("Mean Absolute Error: " + str(mean_absolute_error(yrf_pred, y_test)))

# buildROC(y_test, yrf_pred)

# # confusion matrix plotting
# from sklearn.metrics import confusion_matrix
# cmrf = confusion_matrix(y_test, yrf_pred, labels=rf.classes_) 

# # labelling
# disp = ConfusionMatrixDisplay(confusion_matrix=cmrf, display_labels=rf.classes_)
# disp.plot()
# plt.show()
########################################################XGB

# from xgboost import XGBClassifier
# import xgboost as xgb
# import numpy as np


# xgbm = XGBClassifier(n_estimators=250,  n_jobs=2, random_state=56)
# xgbm.fit(X_train, y_train,
#           eval_set= [(X_train, y_train), (X_test, y_test)],
#           early_stopping_rounds=5,
#           verbose=False)

# # Predict on new data
# yxgb_pred = xgbm.predict(X_test)
# print("Accuracy:", xgbm.score(X_test, y_test))
# print("Mean Absolute Error: " + str(mean_absolute_error(yxgb_pred, y_test)))


# buildROC(y_test, yxgb_pred)

# # confusion matrix plotting
# from sklearn.metrics import confusion_matrix
# cmxgb = confusion_matrix(y_test, yxgb_pred, labels=xgbm.classes_) 

# # labelling
# dispxgb = ConfusionMatrixDisplay(confusion_matrix=cmxgb, display_labels=xgbm.classes_)
# dispxgb.plot()
# plt.show()


# #####Timing END
# end_time = time.monotonic()
# print(timedelta(seconds=end_time - start_time))

###########################################  GNB

# from sklearn.naive_bayes import GaussianNB


# # Train the Naive Bayes model
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)

# # Make predictions on the test set
# ygnb_pred = gnb.predict(X_test)
# print("Accuracy:", gnb.score(X_test, y_test))

# buildROC(y_test, ygnb_pred)

# # confusion matrix plotting
# from sklearn.metrics import confusion_matrix
# cmg = confusion_matrix(y_test, ygnb_pred, labels=gnb.classes_) 

# # labelling
# dispgnb = ConfusionMatrixDisplay(confusion_matrix=cmg, display_labels=gnb.classes_)
# dispgnb.plot()
# plt.show()

##########################################  KNN

from sklearn.neighbors import KNeighborsClassifier

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions on the test set
yknn_pred = knn.predict(X_test)
print("Accuracy:", knn.score(X_test, y_test))


buildROC(y_test, yknn_pred)

# confusion matrix plotting
from sklearn.metrics import confusion_matrix
cmk = confusion_matrix(y_test, yknn_pred, labels=knn.classes_) 

# labelling
disp = ConfusionMatrixDisplay(confusion_matrix=cmk, display_labels=knn.classes_)
disp.plot()
plt.show()


# from sklearn.model_selection import RandomizedSearchCV

# # Define the hyperparameters and their respective ranges
# param_dist = {'n_neighbors': np.arange(3, 12),
#               'algorithm': ['auto'], 
#               'leaf_size': [3, 15], 
#               'weights': ['uniform', 'distance'],
#               'p': [1, 2]}

# # Initialize the KNN model
# knn = KNeighborsClassifier()

# # Perform Random Search
# random_search = RandomizedSearchCV(knn, param_distributions=param_dist, n_iter=100, cv=5)
# random_search.fit(X_train, y_train)

# # Print the best hyperparameters
# # print("Best hyperparameters: ", random_search.best_params_)



# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import cross_val_score
# ###BEST model parameters
# # set the number of nearest neighbors to consider
# n_neighbors = 3
# # set the weighting method to be used
# weights = 'distance'
# # set the power parameter 
# p = 1
# # set the algorithm used to find the nearest neighbors
# algorithm = 'auto'
# # set the size of the leaf in the k-d tree algorithm
# leaf_size = 15

# # create a KNN classifier with BEST parameters
# best_knn = KNeighborsClassifier(n_neighbors=n_neighbors,
#                                 weights=weights,
#                                 algorithm=algorithm,
#                                 leaf_size=leaf_size,
#                                 p = p)

# best_knn.fit(X_train, y_train)

# # Make predictions on the test set
# ybknn_pred = best_knn.predict(X_test)
# print("Accuracy:", best_knn.score(X_test, y_test))


# scores = cross_val_score(best_knn, X, y, cv=5)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() *2))


# best_knn.score

# buildROC(y_test, ybknn_pred)

# buildROC(y_test, best_knn.predict(X_test))


# # confusion matrix plotting
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, ybknn_pred, labels=best_knn.classes_) 

# # labelling
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_knn.classes_)
# disp.plot()
# plt.show()

# #####Timing END
# end_time = time.monotonic()
# print(timedelta(seconds=end_time - start_time))



###################################
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report


# svm = SVC(kernel='linear', C=1, random_state=56) 
# svm.fit(X_train, y_train)

# y_pred = svm.predict(X_test)


# # confusion matrix plotting
# from sklearn.metrics import confusion_matrix
# cmsvm = confusion_matrix(y_test, y_pred, labels=svm.classes_) 

# # labelling
# disp = ConfusionMatrixDisplay(confusion_matrix=cmsvm, display_labels=svm.classes_)
# disp.plot()
# plt.show()


# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test,y_pred))


# #####Timing END
end_time = time.monotonic()
print("code done", timedelta(seconds=end_time - start_time))


