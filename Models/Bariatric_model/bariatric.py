

#basic load
import pandas as pd
import numpy as np
from datetime import  datetime
import seaborn as sns
import matplotlib.pyplot as plt 
import sqlalchemy


#ML load
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


#####Timing START
import time
from datetime import timedelta
current_time = datetime.now()
start_time = time.monotonic()
print("starting Bari py script", current_time)

################################################

#### @wraps for making sure the function inherits its name and properties
from functools import wraps

def logger(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        """wrapper documentation"""
        print(f"----- {function.__name__}: start -----")
        output = function(*args, **kwargs)
        print(f"----- {function.__name__}: end -----")
        return output
    return wrapper

#### @timeit records the time a function starts / stops 
import time

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f'{func.__name__} took {end - start:.6f} seconds to complete')
        return result
    return wrapper


### Visuals function for CM
from sklearn import metrics

@logger 
@timeit
def confusion_matrix_acc(y_test, y_pred, classifier): 
    '''
    Purpose: 
        Provide Confusion Matrix and ROCAUC visuals as well as model accuracy
    Args:
        y_test = test data
        y_pred = prediction data
        classifier = model classifier 
    Returns:
        Confusion Matrix and 
    '''
    cmxg = confusion_matrix(y_test, y_pred, labels=classifier.classes_) 
    
    # labelling 
    disp = ConfusionMatrixDisplay(confusion_matrix=cmxg, display_labels=classifier.classes_) 
    disp.plot() 
    plt.show() 
    
    # printing accuracy 
    print("The accuracy for the Model is: ", accuracy_score(y_test, y_pred))


### Visuals function for ROC
@logger 
@timeit
def buildROC(target_test,test_preds):
    fpr, tpr, threshold = metrics.roc_curve(target_test, test_preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate') 


### Data oversampling 
from imblearn.over_sampling import SMOTE

@logger
@timeit
def oversample_data(X, y, random_state=56):
    '''
    Purpose: 
        Oversample data so goal outcome is balanced with non-goal
    Args: 
        X = train data
        y = test data
    Returns: 
        Provides balanced data and prints numbers of each
    '''
    os = SMOTE(random_state=56)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    columns = X_train.columns
    y_train = y_train.astype(int)
    os_data_X,os_data_y=os.fit_resample(X_train, y_train)
    os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
    os_data_y= pd.DataFrame(data=os_data_y,columns=['Bari'])
    
    
    # we can Check the numbers of our data
    print("length of oversampled data is ",len(os_data_X))
    print("Number of no Goal in oversampled data",len(os_data_y[os_data_y['Bari']==0]))
    print("Number of Goal",len(os_data_y[os_data_y['Bari']==1]))
    print("Proportion of non Goal data in oversampled data is ",len(os_data_y[os_data_y['Bari']==0])/len(os_data_X))
    print("Proportion of Goal data in oversampled data is ",len(os_data_y[os_data_y['Bari']==1])/len(os_data_X))
    
    
    count_no_sub = len(os_data_X)
    count_sub = len(os_data_y)
    pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
    print("percentage of non Goal is", pct_of_no_sub*100)
    pct_of_sub = count_sub/(count_no_sub+count_sub)
    print("percentage of Goal", pct_of_sub*100)
    
    return os_data_X, os_data_y


### Label Encoding function 
    
@logger
@timeit
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

################################################


###Connect to SQL Server
con = sqlalchemy.create_engine('mssql://SERVER/DATABASE?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server',
                                fast_executemany=True)    

#Check for database connection
if not con:
    raise Exception("Connection to DbHfProd Error")
else: 
    print("Connection to DbHfProd Successful")

####################
#Data selection query. Structure the WHERE clause to best fit needs of data 

#####Timing 
end_time = time.monotonic()
print("Starting Data pull from SQL",timedelta(seconds=end_time - start_time), current_time)

data = pd.read_sql_query('''
     select  bt.Bari, 
     cc.PERSON_ID,
     HfClaimId,  
     RenderingZipCode, 
     MemberZipCode, 
     DiagnosisCodePrinciple, 
     cc.HCPCS, 
     BillingTaxId, 
     BillingZipCode,   
     DiagnosisCode1, 
     ProviderSpecialtyCode, 
     PlaceOfService 
       from TableData cc
               left join (Select CAST(1 as varchar) as Bari, * from Bari) bt
                 on  bt.PERSON_ID = cc.PERSON_ID
                 and  bt.ServiceStartDate = cc.ServiceStartDate
                 and bt.HCPCS = cc.HCPCS
      where  YEAR(cc.ServiceStartDate) =2019  
                         ''' , con)
                         
                         
# Separate target from predictors
y = data.Bari
X = data.drop(['Bari'], axis=1)
                         
                
#####Timing END
end_time = time.monotonic()
print('Load / column removal complete', timedelta(seconds=end_time - start_time))

###########################################################################
print("starting data manipulation")


## categorical columns
cols = data[['PERSON_ID', 'HfClaimId', 'RenderingZipCode', 'MemberZipCode', 'DiagnosisCodePrinciple', 'HCPCS', 'BillingTaxId', 'BillingZipCode',  'DiagnosisCode1', 'ProviderSpecialtyCode', 'PlaceOfService' ]]

### label encoding
label_encode_columns(data, cols)

### replace all null columns with 0
X = data.replace(np.nan,0)
y = y.replace(np.nan,0)

### CHANGE non int columns to int columns
X = X.astype({ 'Bari':'int', 'RenderingZipCode':'int', 'PERSON_ID':'int',  'MemberZipCode':'int', 'BillingTaxId':'int', 'BillingZipCode':'int', 'PlaceOfService':'int' })

############### Over Sampling 

os_data_X, os_data_y = oversample_data(X, y)

#rename and ravel (makes a flattened array) data
X = os_data_X
y = os_data_y

y = np.ravel(y)


########################################### MODEL ##########################################

#features to use
mi_cols = [ 'MemberZipCode', 'HCPCS', 'DiagnosisCodePrinciple', 'DiagnosisCode1', 'PlaceOfService', 'ProviderSpecialtyCode',  'RenderingZipCode' ]
 #  

### set X with selected features 
X=X[mi_cols]
y=y


### Heatmap for correlation 
corr_matrix = X.corr().abs()
mask = np.triu(np.ones_like(corr_matrix))
sns.heatmap(corr_matrix, cmap='ocean_r',  yticklabels=True, annot=False, mask=mask)


#make y a list!
y = list(y)

#####Split to Train Test  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)

####################################  Model Implementation ##############################

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# Create a binary classification problem by setting objective='binary:logistic'
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.1,  random_state = 56)
# max_depth=6, reg_alpha=0.01, reg_lambda=0.01,

# Fit the model to the training data
xgb_clf.fit(X_train, y_train)

#Cross validation on model
scores = cross_val_score(xgb_clf, X_train, y_train, cv=2)

# Print the mean and standard deviation of the cross-validation scores
print("Accuracy XGB: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
##^

# Predict the binary outcomes for the test data
yxb_pred = xgb_clf.predict(X_test)


############## visuals 
confusion_matrix_acc(y_test, yxb_pred, xgb_clf)

buildROC(y_test, yxb_pred)


################################################################################## END
#Close SQL connection
con.dispose()

#####Timing END
end_time = time.monotonic()
print('Model complete', timedelta(seconds=end_time - start_time))

         
###################################
# import pickle

# # save the model to a file
# with open('Bariatric_model.pkl', 'wb') as f:
#     pickle.dump(brf, f)                    
                   

