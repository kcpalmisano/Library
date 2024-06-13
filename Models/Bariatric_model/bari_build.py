
#basic load
import pandas as pd
import numpy as np
from datetime import datetime, date
import seaborn as sns
import matplotlib.pyplot as plt 
import pyodbc
import sqlalchemy
from sqlalchemy.orm import sessionmaker 

#ML load
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import cross_validate



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


### Visuals function for CM and ROC
from sklearn import metrics

@logger 
@timeit
def confusion_matrix_and_roc(y_test, y_pred, classifier): 
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
    print("The accuracy for XGB is: ", accuracy_score(y_test, y_pred))


     #ROC Graph
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred) 
    roc_auc = metrics.auc(fpr, tpr) 
    plt.title('Receiver Operating Characteristic') 
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc) 
    plt.legend(loc = 'lower right') 
    plt.plot([0, 1], [0, 1],'r--') 
    plt.ylabel('True Positive Rate') 
    plt.xlabel('False Positive Rate') 
    plt.show()    


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



####################
#Data selection query. Structure the WHERE clause to best fit needs of data 

# #####Timing 
# end_time = time.monotonic()
# print("Starting Data pull from SQL",timedelta(seconds=end_time - start_time), current_time)

# data = pd.read_sql_query('''
#      select  bt.Bari, *
#        from MEDICAL.ClaimsData cc
#                left join (Select CAST(1 as varchar) as Bari, * from Bariatric.CompletedSurgeries) bt
#                  on  bt.PERSON_ID = cc.PERSON_ID
#                  and  bt.ServiceStartDate = cc.ServiceStartDate
#                  and bt.HCPCS = cc.HCPCS
#       where  YEAR(cc.ServiceStartDate) =2022
#       --and MONTH(cc.ServiceStartDate) <=6
#       order by cc.PERSON_ID, cc.ServiceStartDate   
#                          ''' , con)

'''


select    ----5648
fe.person_id, 
fe.funds_program,		---- 268300010 / 268300001
fe.funds_programname,  ---- Bariatric COE / Bariatric
fe.customertypecode,	---- 1 / 2
fe.funds_source,		---- 268300000, 268300001, 268300003, 268300004, 268300005, 268300006, 268300007, 268300008, 268300010
fe.funds_sourcename,	---- 5 Star Referral, Hospital Reports, Claims, Future Moms Reports, Marketing List, Self Referral, Empire Referral, Other, Member Services
fe.funds_mshswarmtransfer, -- 0/1
fe.funds_outreach,		---- 268300000, 268300001, 268300002, 268300003, 268300004, 268300005, 268300006
fe.funds_outreachname, ---- 1st Attempt, 2nd Attempt, 3rd Attempt, Bad Number, Spoke to Participant, Unable to Reach, OOA
fe.funds_solicitationletter, --0/1
fe.funds_solicitationlettername, -- Net Sent / Sent
fe.statecode,			---- 0/1
fe.statecodename,		---- Active / Inactive
fe.statuscode,			---- 2, 268300123, 268300303, 268300304, 268300305, 268300308, 268300309, 268300310, 268300311, 268300312, 268300313, 268300314, 268300315, 268300316, 268300317
fe.statuscodename	---- Duplicate Record, Solicitation, Resolve, Outreach, Participating, Review, Transferred to Preferred, Program Completed, Switch 5 Star, Disenrolled, Self Disenroll, Lost Eligibility, Cancelled, Completed, Error
fe.Bari,
cc.*
FROM ClaimsData cc
left join (Select CAST(1 as varchar) as Bari, person_id, funds_program, funds_programname,  customertypecode, funds_source, funds_sourcename, funds_mshswarmtransfer,
funds_outreach,	funds_outreachname, funds_solicitationletter, funds_solicitationlettername, statecode,	statecodename,	statuscode,	statuscodename from funds_enrollment
where funds_programname LIKE '%bari%') fe
 on  fe.PERSON_ID = cc.PERSON_ID
 where YEAR(cc.ServiceStartDate) =2020 
 and fe.PERSON_ID is null
'''


#####Timing 
end_time = time.monotonic()
print("Starting Data pull from SQL",timedelta(seconds=end_time - start_time), current_time)


#######  Read in data               
datapull = pd.read_excel(r'C:\PATH\Data\Bari_data.xlsx')

data = datapull
####################################### Data exploration 

#dfu = data.nunique()


###drop id columns and singular value columns *** EXCEL ***
data = data.drop(columns=['MemberSSN', 'PatientSSN', 'RenderingZipCode4', 'RenderingTaxId', 'BillingZipCode4', 'DiagnosisCodeAdmitPOA', 'DiagnosisCode4POA',
'DiagnosisCode5POA', 'ProfitabilityCode', 'Region', 'ProviderClassCode', 'PackageNr', 'EmployerGroupDepartmentNr',
'RateCategory', 'RateSubcategory', 'ProcedureCodeSurgicalDesc', 'Deceased', 'DiagnosisCodePrinciplePOA', 'DiagnosisCode1POA',
  'DiagnosisCode2POA', 'DiagnosisCode3POA', 'MemberPenaltyAmount', 'ICDVersion' ])

# #drop id columns and singular value columns  *** SQL ***
# data = data.drop(columns=['MemberSSN', 'MemberRelationshipCode', 'PatientSSN', 'RenderingZipCode4',  'DiagnosisCode5POA' ,
#                           'BillingZipCode4', 'DiagnosisCodeAdmitPOA', 'DiagnosisCode4POA','ProfitabilityCode', 'PackageNr','EmployerGroupDepartmentNr',
#                           'RateCategory', 'RateSubcategory', 'ProcedureCodeSurgicalDesc','Deceased', 'DiagnosisCodePrinciplePOA', 'DiagnosisCode1POA',  
#                           'DiagnosisCode2POA', 'DiagnosisCode3POA', 'MemberPenaltyAmount', 'ICDVersion' ])


#drop more id columns or columns that provide no use
data = data.drop(columns= ['PaidDate', 'DependentNr', 'ClaimNr', 'ClaimAdjustmentNr', 'ClaimLineNr', 'GroupNr', 'SubgroupNr', 'IsSingleContract',
                'MemberName', 'PatientName', 'MemberAddress1', 'MemberAddress2', 'RenderingName', 'RenderingAddress1', 'RenderingAddress2',
                'BillingName', 'BillingAddress1', 'BillingAddress2', 'BilledServiceUnitCount', 'InvestigationClaimCode', 'AuthorizationNr', 
                'DiagnosisCodeAdmit', 'EPIN', 'DenialReasonCode', 'DenialReasonDescription', 'ClaimEntryDate', 'HealthCardId', 'BillingNPI', 
                'RenderingNPI', 'DischargeStatus',  'TypeOfBillCode', 'DiagnosisRelatedGroupType', 'MemberZipCode4', 'PostDate'
                  ])


#####Timing END
end_time = time.monotonic()
print('Load / column removal complete', timedelta(seconds=end_time - start_time))

###########################################################################
print("starting data manipulation")

##Clean up DOB to age
@logger
@timeit
def age(birthdate):
    today = date.today()
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    return age

#apply the function 
data['PatientDOB'] = data['PatientDOB'].apply(age)

#Rename the field
data = data.rename({'PatientDOB':'Patient_Age'}, axis=1)

#histogram
# data.Patient_Age.hist()
# plt.title('Histogram of Age')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.savefig('hist_age')

#####Timing END
end_time = time.monotonic()
print('Age complete', timedelta(seconds=end_time - start_time))
#-------------------------------------------------------------------


# ###Graphing set up
# plt.rc("font", size=14)

# sns.set(style="white")
# sns.set(style="whitegrid", color_codes=True)


#data['Bari'].nunique()


## see the amount of Bari in the data
data['Bari'].value_counts()

sns.countplot(x='Bari', data=data,palette='hls')
plt.show()
plt.savefig('count_plot')


count_no_sub = data['Bari'].isnull().sum()
count_sub = len(data[data['Bari']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
#print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
#print("percentage of subscription", pct_of_sub*100)


dd = data.groupby('Bari').mean()
####max/min/etc.
descipt = data.describe().apply(lambda s: s.apply(lambda x: format(x, 'f')))

#----------------------------------------------------------------
data1 = data

# Separate target from predictors
y = data1.Bari
X = data1.drop(['Bari'], axis=1)

#####Null value fixing

# calculate the percentage of null values in each column
null_percentage = X.isnull().mean() * 100


# Get names of columns with missing values
cols_with_missing = [col for col in X.columns
                     if X[col].isnull().any()]


dmin = data.head(50)

#Remove columns that have 40% or more NULL columns 
NA_val = X.isna().sum()
@logger
@timeit
def na_filter(na, threshold = .4): #only select variables that passees the threshold
    col_pass = []
    for i in na.keys():
        if na[i]/X.shape[0]<threshold:
            col_pass.append(i)
    return col_pass
dataNA = X[na_filter(NA_val)]
dataNA.columns

##########################

### Categorical to numeric
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


data_done = dataNA

### select categorical columns using select_dtypes
cat_cols = data_done.select_dtypes(include=['object']).columns.tolist()

## convert to datetime
data_done['ServiceStartDate'] = pd.to_datetime(data_done['ServiceStartDate'], errors='coerce')

#Split into year / month / week
data_done['StartDay_year'] = data_done['ServiceStartDate'].dt.year
data_done['StartDay_month'] = data_done['ServiceStartDate'].dt.month
data_done['StartDay_week'] = data_done['ServiceStartDate'].dt.isocalendar().week


## convert to datetime
data_done['ServiceEndDate'] = pd.to_datetime(data_done['ServiceEndDate'], errors='coerce')

#Split into year / month / week
data_done['EndtDay_year'] = data_done['ServiceEndDate'].dt.year
data_done['EndDay_month'] = data_done['ServiceEndDate'].dt.month
data_done['EndDay_week'] = data_done['ServiceEndDate'].dt.isocalendar().week


### Remove old columns
data_done = data_done.drop(columns=[ 'ServiceStartDate', 'ServiceEndDate' ])

### check column types 
#col_types = data_done.dtypes

### Label encoding for categorical columns 
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

## categorical columns
cols = data_done[['MemberRelationshipCode', 'MemberCity', 'MemberState', 'RenderingCity', 'RenderingState', 'ServiceRenderingType', 
           'DiagnosisCodePrinciple', 'DiagnosisCode1', 'HCPCS', 'PlaceOfService',  'InNetworkCode', 'CountyCode', 'PatientGender',
           'ProviderSpecialtyCodeAlt', 'BillingCity', 'BillingState', 'Par', 'PrimaryCarrierResponsibilityCode', 'ProcesserUnitId',
           'ProviderSpecialtyCode', 'ProviderLocationCode']]

### label encoding
label_encode_columns(data_done, cols)

### removal of cost based data for possible data leakage
data_done = data_done.drop(columns=[ 'ClaimChargedAmount', 'ClaimPaidAmount', 'ClaimLineChargedAmount', 'ClaimLinePaidAmount', 'CopayAmount', 'CoinsuranceAmount', 'DeductibleAmount', 'ApprovedAmount', 'CoveredExpenseAmount' ])


### Check all unique values in a column
# data_done['ProcesserUnitId'].unique()


#count the NULLs in each column and total them up 
x_na_sum = data_done.isna().sum()  

### replace all null columns with 0
X = data_done.replace(np.nan,0)
y = y.replace(np.nan,0)


### CHANGE non int columns to int columns
X = X.astype({'StartDay_week':'int', 'EndDay_week':'int', 'RenderingZipCode':'int', 'PERSON_ID':'int', 'MemberZipCode':'int' })

X.dtypes


#put Bari back with transformed data into a dataframe for useage
results = pd.concat([X,y], axis=1) 


#####  Heatmap
corr_matrix = results.corr().abs()

mask = np.triu(np.ones_like(corr_matrix))

sns.heatmap(corr_matrix, cmap='ocean_r',  yticklabels=True, annot=False, mask=mask)


#####Timing END
end_time = time.monotonic()
print('data manipulation complete', timedelta(seconds=end_time - start_time))



###########Over Sampling 

from imblearn.over_sampling import SMOTE


os = SMOTE(random_state=56)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)
columns = X_train.columns
os_data_X,os_data_y=os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['Bari'])



# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no Bari in oversampled data",len(os_data_y[os_data_y['Bari']==0]))
print("Number of Bari",len(os_data_y[os_data_y['Bari']==1]))
print("Proportion of no Bari data in oversampled data is ",len(os_data_y[os_data_y['Bari']==0])/len(os_data_X))
print("Proportion of Bari data in oversampled data is ",len(os_data_y[os_data_y['Bari']==1])/len(os_data_X))

##rslts = results.columns = [ ]

count_no_sub = len(os_data_X)
count_sub = len(os_data_y)
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription", pct_of_sub*100)


X = os_data_X
y = os_data_y

y = np.ravel(y)


#####Timing END
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

#######################################################################  FEATURES

##########################  Mutual Information 



discrete_features = X.dtypes == int


from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores               #mi_scores[::3]  # slicing for every 3rd feature

#visual 
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(12, 8))
plot_mi_scores(mi_scores)


# Select the features with MI scores greater than 0.2
mi_features = mi_scores[mi_scores > 0.2].index.tolist()

# Print the selected features
print(mi_features)

#### above .2
mi_cols = ['RenderingZipCode', 'MemberZipCode', 'DiagnosisCodePrinciple', 'HCPCS', 'BillingTaxId', 'BillingZipCode', 'ProviderSpecialtyCodeAlt',  'DiagnosisCode1', 'ProviderSpecialtyCode', 'PlaceOfService' ]

ht_mi = pd.DataFrame({'features': mi_features, 'mi_scores': mi_scores})

#####  Heatmap
sns.heatmap(ht_mi.set_index('mi_cols'), cmap= 'ocean_r', yticklabels=True, annot=False, mask=mask )

#####Timing END
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

###################################   CHI ^2

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


df = X[X >=0]

df = df.loc[:, (X >=0).all()]


# Feature importance based on chi-square test
selector = SelectKBest(chi2, k=20)
X_new = selector.fit_transform(df, y)

# Get the boolean mask of selected features
selected_features = selector.get_support()

# Print the names of the selected features
feature_names = list(df.columns)
selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_features[i]]
print(selected_feature_names)

chi_cols = ['MemberCity', 'MemberState', 'MemberZipCode', 'RenderingCity', 'RenderingZipCode', 'BillingCity',
            'BillingZipCode', 'BillingTaxId', 'DiagnosisCodePrinciple', 'DiagnosisCode1', 'HCPCS',
            'ProviderSpecialtyCode', 'InNetworkCode', 'ProviderSpecialtyCodeAlt', 'Patient_Age',
            'StartDay_month', 'StartDay_week', 'EndDay_month', 'EndDay_week']

# Get the scores of the selected features
scores = selector.scores_

# Create a dataframe with the feature names and their importance scores
df_scores = pd.DataFrame({'Feature': selected_feature_names, 'Score': scores})

# Sort the dataframe by score in descending order
df_scores = df_scores.sort_values(by='Score', ascending=False)

# Create a bar plot of feature importance
sns.barplot(x='Score', y='Feature', data=df_scores, color='b')

# Add labels and title
plt.xlabel('Chi-Squared Score')
plt.ylabel('Feature Name')
plt.title('Feature Importance based on Chi-Squared Test')
plt.show()

#####Timing END
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

###################################    RFE
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# Create a Random Forest classifier object
rf = RandomForestClassifier(random_state=56)

# Use RFECV to perform RFE with cross-validation
rfecv = RFECV(estimator=rf, step=1, cv=5, scoring='accuracy')
rfecv.fit(X, y)


# Print the optimal number of features and their rankings
print("Optimal number of features: %d" % rfecv.n_features_)
print("Feature rankings: ", rfecv.ranking_)
print("Selected Features: ", X.columns[rfecv.support_])

rfe_cols = ['HCPCS']

#####Timing END
end_time = time.monotonic()
print("data changes finished", timedelta(seconds=end_time - start_time))

########################################### MODELS ##########################################


mi_cols = ['RenderingZipCode', 'MemberZipCode', 'DiagnosisCodePrinciple', 'HCPCS', 'BillingTaxId', 'BillingZipCode',  'DiagnosisCode1', 'ProviderSpecialtyCode', 'PlaceOfService' ]


chi_cols = ['MemberCity', 'MemberState', 'MemberZipCode', 'RenderingCity', 'RenderingZipCode', 'BillingCity',
            'BillingZipCode', 'BillingTaxId', 'DiagnosisCodePrinciple', 'DiagnosisCode1', 'HCPCS',
            'ProviderSpecialtyCode', 'InNetworkCode', 'ProviderSpecialtyCodeAlt', 'Patient_Age',
            'StartDay_month', 'StartDay_week', 'EndDay_month', 'EndDay_week']

fe_cols = ['HCPCS']

X=X[mi_cols]
X=X[chi_cols]
X=X[rfe_cols]
y=y


#####Split to Train Test  RANDOM STATE =56 
#make y a list!
y = list(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=56)


#####Timing END
end_time = time.monotonic()
print("data changes finished", timedelta(seconds=end_time - start_time))

##################################################################

# Divide data into training and validation subsets
X_train_full, X_test_full, y_train, y_test = train_test_split(X, y,  test_size=0.2,  random_state=56)

# Drop columns with missing values (simplest approach)
cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()] 
X_train_full.drop(cols_with_missing, axis=1, inplace=True)
X_test_full.drop(cols_with_missing, axis=1, inplace=True)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numerical_cols
#### ClaimLineStatusCode, BillingZipCode, BillingTaxId, PaidServiceUnitCount, Patient_Age, BenefitPaymentTierCode, StartDay_year, StartDay_month, EndtDay_year, EndDay_month 

X_train = X_train_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

########################################################Data scaling
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Create an instance of the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data
scaler.fit(X)

# Transform the data
X_scaled = scaler.transform(X)
##########



# ####################################  Model Implementation ##############################
import statsmodels.api as sm

logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


###-----------------
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=56)
logreg = LogisticRegression(random_state=56)
logreg.fit(X_train, y_train)


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


#calculate precision, recal, f1        
#precision =  true positives / (true positives + false positives) 
#recall ability to find all positives      = true positives / (true positives + false negatives)
#F1 best score is 1 and worst is 0
#Support is number of occurences in each class from y_test 
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# #--------------------------------------------------------
# ##ROC Curve
# ##the more to the left and away from red the better
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Log Reg')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()



#### #MODEL FINE TUNNING and visual out comes

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# define the parameter grid
param_grid = {'C': [0.1, 1, 10, 100],
              'penalty': ['none', 'l1', 'l2']}

# create the logistic regression model
logreg = LogisticRegression()

# create the grid search object
grid_search = GridSearchCV(logreg, param_grid, cv=5,refit=True)

# fit the grid search to the data
grid_search.fit(X_train, y_train)

# print the best parameters and the best score
print("Best parameters: {}".format(grid_search.best_params_))
print("Best score: {:.2f}".format(grid_search.best_score_))

# make predictions using the best parameters and evaluate the accuracy
y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Logistic Regression on test set: {:.2f}".format(accuracy))

# '''
# Best score: 0.60
# Accuracy of Logistic Regression on test set: 0.67
# '''

#####Timing END
end_time = time.monotonic()
print('Log Reg complete', timedelta(seconds=end_time - start_time))
################################################################### Random Forest

from sklearn.ensemble import RandomForestClassifier

# Initialize the model
rf = RandomForestClassifier(random_state=56)

# Train the model
rf.fit(X_train, y_train)

# Predict on new data
yrf_pred = rf.predict(X_test)
print('Accuracy of Random Forest Classifier on test set: {:.2f}'.format(rf.score(X_test, y_test)))


#--------------------------------------------------------
##ROC Curve
##the more to the left and away from red the better

rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest Regression (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Random Forest')
plt.legend(loc="lower right")
plt.savefig('RF_ROC')
plt.show()

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, yrf_pred)
print(confusion_matrix)


buildROC(yrf_pred, y_test)

# confusion matrix plotting
from sklearn.metrics import confusion_matrix
cmrf = confusion_matrix(y_test, yrf_pred, labels=rf.classes_) 

# labelling
disp = ConfusionMatrixDisplay(confusion_matrix=cmrf, display_labels=rf.classes_)
disp.plot()
plt.show()

############## Search
# #-------------------------------GridSearch

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV

# # Define the parameter grid
# param_grid = {'n_estimators': [20, 50, 100], 'max_depth': [5, 10, 15]}


# # Create an instance of GridSearchCV with the desired parameters
# grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')

# # Fit the GridSearchCV object to the data
# grid_search.fit(X_train, y_train)

# # Print the best parameters and the best score
# crf = grid_search.best_estimator_
# print("Best parameters: ", grid_search.best_params_)
# print("Best score: ", grid_search.best_score_)

# # Predict on the test set
# y_pred = crf.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy: ", accuracy)

###-------------------------------------------------------Random Search
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.tree import export_graphviz
# from scipy.stats import randint as sp_randint

# # Define the parameter grid
# param_dist = {'n_estimators': sp_randint( 50, 100, 200), 
#               'max_depth': sp_randint( 3, 5, 10),
#               'min_samples_split': sp_randint(2, 10, 15),
#               'min_samples_leaf': sp_randint(1, 3, 5),
#               'max_features': ['sqrt']}

# # Instantiate the random search
# random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=2, cv=5, random_state=56)

# # Fit the random search to the data
# random_search.fit(X, y)

# # Get the best model
# best_model = random_search.best_estimator_

# best_params= random_search.best_params_

# # Select one tree from the forest
# estimator = best_model.estimators_[0]


# feature_names = [f'feature{i}' for i in range(X_train.shape[1])]
# target_names = list(range(y_train.nunique()))



# #-----------------------------------------------------------------------visual 
# ###features 

# # Get feature importances
# importances = crf.feature_importances_

# # Sort feature importances in descending order
# indices = np.argsort(importances)[::-1]

# # Create plot
# plt.figure()

# # Create plot title
# plt.title("Feature Importance")

# # Add bars
# plt.bar(range(X.shape[1]), importances[indices])

# # Add feature names as x-axis labels
# plt.xticks(range(X.shape[1]), rotation=90)

# # Show plot
# plt.show()

##############---------------------
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import make_classification
# from sklearn.tree import export_graphviz
# from six import StringIO  
# from IPython.display import Image  
# import pydotplus

# best_tree = None
# best_acc = 0
# for i, tree in enumerate(crf.estimators_):
#     acc = accuracy_score(y_test, tree.predict(X_test))
#     if acc > best_acc:
#         best_acc = acc
#         best_tree = tree
        
        

# # Export as dot file
# dot_data = tree.export_graphviz(best_tree)  

# # Draw graph
# graph = pydotplus.graph_from_dot_data(dot_data)  

# # Show graph
# Image(graph.create_png())        
        
##################################################### XG BOOST
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# Create a binary classification problem by setting objective='binary:logistic'
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', random_state = 56)


# Fit the model to the training data
xgb_clf.fit(X_train, y_train)

#Cross validation on model
scores = cross_val_score(xgb_clf, X_train, y_train, cv=2)

# Print the mean and standard deviation of the cross-validation scores
print("Accuracy XGB: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
##^

# Predict the binary outcomes for the test data
yxb_pred = xgb_clf.predict(X_test)

# Evaluate the model using accuracy
accuracy = (yxb_pred == y_test).mean()
print("Accuracy: ", accuracy)

############## visuals 
# confusion matrix plotting
from sklearn.metrics import confusion_matrix
cmxg = confusion_matrix(y_test, yxb_pred, labels=xgb_clf.classes_) 

# labelling
disp = ConfusionMatrixDisplay(confusion_matrix=cmxg, display_labels=xgb_clf.classes_)
disp.plot()
plt.show()


# printing accuracy 
print("The accuracy for XGB is: ", accuracy_score(y_test, yxb_pred))

##ROC graph    
buildROC(yxb_pred, y_test)

############################## Search
# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
    

##initialize domain space for range of values
space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }


###function that defines the objective
def objective(space):
    clf=xgb.XGBClassifier(
                    n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']))
    
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    clf.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="auc",
            early_stopping_rounds=10,verbose=False)
    

    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }

###check various hyperparameters over 100 times
trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)

#view best hyperparameters
print("The best hyperparameters are : ","\n")
print(best_hyperparams)



best_hyperparams = 
{'colsample_bytree': 0.9999765937259769, 'gamma': 2.148501672713458, 'max_depth': 18.0, 'min_child_weight': 1.0, 'reg_alpha': 44.0, 'reg_lambda': 0.00046887662942335373}


# using BEST hyperparameters for model
xgbb_clf = xgb.XGBClassifier(objective='binary:logistic',
                                      gamma = 2.1485,
                                    max_depth= 18,
                                    min_child_weight= 1.0,
                                    reg_alpha= 44.0,
                                    reg_lambda= 0.00046)

# Fit the model to the training data
xgbb_clf.fit(X_train, y_train)

# Predict the binary outcomes for the test data
yxbb_pred = xgbb_clf.predict(X_test)

# Evaluate the model using accuracy
accuracy = (yxbb_pred == y_test).mean()
print("Accuracy: ", accuracy)

############## visuals 
# confusion matrix plotting
from sklearn.metrics import confusion_matrix
cmxgb = confusion_matrix(y_test, yxbb_pred, labels=xgbb_clf.classes_) 

# labelling
disp = ConfusionMatrixDisplay(confusion_matrix=cmxgb, display_labels=xgbb_clf.classes_)
disp.plot()
plt.show()


# printing accuracy 
print("The accuracy is: ", accuracy_score(y_test, yxbb_pred))

##ROC graph    
buildROC(yxbb_pred, y_test)




######################################################Gradient boosting    

# Import the necessary libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score


# Initialize BASE model
gb = GradientBoostingClassifier(random_state = 56)

# Train the model
gb.fit(X_train, y_train)

# Predict on new data
ygb_pred = gb.predict(X_test)
print('Accuracy of GB Classifier on test set: {:.2f}'.format(gb.score(X_test, y_test)))

# #--------------------------------------------------------
##ROC Curve
##the more to the left and away from red the better

gb_roc_auc = roc_auc_score(y_test, gb.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, gb.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Gradient Boosting (area = %0.2f)' % gb_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic GB')
plt.legend(loc="lower right")
plt.savefig('GB_ROC')
plt.show()

##ROC graph    
buildROC(ygb_pred, y_test)

# confusion matrix plotting
from sklearn.metrics import confusion_matrix
cmgb = confusion_matrix(y_test, ygb_pred, labels=gb.classes_) 

# labelling
disp = ConfusionMatrixDisplay(confusion_matrix=cmgb, display_labels=gb.classes_)
disp.plot()
plt.show()



################################################################################## END
#Close SQL connection
#conn.close()
#cursor.close()
#engine.dispose()
con.dispose()


#####Timing END
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
