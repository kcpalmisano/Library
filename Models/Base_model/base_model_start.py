
# -*- coding: utf-8 -*-
'''
Cpalmisano  12/02/22
GENERALIZED MODEL STARTING POINT TO EVALUATE VARIOUS MODELS

Designed to see correlation of data and then balance dataset using SMOTE
Then see various verisons of feature importance (MI, Chi2, RFE) to determine best approach
Then test various models using ROC/AUC and plot them as well as visualize
their various confusion matrixes
'''

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
    print('connections closed')        
                

def buildROC(target_test, test_preds, model_name, color):
    '''
    Plots the Receiver Operating Characteristic (ROC) curve and calculates the Area Under the Curve (AUC).

    Args:
        target_test (array-like): True binary labels in the test set. Shape (n_samples,).
        test_preds (array-like): Target scores, can either be probability estimates of the positive class, 
                                 confidence values, or binary decisions. Shape (n_samples,).
        model_name (str): Name of the model to be displayed in the legend.
        color (str): Color of the ROC curve.

    Returns:
        None: The function plots the ROC curve and prints the AUC.
      
    EX: >>> buildROC(y_test, y_pred_prob, 'Logistic Regression', 'blue')
    '''
    fpr, tpr, threshold = metrics.roc_curve(target_test, test_preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, color, lw=2, label='%s (AUC = %0.2f)' % (model_name, roc_auc))
    plt.legend(loc='lower right')
    
def plot_combined_roc(y_test, models_preds_colors):
    '''
    Plots combined ROC curves for multiple models.

    Args:
        y_test (array-like): True binary labels in the test set. Shape (n_samples,).
        models_preds_colors (list of tuples): Each tuple contains (predictions, model_name, color).

    Returns:
        None: The function plots the combined ROC curves.
    '''
    plt.figure()
    for preds, model_name, color in models_preds_colors:
        buildROC(y_test, preds, model_name, color)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.show()    

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


####################!!!
#Data selection query. Structure the WHERE clause to best fit needs of data 

#####Timing 
end_time = time.monotonic()
print("Starting Data pull from SQL",timedelta(seconds=end_time - start_time))

data = pd.read_sql_query('''
         SELECT top 100000 
         CASE WHEN fe.funds_program = '268300000' THEN 1 ELSE 0 END AS FLAG,
                 *
            FROM TABLE cft
            LEFT JOIN OTHER_TABLE fe on 
            fe.person_id = cft.Person_Id
            WHERE YEAR(ServiceStartDate) >= 2019;
                          ''', con)
                          
end_time = time.monotonic()
print("Finished Data pull from SQL",timedelta(seconds=end_time - start_time))                          


### Set identifying columns up to drop 
column_list = ['COLUMN1', 'COLUMN2' ]

data = data.drop(columns=column_list)

#-----------------------------------------------------------
#set up target dataframe
target_id = data[['person_id', 'FLAG']]    
target = data[['FLAG']]


data = data.drop(columns=['person_id'])
#-----------------------------------------------------

##Clean up DOB to age
def age(birthdate):
    today = date.today()
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    return age

#apply the function 
data['tDOB'] = data['DOB'].apply(age)

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


data['FLAG'].value_counts()

###Count plot 
sns.countplot(x='FLAG', data=data,palette='hls')
plt.show()
plt.savefig('count_plot')


count_no_sub = len(data[data['FLAG']==0])
count_sub = len(data[data['FLAG']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription", pct_of_sub*100)



### age based plot
data.Patient_Age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('hist_age')                          
                          
############################################################!!!

data_clean = data

#numeric gender
data_gnd = pd.get_dummies(data_clean['PatientGender'])


#join binary gender back to dataframe 
data_done = pd.concat((data_clean, data_gnd), axis=1)
data_done = data_done.drop(columns=['F', 'M'])

## convert to datetime
data_done['ServiceStartDate'] = pd.to_datetime(data_done['ServiceStartDate'], errors='coerce')

data_done['StartDay_year'] = data_done['ServiceStartDate'].dt.year
data_done['StartDay_month'] = data_done['ServiceStartDate'].dt.month
data_done['StartDay_week'] = data_done['ServiceStartDate'].dt.isocalendar().week


# Replace True with 1 and False with 0   SQL PULL ONLY
data_done['FLAG'] = data_done['FLAG'].replace({True: 1, None: 0})

##########   Imputer/ OHE/ as needed

results = data_done

data_cols = data_done.columns

#Remove columns that have 30% or more NULL values in columns 
def na_filter(na, threshold=0.3):  # Only select variables that pass the threshold
  col_pass = []
  for col, value in na.items():  # Iterate over key-value pairs
    if value / results.shape[0] < threshold:
      col_pass.append(col)
  return col_pass

NA_val = results.isna().sum()
filtered_cols = na_filter(NA_val)

results_fil = results[filtered_cols] # Filter using list of column names


### Label Encoding function 
from sklearn.preprocessing import LabelEncoder

def label_encode_columns(df, columns):
  """
  Purpose:
    Label encode a set of columns

  Args:
    df (pandas DataFrame): The input DataFrame
    columns (list of str): The list of column names to label encode

  Returns:
    The transformed DataFrame with the specified columns label encoded
  """
  for col in columns:
    le = LabelEncoder()
    try:
      df[col] = le.fit_transform(df[col].astype(str))
    except ValueError as e:
      print(f"Error encoding column '{col}': {e}")  # Notify about encoding error

  return df


def get_non_numeric_columns(df):
  """
  This function takes a pandas DataFrame and returns a list containing
  all columns that are not numeric (object, string, etc.).
  """
  non_numeric_cols = []
  for col in df.columns:
    # Exclude numeric data types (int, float)
    if not pd.api.types.is_numeric_dtype(df[col]):
      non_numeric_cols.append(col)
  return non_numeric_cols


# Get non-numeric field names
non_numeric_cols = get_non_numeric_columns(results_fil)

# Label encode columns (call only once)
results_fil = label_encode_columns(results_fil, non_numeric_cols)  # Avoid modifying original

#############################################################!!!
results = results_fil.dropna()

X = results.loc[:, results.columns != 'FLAG']
y = results.loc[:, results.columns == 'FLAG']

from imblearn.over_sampling import SMOTE
###Synthetic Minority Oversampling Technique

## Train/Test/Splite to set up for SMOTE (equalizing at 50% 0/1 outcomes)
os = SMOTE(random_state=56)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)
columns = X_train.columns

# Check for missing values in X_train
#print(X_train.isnull().sum())  # Shows the count of missing values per column

na_vals = X_train.isnull().sum()

##Count values of y_train (1/0)
print(y_train.value_counts(dropna=False))  

X_train = X_train.dropna()
y_train = y_train.dropna()

#change to int if not already
if not pd.api.types.is_numeric_dtype(X_train):
    X_train = X_train.astype(int)


os_data_X,os_data_y=os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['FLAG'])


# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no FLAG in oversampled data",len(os_data_y[os_data_y['FLAG']==0]))
print("Number of FLAG",len(os_data_y[os_data_y['FLAG']==1]))
print("Proportion of no FLAG data in oversampled data is ",len(os_data_y[os_data_y['FLAG']==0])/len(os_data_X))
print("Proportion of FLAG data in oversampled data is ",len(os_data_y[os_data_y['FLAG']==1])/len(os_data_X))

##rslts = results.columns = [ ]

# DOUBLE Check the numbers of our data
count_no_sub = len(os_data_X)
count_sub = len(os_data_y)
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription", pct_of_sub*100)

#Visualize the SMOTE process to make sure we have an even 0/1
sns.countplot(x='FLAG', data=os_data_y, hue='FLAG', palette='hls', legend=False)
plt.show()
plt.savefig('count_plot')

##################################### Correlation 
#####  Heatmap **Crappy**
corr_matrix = os_data_X.corr().abs()
mask = np.triu(np.ones_like(corr_matrix))
sns.heatmap(corr_matrix, cmap='ocean_r',  yticklabels=True, annot=False, mask=mask)

# Set a threshold for highly correlated features
threshold = 0.8

# Create a mask to identify highly correlated features
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation above the threshold
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]


# Drop the highly correlated features from the DataFrame
df_reduced = os_data_X.drop(columns=to_drop)

print(f"Removed columns: {to_drop}")
print("Reduced DataFrame shape:", df_reduced.shape)


## heatmap **Better**
import seaborn as sns
#from scipy.cluster.hierarchy import dendrogram, linkage

# Select top x features based on importance
top_features = df_reduced.columns[:50]

# Calculate correlation matrix
cor_matrix = df_reduced[top_features].corr().abs()

# Generate heatmap 
fig, ax = plt.subplots(figsize=(25, 12))  # Adjust figure size as needed
mask = np.triu(np.ones_like(cor_matrix))
sns.heatmap(cor_matrix, cmap='ocean_r', yticklabels=True, annot=False, mask=mask, ax=ax)

plt.show()


###########################################!!!  Mutual Information 

# update data with SMOTE based data 
X = os_data_X
y = os_data_y

X = X.drop(columns=['Person_Id'])

discrete_features = X.dtypes == int


from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores               #mi_scores[::3]  # slicing for every 3rd feature

hi_mi = mi_scores[mi_scores > 0.2]


#visual 
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(hi_mi)


########################################################## Chi 2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# Feature importance based on chi-square test, k = fields wanted
selector = SelectKBest(chi2, k=12)
X_new = selector.fit_transform(X, y)

# Get the boolean mask of selected features
selected_features = selector.get_support()

# Print the names of the selected features
feature_names = list(X.columns)
selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_features[i]]
print(selected_feature_names)

##########################################################  RFE

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression  


# Define the RFE estimator
selector = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=16)

# Fit the RFE model to your data
selector.fit(X, y)

# Get the features selected by RFE
selected_features = X.columns[selector.support_]

# Get the ranking of features (optional)
feature_ranking = selector.ranking_

print("Selected features:", selected_features)
print("\nFeature ranking (lower is better):")
print(feature_ranking)

###################################### 

results_vars=results.columns.values.tolist()
# X=[i for i in results_vars if i not in y]

#make y a list!
# y = list(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=56)


#####Timing END
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

#####################################!!!
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load your data and split it into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=56)

# Train the random forest classifier
clf = RandomForestClassifier(n_estimators=200, random_state=56)
clf.fit(X_train, y_train)

#### Feature Importance 
# Get the feature importance values
importance = clf.feature_importances_

# Create a dataframe to store the feature importance values and feature names
feature_importance = pd.DataFrame({"feature": X.columns, "importance": importance})

# Sort the dataframe by the importance values
feature_importance = feature_importance.sort_values(by="importance", ascending=False)

# Print the order of important features
print(feature_importance)


# Plot the feature importance values as a bar chart
plt.bar(range(feature_importance.shape[0]), feature_importance["importance"])
plt.xticks(range(feature_importance.shape[0]), feature_importance["feature"], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()


columns1 = ['COLUMN1', 'COLUMN3']


X=os_data_X[columns1]
y=os_data_y['FLAG']

##############  MODELS ##############!!!
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# Define a function to evaluate models
def evaluate_model(model, X_test, y_test):
    '''
 Evaluates a machine learning model on the test set using various performance metrics.

 Args:
     model (object): Trained machine learning model.
     X_test (array-like): Features of the test set. Shape (n_samples, n_features).
     y_test (array-like): True labels of the test set. Shape (n_samples,) or (n_samples, n_classes) for one-hot encoded labels.

 Returns:
     tuple: Contains the following evaluation metrics:
         - accuracy (float): Accuracy of the model.
         - precision (float): Precision of the model.
         - recall (float): Recall of the model.
         - f1 (float): F1 score of the model.
         - cm (array): Confusion matrix of the model's predictions.

 Example:
     >>> accuracy, precision, recall, f1, cm = evaluate_model(rf_model, X_test, y_test)
 '''
    y_pred = model.predict(X_test)
    if len(y_test.shape) > 1:  # For neural networks
        y_test = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, cm

def confu_matrix(y_test, y_pred):
    '''
    Args: 
        y_test (array): 1 line of binary outcomes
        y_pred (array): 1 line of of binary prediction outcomes
        
    Returns: 
        Colored confusion matrix with numbers and legend
    '''

    cm = confusion_matrix(y_test, y_pred)

    ###Visual 
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='ocean_r')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=56)

# Reshape to 1D array if necessary
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)


############# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_reg_results = evaluate_model(log_reg, X_test, y_test)
print("Logistic Regression Results:", log_reg_results)

# Predict probabilities
y_pred_prob = log_reg.predict_proba(X_test)[:, 1]

buildROC(y_test, y_pred_prob, 'Logistic Regression', 'b')

y_pred_lg = log_reg.predict(X_test)

confu_matrix(y_test, y_pred_lg)

############ Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=56)
rf.fit(X_train, y_train)
rf_results = evaluate_model(rf, X_test, y_test)
print("Random Forest Results:", rf_results)

# Predict probabilities
y_prob_rf = rf.predict_proba(X_test)[:, 1]
buildROC(y_test, y_prob_rf, 'Random Forest', 'g')

y_pred_rf = rf.predict(X_test)

confu_matrix(y_test, y_pred_rf)


################ XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
xgb_results = evaluate_model(xgb, X_test, y_test)
print("XGBoost Results:", xgb_results)

y_pred_prob_xgb = xgb.predict_proba(X_test)[:, 1]
buildROC(y_test, y_pred_prob_xgb, 'XGBoost', 'm')

y_pred_xgb = xgb.predict(X_test)

confu_matrix(y_test, y_pred_xgb)

################## K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_results = evaluate_model(knn, X_test, y_test)
print("K-Nearest Neighbors Results:", knn_results)

y_pred_prob_knn = knn.predict_proba(X_test)[:, 1]
buildROC(y_test, y_pred_prob_knn, 'KNN', 'c')

y_pred_knn = knn.predict(X_test)

confu_matrix(y_test, y_pred_knn)

################# Decision Tree
dt = DecisionTreeClassifier(random_state=56)
dt.fit(X_train, y_train)
dt_results = evaluate_model(dt, X_test, y_test)
print("Decision Tree Results:", dt_results)

y_pred_prob_dt = dt.predict_proba(X_test)[:, 1]
buildROC(y_test, y_pred_prob_dt, 'Decision Tree', 'y')

y_pred_dt = dt.predict(X_test)

confu_matrix(y_test, y_pred_dt)


############## Neural Network
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
nn = Sequential()
nn.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
nn.add(Dropout(0.2))
nn.add(Dense(32, activation='relu'))
nn.add(Dense(y_train_cat.shape[1], activation='softmax'))
nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
nn.fit(X_train, y_train_cat, epochs=50, batch_size=32, verbose=2)
nn_results = evaluate_model(nn, X_test, y_test_cat)
print("Neural Network Results:", nn_results)


# Predict probabilities for the test set
y_pred_prob_nn = nn.predict(X_test)

# For multi-class classification, we need to compute the ROC curve and AUC for each class separately
# Here, we simplify by considering the probability of the positive class for binary classification
if y_pred_prob_nn.shape[1] == 2:
    y_pred_prob_nn = y_pred_prob_nn[:, 1]  # Use the probability of the positive class
# Convert y_test to the correct format for roc_curve
y_test_binary = np.argmax(y_test_cat, axis=1) if y_test_cat.shape[1] > 1 else y_test

# Call buildROC with the true labels and predicted probabilities
buildROC(y_test_binary, y_pred_prob_nn, 'Neural Network', 'purple')

confu_matrix(y_test, y_pred_prob_nn)


############ Support Vector Classifier
# svc = SVC(probability=True)
# svc.fit(X_train, y_train)
# svc_results = evaluate_model(svc, X_test, y_test)
# print("Support Vector Classifier Results:", svc_results)

# y_pred_prob_svc = svc.predict_proba(X_test)[:, 1]
# buildROC(y_test, y_pred_prob_svc)



############### Gradient Boosting Classifier
gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbc.fit(X_train, y_train)
gbc_results = evaluate_model(gbc, X_test, y_test)
print("Gradient Boosting Classifier Results:", gbc_results, 'Gradient Boosting', 'orange')

y_pred_prob_gbc = gbc.predict_proba(X_test)[:, 1]
buildROC(y_test, y_pred_prob_gbc, 'Gradient Boosting', 'orange')

y_pred_gbc = gbc.predict(X_test)

confu_matrix(y_test, y_pred_gbc)


# Print confusion matrices
print("Confusion Matrix for Logistic Regression:\n", log_reg_results[4])
print("Confusion Matrix for Random Forest:\n", rf_results[4])
#print("Confusion Matrix for Support Vector Classifier:\n", svc_results[4])
print("Confusion Matrix for Gradient Boosting Classifier:\n", gbc_results[4])
print("Confusion Matrix for Neural Network:\n", nn_results[4])
print("Confusion Matrix for XGBoost:\n", xgb_results[4])
print("Confusion Matrix for KNN:\n", knn_results[4])
print("Confusion Matrix for Decision Tree:\n", dt_results[4])


plot_combined_roc(y_test, [
    (y_pred_prob, 'Logistic Regression', 'b'),
    (y_prob_rf, 'Random Forest', 'g'),
    ##(y_pred_prob_svc, 'SVC Model', 'lime'),
    (y_pred_prob_knn, 'KNN', 'c'),
    (y_pred_prob_xgb, 'XGBoost', 'm'),
    (y_pred_prob_dt, 'Decision Tree', 'y'),  
    (y_pred_prob_nn, 'Neural Network', 'purple'),
    (y_pred_prob_gbc, 'Gradient Boosting', 'orange')
])

             
close_all_connections()      
               
                          
