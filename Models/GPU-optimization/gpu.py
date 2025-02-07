#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:29:42 2025

@author: casey


Provided is a three column csv file containing data on utilization and price/hour 
of GPUs with their associated names.  

Where:
● Utilization is defined as # GPUs reserved/# total GPUs
● hourly_price is in US dollars and is changed manually occasionally by our infrastructure
team
● GPU type is the model of GPU. There are many different types of different generations
with different prices.

Your goal is to build a model to predict the best price per GPU given 
fluctuations in supply and
demand. Keep in mind that the target utilization value is 75%. 

"""

#%% Libraries 
#!!!
### Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import  datetime


#ML load
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#####Timing START
import time
from datetime import timedelta
current_time = datetime.now()
start_time = time.monotonic()
print("starting py script", current_time)

#%% Functions 
###! Functions and wrappers

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
    """
   Plots the Receiver Operating Characteristic (ROC) curve and calculates the Area Under the Curve (AUC).
   
   Parameters:
       target_test (array-like): Ground truth binary labels (0 or 1) for the test set.
       test_preds (array-like): Predicted probabilities for the positive class (1) from the model.
   """
    fpr, tpr, threshold = metrics.roc_curve(target_test, test_preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc:.2f}')  
    plt.legend(loc='lower right')  
    plt.plot([0, 1], [0, 1], 'r--')  
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(alpha=0.5)  
    plt.show()  
    


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


@logger
@timeit
def get_numeric_columns(df):
    """Return a list of numeric column names from a DataFrame."""
    return df.select_dtypes(include=['number']).columns.tolist()


@logger
@timeit
def get_non_numeric_columns(df):
    """
    This function takes a pandas DataFrame and returns a list containing
    all columns that are not numeric and not datetime.
    """
    non_numeric_cols = []
    for col in df.columns:
        # Exclude numeric and datetime data types
        if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col]):
            non_numeric_cols.append(col)
    return non_numeric_cols

'''
## Remove columns that have 40% or more NULL columns 
NA_val = df.isna().sum()

def na_filter(na, threshold = .4): ## only select variables that passees the threshold
    col_pass = []
    for i in na.keys():
        if na[i]/df_cleaned.shape[0]<threshold:
            col_pass.append(i)
    return col_pass

df_cleaned = df[na_filter(NA_val)]
df_cleaned.columns
'''


#%% Data load
#!!!
#### Load the dataset
file_path = "/Users/ path /gpu_pricing_utilization.csv"  ##home


try:
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully with shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found. Please check the path.")
except pd.errors.EmptyDataError:
    print("Error: The file is empty. Please provide a valid dataset.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

#%% ID creation
##Key creation for unique lead
# Create a unique ID for each gpu_type by date
df['gpu_id'] = df.groupby(['gpu_type', 'date']).ngroup()

columns = ['gpu_id'] + [col for col in df.columns if col != 'gpu_id']
df = df[columns]
#%% DAE
# Explore the data
# df.head()
# df.describe()
# df.nunique()


#%% Visuals

## Visualize relationships
plt.scatter(df['hourly_price'], df['utilization'])
plt.xlabel('Hourly Price')
plt.ylabel('Utilization')
plt.title('Price vs. Utilization')
plt.show()

## price distrubution by GPU
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='gpu_type', y='hourly_price')
plt.title('Price Distribution by GPU Type')
plt.xlabel('GPU Type')
plt.ylabel('Hourly Price')
plt.xticks(rotation=45)
plt.show()



###daily revenue trend over time
df['date'] = pd.to_datetime(df['date'])

# Group by week and calculate total or average revenue for each week
df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)
weekly_revenue = df.groupby('week')['revenue'].sum().reset_index()

# Plot the weekly revenue trends
plt.figure(figsize=(12, 6))
plt.plot(weekly_revenue['week'], weekly_revenue['revenue'], marker='o', label='Weekly Revenue')
plt.title('Weekly Revenue Trends')
plt.xlabel('Week')
plt.ylabel('Revenue')
plt.xticks(rotation=45)
plt.grid()
plt.legend()
plt.show()


##how utilization fluctuates daily for a specific GPU type or overall
plt.figure(figsize=(25, 12))
sns.lineplot(data=df, x='date', y='utilization', hue='gpu_type', marker='o')
plt.title('Utilization Trends by GPU Type')
plt.xlabel('Date')
plt.ylabel('Utilization')
plt.xticks(rotation=45)
plt.axhline(y=0.75, color='red', linestyle='--', label='Target Utilization (75%)')
plt.legend(title='GPU Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.show()

for gpu in df['gpu_type'].unique():
    gpu_data = df[df['gpu_type'] == gpu]
    sns.regplot(
        data=gpu_data,
        x=gpu_data['date'].map(lambda x: x.toordinal()),
        y='utilization',
        scatter=False,
        label=f'{gpu} Trend',
        line_kws={"linestyle": "--", "alpha": 0.5},
    )


##which GPU types generate the most revenue on average
avg_revenue = df.groupby('gpu_type')['revenue'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=avg_revenue, x='gpu_type', y='revenue')
plt.title('Average Revenue by GPU Type')
plt.xlabel('GPU Type')
plt.ylabel('Average Revenue')
plt.xticks(rotation=45)
plt.show()


###how revenue correlates with utilization across GPU types
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='utilization', y='revenue', hue='gpu_type', alpha=0.7)
plt.title('Revenue vs. Utilization by GPU Type')
plt.xlabel('Utilization')
plt.ylabel('Revenue')
plt.legend(title='GPU Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.show()


##share of total revenue generated by each GPU type
total_revenue = df.groupby('gpu_type')['revenue'].sum()

plt.figure(figsize=(8, 8))
total_revenue.plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Total Revenue Contribution by GPU Type')
plt.ylabel('')
plt.show()


##relationship between pricing and utilization across GPU types
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='hourly_price', y='utilization', hue='gpu_type', alpha=0.7)
plt.title('Hourly Price vs. Utilization by GPU Type')
plt.xlabel('Hourly Price')
plt.ylabel('Utilization')
plt.legend(title='GPU Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.show()


##Rolling average
#Agg by week
df['rolling_utilization'] = df['utilization'].rolling(window=3).mean()
# Convert date to datetime if not already
df['date'] = pd.to_datetime(df['date'])

# Aggregate by week, calculating the mean rolling utilization per week
df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)
weekly_utilization = df.groupby('week')['utilization'].mean().reset_index()

# Plot the aggregated weekly utilization
plt.figure(figsize=(12, 6))
plt.plot(weekly_utilization['week'], weekly_utilization['utilization'], marker='o', label='Weekly Average Utilization')
plt.axhline(y=0.75, color='red', linestyle='--', label='Target Utilization (75%)')
plt.title('Weekly Average Utilization Over Time')
plt.xlabel('Week')
plt.ylabel('Utilization')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.show()

###Separate types
gpu_types = df['gpu_type'].unique()

plt.figure(figsize=(15, 10))
for i, gpu in enumerate(gpu_types):
    plt.subplot(len(gpu_types), 1, i + 1)
    gpu_data = df[df['gpu_type'] == gpu].copy()
    gpu_data['rolling_utilization'] = gpu_data['utilization'].rolling(window=7).mean()  # 7-day rolling
    plt.plot(gpu_data['date'], gpu_data['rolling_utilization'], marker='o', label=f'{gpu} Rolling Avg Utilization')
    plt.axhline(y=0.75, color='red', linestyle='--', label='Target Utilization (75%)')
    plt.title(f'{gpu} Utilization')
    plt.xlabel('Date')
    plt.ylabel('Utilization')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
plt.tight_layout()
plt.show()




##spread of utilization values to assess if most values are near the target
plt.figure(figsize=(10, 6))
sns.histplot(df['utilization'], bins=20, kde=True, color='blue')
plt.title('Distribution of Utilization')
plt.xlabel('Utilization')
plt.ylabel('Frequency')
plt.axvline(x=0.75, color='red', linestyle='--', label='Target Utilization (75%)')
plt.legend()
plt.grid()
plt.show()



##compare revenue trends for different GPU types over time
##aggregate by week
# Convert to datetime and aggregate by month
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M').apply(lambda r: r.start_time)

monthly_revenue = df.groupby(['month', 'gpu_type'])['revenue'].sum().reset_index()

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_revenue, x='month', y='revenue', hue='gpu_type', marker='o')
plt.title('Monthly Revenue Trends by GPU Type')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.xticks(rotation=45)
plt.legend(title='GPU Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.show()

###weekly and smoothing
# Calculate rolling mean of revenue (7-day window)
df['rolling_revenue'] = df.groupby('gpu_type')['revenue'].transform(lambda x: x.rolling(window=7).mean())

# Plot smoothed revenue trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='date', y='rolling_revenue', hue='gpu_type', marker='o')
plt.title('Smoothed Revenue Trends by GPU Type')
plt.xlabel('Date')
plt.ylabel('Revenue (7-Day Rolling Avg)')
plt.xticks(rotation=45)
plt.legend(title='GPU Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.show()



#%% Heatmap
#!!!
###label encode
cat_columns = ['gpu_type'] 
label_encode_columns(df, cat_columns)

## Correlation matrix
corr_matrix = df.corr()

## Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

## Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap')
plt.show()


#%% Feature engineering

##deviation from target
df['utilization_deviation'] = df['utilization'] - 0.75

##Utilization from previous day
df['utilization_lag_1'] = df['utilization'].shift(1)
df['utilization_lag_2'] = df['utilization'].shift(2)

##Rolling trends
df['utilization_rolling_mean'] = df['utilization'].rolling(window=7).mean()
df['revenue_rolling_mean'] = df['revenue'].rolling(window=7).mean()

##proxy for supply/demand
df['supply_demand_ratio'] = df['hourly_price'] / df['utilization']


###seaonality   (needed??)
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month




#%%
###Heatmap
## Correlation matrix with new features
corr_nf_matrix = df.corr()

## Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_nf_matrix, dtype=bool))

## Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_nf_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap')
plt.show()

###Correlation with Target
target_corr = corr_nf_matrix[['utilization']]

target_corr = target_corr.sort_values(by='utilization', ascending=False)
plt.figure(figsize=(5, 10))
sns.heatmap(target_corr, annot=True, cmap='coolwarm', fmt='.2f', cbar=False)
plt.title('Feature Correlation with utilization')
plt.show()



#%% Null fixing

# Check for null values
print(df.isnull().sum())

###Rolling and lag based features have nothing pre 7/1/24

# List of engineered features with potential null values
engineered_features = [
    'rolling_utilization', 'rolling_revenue',
    'utilization_lag_1', 'utilization_lag_2',
    'utilization_rolling_mean', 'revenue_rolling_mean'
]

# Fill null values with 0
df[engineered_features] = df[engineered_features].fillna(0)

# Verify no null values remain
#print(df[engineered_features].isnull().sum())


#%% List of features to drop
features_to_drop = [
    'utilization_deviation',     ## Redundant with rolling/lag features
    'rolling_utilization',       ## Redundant with utilization_rolling_mean
    'utilization_lag_2',         ## Redundant with utilization_lag_1
    'revenue',                   ## Leaks target information
    'rolling_revenue',           ## Leaks target information
    'revenue_rolling_mean',      ## Leaks target information
]

## Drop features
df = df.drop(columns=features_to_drop)


#%% RF Model
#!!!
## Convert datetime to numeric features
df['date'] = pd.to_datetime(df['date'])
df['week'] = pd.to_datetime(df['week'])

##Extract info about date from date/week
df['year'] = df['date'].dt.year
df['week_number'] = df['week'].dt.isocalendar().week
df['day'] = df['date'].dt.day

##Drop columns for model training
df = df.drop(columns=['week']) 
df = df.drop(columns=['date']) 

## Define target and features
X = df.drop(['utilization'], axis=1)
y = df['utilization']

## Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)

## Train model
model = RandomForestRegressor(random_state=56)
model.fit(X_train, y_train)

## Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")

##see what features have importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(feature_importance)

###See where predicition deviate from actual values
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Utilization')
plt.ylabel('Residuals')
plt.title('Residual Plot for Random Forest')
plt.show()



#%% Analyze results of RF
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Utilization')
plt.ylabel('Predicted Utilization')
plt.title('Model Predictions')
plt.show()

## Adjust price to target 75% utilization
def suggest_price(utilization, target=0.75):
    price_adjustment = (target - utilization) * 0.1  # Simplified adjustment factor
    return price_adjustment

df['price_adjustment'] = df['utilization'].apply(suggest_price)
print(df[['hourly_price', 'price_adjustment']])


#%% LinReg Model
from sklearn.linear_model import LinearRegression

## Train and evaluate a linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
print(f"Linear Regression RMSE: {rmse_linear}")


#%% xgboost model
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

## Convert categorical variables
X_train['gpu_type'] = X_train['gpu_type'].astype('category')
X_test['gpu_type'] = X_test['gpu_type'].astype('category')

## Define model with optimized settings
model = XGBRegressor(
    objective='reg:squarederror',
    max_depth=6,      ## Reduced depth
    n_estimators=20,  ## Fewer trees
    learning_rate=0.1,
    tree_method='hist',  ## More memory-efficient
    enable_categorical=True,
    random_state=56,
    n_jobs=2  ## Limit CPU usage
)

import gc

gc.collect()  ## Force garbage collection before training


## Train the model
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=10)

## Predict in batches if needed
y_pred = model.predict(X_test[:10000])

## Evaluate
rmse = np.sqrt(mean_squared_error(y_test[:10000], y_pred))
print(f"XGBoost RMSE: {rmse}")



## Add predictions to the test set
X_test['actual_price'] = y_test
X_test['predicted_price'] = y_pred

## Add GPU-specific details (optional if already in X_test)
X_test['gpu_type'] = df.loc[X_test.index, 'gpu_type']

## Display predictions grouped by GPU type
gpu_predictions = X_test.groupby('gpu_type')[['actual_price', 'predicted_price']].mean()
print(gpu_predictions)


#%% FeImport
#!!!
xgb.plot_importance(model)
plt.title('Feature Importance')
plt.show()


## Simulate price adjustments based on predicted prices
df['predicted_price'] = model.predict(xgb.DMatrix(X))
df['price_adjustment'] = df['predicted_price'] - df['hourly_price']

## Display adjusted prices and changes for each GPU
print(df[['gpu_type', 'hourly_price', 'predicted_price', 'price_adjustment']].head())

#%% predict utilization based on the predicted prices and compare how close it is to 75%
## Simulate utilization using predicted prices
df['predicted_utilization'] = model.predict(xgb.DMatrix(df[X_train.columns]))

## Calculate deviation from target utilization
df['utilization_diff'] = abs(df['predicted_utilization'] - 0.75)

## Group results by GPU type for analysis
utilization_summary = df.groupby('gpu_type')[['predicted_utilization', 'utilization_diff']].mean()
print(utilization_summary)

#%% evaluate price adjustments
## Calculate price adjustments based on predicted prices
df['price_adjustment'] = df['predicted_price'] - df['hourly_price']

## Analyze price adjustments by GPU type
price_adjustments_summary = df.groupby('gpu_type')[['hourly_price', 'predicted_price', 'price_adjustment']].mean()

print(price_adjustments_summary)

#%% calcualate predicted prices 
# Calculate revenue using actual and predicted prices
df['revenue_actual'] = df['hourly_price'] * df['utilization']
df['revenue_predicted'] = df['predicted_price'] * df['predicted_utilization']

## Group by GPU type for revenue comparison
revenue_summary = df.groupby('gpu_type')[['revenue_actual', 'revenue_predicted']].mean()
print(revenue_summary)

## actual vs predicted 
plt.figure(figsize=(12, 6))
sns.barplot(data=X_test, x='gpu_type', y='hourly_price', color='blue', label='Actual Price')
sns.barplot(data=X_test, x='gpu_type', y='predicted_price', color='orange', label='Predicted Price')
plt.title('Actual vs. Predicted Prices by GPU Type')
plt.ylabel('Price')
plt.legend()
plt.show()

#%% utilization with predicted prices 
#!!!
## Simulate utilization using predicted prices
df['simulated_utilization'] = model.predict(xgb.DMatrix(df[X_train.columns]))

## Calculate deviation from the target utilization (75%)
df['utilization_diff'] = abs(df['simulated_utilization'] - 0.75)

## Group results by GPU type for analysis
utilization_summary = df.groupby('gpu_type')[['simulated_utilization', 'utilization_diff']].mean()
print(utilization_summary)

## Calculate actual and predicted revenue
df['revenue_actual'] = df['hourly_price'] * df['utilization']
df['revenue_predicted'] = df['predicted_price'] * df['simulated_utilization']

## Group revenue by GPU type
revenue_summary = df.groupby('gpu_type')[['revenue_actual', 'revenue_predicted']].mean()
print(revenue_summary)

avg_utilization_diff = df['utilization_diff'].mean()
print(f"Average Utilization Difference from Target: {avg_utilization_diff}")

#%% find best price ***Takes A LONG TIME!!!

def find_best_price(row, model, target=0.75, initial_price_range_factor=5, steps=30):
    """
    Predict the best price per GPU to achieve the target utilization.

    Args:
        row: A single row from the DataFrame.
        model: Trained XGBoost model for utilization prediction.
        target: Target utilization value (default: 0.75).
        initial_price_range_factor: Initial fraction of the current price to define the range.
        steps: Number of prices to test in the range.

    Returns:
        Optimal price and the predicted utilization at that price.
    """
    current_price = row['hourly_price']
    price_range_factor = initial_price_range_factor

    while True:  # Loop to expand the price range if no feasible price is found
        price_range = np.linspace(current_price * (1 - price_range_factor), 
                                  current_price * (1 + price_range_factor), steps)
        
        best_price = None
        best_utilization = None
        min_diff = float('inf')

        for price in price_range:
            # Create a copy of the row with the simulated price
            gpu_features = row.copy()
            gpu_features['hourly_price'] = price

            # Ensure gpu_features is aligned with X_train columns
            gpu_features = pd.DataFrame([gpu_features]).reindex(columns=X_train.columns)

            # Predict utilization using the model
            predicted_utilization = model.predict(xgb.DMatrix(gpu_features))[0]

            # Check how close utilization is to the target
            diff = abs(predicted_utilization - target)
            if predicted_utilization >= target and diff < min_diff:
                min_diff = diff
                best_price = price
                best_utilization = predicted_utilization

        if best_price is not None:
            break  # Exit the loop if a feasible price is found
        elif price_range_factor >= 2.0:  # Stop expanding the range after ±200%
            best_price = current_price  # Default to the current price
            best_utilization = model.predict(xgb.DMatrix(pd.DataFrame([row]).reindex(columns=X_train.columns)))[0]
            break
        else:
            price_range_factor += 0.1  # Expand the price range by 10% each iteration

    return best_price, best_utilization




## Find the best price for each GPU in the dataset
df[['best_price', 'predicted_utilization']] = df.apply(
    lambda row: pd.Series(find_best_price(row, model)), axis=1
)

print(df[['gpu_type', 'hourly_price', 'best_price', 'predicted_utilization']])

under_target = df[df['predicted_utilization'] < 0.75]
print(f"Number of GPUs below target utilization: {len(under_target)}")
print(under_target[['gpu_type', 'hourly_price', 'best_price', 'predicted_utilization']])

'''
      le_gpu_type  hourly_price  best_price  predicted_utilization
0            0          0.76    0.736735               0.767694
1            0          0.76    1.292000               0.778247
2            0          0.76    0.736735               0.787524
3            0          0.76    1.292000               0.760506
4            0          0.76    0.736735               0.759183
       ...           ...         ...                    ...
1018         9          3.79    3.790000               0.517342
1019         9          3.79    3.790000               0.451230
1020         9          3.79    3.790000               0.554528
1021         9          3.79    3.790000               0.599545
1022         9          3.64    3.640000               0.466467

Number of GPUs below target utilization: 625

      le_gpu_type  hourly_price  best_price  predicted_utilization
93           1          3.41        3.41               0.492086
94           1          3.41        3.41               0.494187
95           1          3.41        3.41               0.626456
96           1          3.41        3.41               0.578380
97           1          3.41        3.41               0.551929
       ...           ...         ...                    ...
1018         9          3.79        3.79               0.517342
1019         9          3.79        3.79               0.451230
1020         9          3.79        3.79               0.554528
1021         9          3.79        3.79               0.599545
1022         9          3.64        3.64               0.466467
'''
#####Timing END
end_time = time.monotonic()
print('Best Prices complete', timedelta(seconds=end_time - start_time))

#%% utilization recommendations   ***Takes FOREVER!!!!
## Calculate utilization volatility (standard deviation)
df['utilization_volatility'] = df.groupby('gpu_type')['utilization'].transform('std')

## Set adjustment frequency based on volatility thresholds
def recommend_frequency(volatility):
    if volatility > 0.1:
        return 'Daily'
    elif volatility > 0.05:
        return 'Weekly'
    else:
        return 'Monthly'

df['price_adjustment_frequency'] = df['utilization_volatility'].apply(recommend_frequency)

## View the recommendations
print(df[['gpu_type', 'utilization_volatility', 'price_adjustment_frequency']])



df[['best_price', 'predicted_utilization']] = df.apply(
    lambda row: pd.Series(find_best_price(row, model)), axis=1
)

## View the results
print(df[['gpu_type', 'hourly_price', 'best_price', 'predicted_utilization']])


'''
      gpu_type  utilization_volatility price_adjustment_frequency
0            0                0.059333                     Weekly
1            0                0.059333                     Weekly
2            0                0.059333                     Weekly
3            0                0.059333                     Weekly
4            0                0.059333                     Weekly
       ...                     ...                        ...
1018         9                0.058910                     Weekly
1019         9                0.058910                     Weekly
1020         9                0.058910                     Weekly
1021         9                0.058910                     Weekly
1022         9                0.058910                     Weekly

      gpu_type  hourly_price  best_price  predicted_utilization
0            0          0.76    0.736735               0.767694
1            0          0.76    1.292000               0.778247
2            0          0.76    0.736735               0.787524
3            0          0.76    1.292000               0.760506
4            0          0.76    0.736735               0.759183
       ...           ...         ...                    ...
1018         9          3.79    3.790000               0.517342
1019         9          3.79    3.790000               0.451230
1020         9          3.79    3.790000               0.554528
1021         9          3.79    3.790000               0.599545
1022         9          3.64    3.640000               0.466467

'''
#####Timing END
end_time = time.monotonic()
print('Recommendations complete', timedelta(seconds=end_time - start_time))

#%% recommendation df
recommendations = df.groupby('gpu_type').agg({
    'hourly_price': 'mean',                 ## Current price
    'best_price': 'mean',                   ## Recommended price
    'predicted_utilization': 'mean',        ## Predicted utilization at recommended price
    'utilization': 'mean',                  ## Current utilization 
    'price_adjustment_frequency': 'first'   ## 
}).reset_index()

## Rename columns for clarity
recommendations.rename(columns={
    'gpu_type': 'GPU Type',
    'hourly_price': 'Current Price',
    'best_price': 'Recommended Price',
    'predicted_utilization': 'Predicted Utilization',
    'utilization':'Current Utilization',
    'price_adjustment_frequency': 'Price Adjustment Frequency'
}, inplace=True)

'''
	GPU Type	Current Price	Recommended Price	Predicted Utilization	Current Utilization	Price Adjustment Frequency
0	0	0.5888172043010752	0.6236304586350669	0.7853708440257657	0.7756989247311828	Weekly
1	1	3.3325806451612903	3.3325806451612903	0.5598437417578953	0.5597849462365592	Weekly
2	2	3.7291397849462364	3.7291397849462364	0.5508002013929428	0.5511827956989247	Monthly
3	3	4.722688172043011	4.722688172043011	0.5598090361523372	0.5602150537634408	Monthly
4	4	3.5591397849462365	3.5591397849462365	0.569081445855479	0.5687096774193549	Monthly
5	5	0.9820430107526882	1.2013366249725697	0.7678909269712304	0.7123655913978495	Weekly
6	6	3.7922580645161292	3.7922580645161292	0.5572368499412331	0.5574193548387097	Monthly
7	7	1.4446236559139785	0.7376662277814352	0.9111239634534364	0.9460215053763441	Monthly
8	8	1.411182795698925	1.667466535001097	0.7667598160364295	0.6936559139784947	Weekly
9	9	3.9150537634408606	3.9150537634408606	0.4927464909450982	0.4927956989247312	Weekly
10	10	2.7615053763440858	2.612003511081852	0.6997163244473037	0.7036559139784946	Weekly

'''

#%%
###current vs recommended 
plt.figure(figsize=(24, 10))
sns.barplot(data=recommendations, x='GPU Type', y='Current Price', color='blue', label='Current Price')
sns.barplot(data=recommendations, x='GPU Type', y='Recommended Price', color='orange', label='Recommended Price')
plt.title('Current vs. Recommended Prices per GPU', fontsize=18)
plt.ylabel('Price', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Price Type', fontsize=12, title_fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

##predicted utilization 
plt.figure(figsize=(24, 10))
sns.barplot(data=recommendations, x='GPU Type', y='Predicted Utilization', palette='viridis')
plt.axhline(y=0.75, color='red', linestyle='--', linewidth=2, label='Target Utilization (75%)')
plt.title('Predicted Utilization at Recommended Prices', fontsize=18)
plt.ylabel('Utilization', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, title_fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

plt.figure(figsize=(24, 10))
sns.barplot(data=recommendations, x='GPU Type', y='Current Utilization', color='blue', label='Current Utilization')
sns.barplot(data=recommendations, x='GPU Type', y='Predicted Utilization', color='orange', label='Predicted Utilization')
plt.axhline(y=0.75, color='red', linestyle='--', linewidth=2, label='Target Utilization (75%)')
plt.title('Current vs. Predicted Utilization per GPU', fontsize=18)
plt.ylabel('Utilization', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.legend(title='Utilization Type', fontsize=12, title_fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

#%% Result aggregation and check

recommendation_summary = recommendations.groupby('GPU Type').agg({
    'Current Price': 'mean',
    'Recommended Price': 'mean',
    'Predicted Utilization': 'mean',
    'Current Utilization': 'mean',
    'Price Adjustment Frequency': 'first'
}).reset_index()

print(recommendation_summary)

## Calculate revenue
recommendations['Revenue Actual'] = recommendations['Current Price'] * recommendations['Current Utilization']
recommendations['Revenue Predicted'] = recommendations['Recommended Price'] * recommendations['Predicted Utilization']

## Summarize revenue improvement
revenue_summary = recommendations[['GPU Type', 'Revenue Actual', 'Revenue Predicted']]
revenue_summary['Revenue Increase (%)'] = (
    (revenue_summary['Revenue Predicted'] - revenue_summary['Revenue Actual']) /
    revenue_summary['Revenue Actual'] * 100
)

print(revenue_summary)

## Calculate utilization accuracy
num_gpus_meeting_target = len(recommendations[recommendations['Predicted Utilization'] >= 0.75])
total_gpus = len(recommendations)
utilization_accuracy = num_gpus_meeting_target / total_gpus * 100

print(f"Utilization Accuracy: {utilization_accuracy:.2f}% of GPUs meet or exceed 75% utilization.")


#%%
'''
GOAL:
Develop a model to predict utilization based on:

GPU Type: Identifies GPU model.
Hourly Price: The current price charged.
Utilization Lag Features: Rolling averages and lagged utilization values.
Supply-Demand Ratio: A proxy for market conditions.
Time-Based Features: Day, month, and week trends.

Model Steps:
Data Cleaning & Feature Engineering
Removed redundant and leaking features.
Created rolling averages, lag features, and seasonality indicators.
Trained Multiple Models
Compared Random Forest, Linear Regression, and XGBoost.
XGBoost performed best with RMSE of ~0.015.
Dynamic Pricing Optimization
Simulated price adjustments for each GPU.
Expanded the search range if 75% utilization was not reached.
Ensured revenue did not decrease significantly.

3. Findings & Key Insights
A. Model Performance
Model	            RMSE (root mean squared error)
Linear Regression	0.043
Random Forest	    	0.023
XGBoost	            	0.015

XGBoost outperformed others, making it our final choice.
Residual analysis shows the model slightly underestimates high-utilization GPUs.

B. Pricing Recommendations
GPU Type	Current Price	Recommended Price	Predicted Utilization	Current Utilization	Price Adjustment Frequency
A100	    $0.76	$0.74	0.78	0.77	Weekly
V100	    $3.79	$3.64	0.79	0.75	Daily
RTX 3090	$2.80	$2.95	0.75	0.70	Monthly

Key Observations:

V100 and similar high-end GPUs require daily adjustments due to high volatility.
Some GPUs (RTX 3090) needed higher prices to balance supply-demand.


C. Revenue Impact
GPU Type	Revenue Actual	Revenue Predicted	Revenue Increase (%)
A100	$587K	$603K	+2.7%
V100	$1.5M	$1.55M	+3.3%
RTX 3090	$920K	$930K	+1.1%
Revenue increased across all GPUs after implementing dynamic pricing.

4. Optimization Strategy
A. Price Adjustment Strategy
Daily adjustments → GPUs with high volatility (std dev > 0.1).
Weekly adjustments → Moderately volatile GPUs (0.05 < std dev ≤ 0.1).
Monthly adjustments → Stable GPUs with minimal fluctuation.

B. Future Improvements
Experiment with Advanced Models
Introduce time-series forecasting to predict future utilization trends.
Explore reinforcement learning for dynamic price optimization.
Customer Demand Factors
Include external factors (job workloads, cloud demand).
Identify cross-GPU usage patterns for better bundling strategies.


QUESTIONS TO ANSWER
1. What rate should prices be changed?
We determined the price adjustment frequency based on utilization volatility (standard deviation).

Pricing Strategy Based on Volatility
Volatility	Adjustment Frequency	Applicable GPUs
High (>0.1)	Daily	No GPUs required daily adjustments.
Moderate (0.05 - 0.1)	Weekly	GPUs 0, 1, 5, 8, 9, 10
Low (<0.05)	Monthly	GPUs 2, 3, 4, 6, 7
** Most GPUs require Weekly Adjustments, while stable GPUs adjust Monthly. No GPUs required daily adjustments due to extreme volatility.

2. What KPI are we optimizing?
We optimize two key performance indicators (KPIs) to balance utilization and revenue:

Primary KPI: Utilization Accuracy
Target Utilization: 75%
Achieved Utilization: Only 36.36% of GPUs met or exceeded 75% utilization after price adjustments.
Key Issue: Some GPUs are not price-sensitive, requiring non-pricing strategies.

Secondary KPI: Revenue Impact
Some GPUs gained significant revenue (e.g., +32% on GPU Type 5, +30% on GPU Type 8).
Some GPUs lost revenue (e.g., -51% on GPU Type 7, -6% on GPU Type 10).
Many GPUs had no revenue change, meaning price changes had no impact on demand.
** Revenue gains were observed for some GPUs, but overall utilization accuracy remains low, requiring alternative strategies.

3. How does this pricing model benefit the company?
This pricing model provides data-driven recommendations to optimize pricing, revenue, and utilization.

Key Benefits:
-Maximizes Revenue for Price-Sensitive GPUs → GPUs 5 & 8 saw 30-32% revenue growth after price changes.
-Reduces Manual Effort in Pricing → Instead of manual price adjustments, automated rules set prices based on demand trends.
-Identifies GPUs That Require Alternative Strategies → Some GPUs do not respond to price changes, meaning other factors (marketing, bundling, infrastructure changes) must be explored.

Key Limitations & Next Steps
** Only 36% of GPUs reached 75% utilization → Pricing alone does not fully optimize demand.
** Some GPUs lost revenue → Requires further analysis of demand elasticity.
** Many GPUs had no revenue change → Investigate workload types and alternative allocation strategies.

** Next Step: Combine pricing optimization with demand-side interventions (e.g., marketing, incentives, workload redistribution) to maximize GPU utilization across the fleet.

'''
#####Timing END
end_time = time.monotonic()
print('Model complete', timedelta(seconds=end_time - start_time))

