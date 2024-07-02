# -*- coding: utf-8 -*-
"""
Created on Thu Jun 1 09:31:06 2024

@author: cpalmisano
"""

import pandas as pd
from scipy import stats
import numpy as np
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
    print('connections closed')        
                


####Connection 
def connect_to_sql_server(server_name, database_name):
    # connect via pyodbc
    conn = pyodbc.connect(f'Driver={{SQL Server}};Server={server_name};Database={database_name};Trusted_Connection=yes;')
    
    # connect via sqlalchemy
    con = sqlalchemy.create_engine(f'mssql://{server_name}/{database_name}?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server', fast_executemany=True)
    
    return conn, con


# connection to server -> SERVER   db -> DATABASE
conn, con = connect_to_sql_server('SERVER', 'DATABASE')
cursor = conn.cursor()
conn.autocommit = True 


#### Data Ingestion 
### over 15.4 mil from 2022 forward 
sql = pd.read_sql('''
                  select top 300000 * from DATABASE.claimsdata
                  where YEAR(servicestartdate) >= 2023  --2022 
                  ''', con)


### Label Encoding function 
from sklearn.preprocessing import LabelEncoder

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
non_numeric_cols = get_non_numeric_columns(sql)

label_encode_columns(sql, non_numeric_cols)



# Select relevant attributes for analysis
numerical_columns = sql.select_dtypes(include=[np.number])  # Select numerical columns

# Calculate correlation matrix
correlation_matrix = numerical_columns.corr(method='pearson')  # Pearson correlation

# # Print the correlation matrix
# print("Correlation Matrix:")
# print(correlation_matrix)


## Removal of columns with Known grouping / clustering 
df_fil = sql.drop(columns = ['X', 'Y', 'Z'], axis=1)


numeric_columns = df_fil.select_dtypes(include=[np.number])  # Select numerical columns

# Calculate correlation matrix
corfil_matrix = numeric_columns.corr(method='pearson')  # Pearson correlation

####Vizuals
#############################################################!!!

import matplotlib.pyplot as plt 
plt.rc("font", size=8)
import seaborn as sns

#####  Heatmap
corr_matrix = df_fil.corr().abs()
mask = np.triu(np.ones_like(corr_matrix))
sns.heatmap(corr_matrix, cmap='ocean_r',  yticklabels=True, annot=False, mask=mask)


mask = np.triu(np.ones_like(corfil_matrix))
sns.heatmap(corfil_matrix, cmap='ocean_r',  yticklabels=True, annot=False, mask=mask)
full_matrix = corfil_matrix.dropna(axis=1, how='all')

full_matrix = full_matrix.dropna()

##############################################
# Minimum correlation threshold 
corr_threshold = 0.12

# Calculate correlation matrix
corr_matrix = df_fil.corr().abs()

# Filter low correlated features (below threshold)
filtered_cols = corr_matrix.sum(axis=0) >= (len(corr_matrix) - 1) * corr_threshold  # Check if any correlation >= threshold
filtered_corr = corr_matrix[filtered_cols]  # Filter rows based on filtered columns
filtered_corr = filtered_corr.loc[:, filtered_corr.columns >= corr_matrix.columns[0]]  # Filter for upper triangle

# Get columns to keep (all with at least one correlation above threshold)
cols_to_keep = filtered_corr.columns.tolist()

# Filter the original DataFrame
df_filtered = df_fil[cols_to_keep]

filtered_corr = filtered_corr.dropna(axis=1)

# Create mask to hide non-significant correlations (optional)
mask = np.triu(np.ones_like(filtered_corr))

# Generate the heatmap
# sns.heatmap(filtered_corr, cmap='ocean_r', yticklabels='auto', annot=False, mask=mask)
# plt.show()

# Create heatmap with annotations
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(filtered_corr, annot=False, ax=ax, cmap="ocean_r")
plt.title("Correlation Heatmap")
plt.show()


corr = filtered_corr

# Set threshold and formatting parameters
thresh = 0.12  # Correlation threshold for annotations
font_size_pos = 12  # Font size for positive correlations
font_size_neg = 10  # Font size for negative correlations
color_pos = "black"  # Text color for positive correlations
color_neg = "red"    # Text color for negative correlations

# Create heatmap 
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=False, ax=ax, cmap="coolwarm", fmt=".2f",
            vmin=-1, vmax=1, cbar=False)  # Adjust vmin/vmax based on your data

# # Annotate only above threshold 
# mask = np.abs(corr) >= thresh
# for (i, j), z in np.ndenumerate(corr):
#     if mask[i, j]:
#         ax.text(j, i, f"{z:.2f}", ha="center", va="center", fontsize=font_size_pos if z > 0 else font_size_neg,
#                 color=color_pos if z > 0 else color_neg)

# plt.title("Correlation Heatmap (|r| >= 0.12)")
# plt.show()

#######################################

#fig, ax = plt.subplots(figsize=(50, 25))
sns.clustermap(full_matrix, method='centroid', metric='euclidean', cmap="ocean_r", annot=False, fmt=".2f")
plt.title("Clustered Correlation Heatmap (centroid)")
plt.show()


f_thresh = .5  # Correlation threshold 
full_corr_pairs = []
for col1 in corr_matrix.columns:
  for col2 in corr_matrix.columns:
    if col1 != col2 and corr_matrix.loc[col1, col2] >= f_thresh:
      correlation = corr_matrix.loc[col1, col2]
      corr_percentage = f"{correlation}"  # Format as percentage
      full_corr_pairs.append((col1, col2, corr_percentage))
      
full_corr_df = pd.DataFrame(full_corr_pairs, columns=['Attribute 1', 'Attribute 2', 'Correlation'])


#################################################!!!

from sklearn.cluster import KMeans  # Example using KMeans
import sklearn 
from sklearn.metrics import silhouette_score
import os

# Set the environment variable to avoid potential memory leaks (optional)
os.environ['OMP_NUM_THREADS'] = '1'  # Limit MKL threads on Windows

def elbow_method(data, min_clusters=2, max_clusters=50):
    """
    This function implements the elbow method to estimate the optimal number of clusters (k).

    Args:
        data: A NumPy array representing the data to be clustered.
        min_clusters: Minimum number of clusters to explore (inclusive).
        max_clusters: Maximum number of clusters to explore (inclusive).

    Returns:
        The estimated number of clusters (k) based on the elbow method.
    """
    # Range of potential values for k
    k_range = range(min_clusters, max_clusters + 1)

    # List to store silhouette scores for each k
    silhouette_scores = []

    for k in k_range:
        # Perform KMeans clustering with current k
        kmeans = KMeans(n_clusters=k, random_state=56)  # Set random state for reproducibility
        kmeans.fit(data)

        # Check if silhouette score calculation is valid (at least 2 clusters)
        if k > 1:
            silhouette_score_val = silhouette_score(data, kmeans.labels_)
        else:
            silhouette_score_val = 0  # Assign a placeholder value for k=1

        silhouette_scores.append(silhouette_score_val)

    # Plot the elbow curve (optional)
    # ... (plot the elbow curve as before)

    # Identify the "elbow" point (maximum rate of decrease in silhouette score)
    elbow_point = np.argmax(np.diff(silhouette_scores)) + min_clusters

    # Return the estimated number of clusters
    return elbow_point

# Estimate optimal number of clusters
n_clusters = elbow_method(full_matrix)  
print("Estimated number of clusters:", n_clusters)


# Get clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=56)  
kmeans.fit(full_matrix)
cluster_labels = kmeans.labels_

# Create DataFrame with cluster information
df = pd.DataFrame(full_matrix, index=full_matrix.index, columns=full_matrix.columns)


# Define a mask to exclude the 'cluster' column from the heatmap
mask = np.zeros_like(df.iloc[:, :-1])  # Create a mask with same dimensions

# Get triangular indices for the upper triangle (excluding diagonal)
upper_triu_indices = np.triu_indices_from(mask, k=1)

# Set mask values to True for the upper triangle entries
mask[upper_triu_indices] = True

# Create heatmap with masking
sns.heatmap(df.iloc[:, :-1], cmap="ocean_r", annot=False, fmt=".2f", mask=mask)
plt.title('Heatmap with KMeans (Masked)')
plt.show()

# Add cluster labels to DataFrame
df['cluster'] = cluster_labels  
#df.to_csv('C:/ *PATH* /correlation.csv')

#########
##visualize the clustering 
clustermap = sns.clustermap(full_matrix, method='ward', metric='euclidean', cmap="ocean_r", annot=False, fmt=".2f")
plt.title(f"Clustermap with Ordered Rows and Columns (Estimated Clusters: {n_clusters})", pad=20)

# Extract data and cluster labels
data = clustermap.data
row_clusters = clustermap.dendrogram_row.reordered_ind
col_clusters = clustermap.dendrogram_col.reordered_ind


# Create a DataFrame 
df = pd.DataFrame(data, index=full_matrix.index[row_clusters], columns=full_matrix.columns[col_clusters])


##################!!!
# Analyze the results
# Look for high correlation coefficients between attributes.
# These may indicate relationships that could be reflected in your schema design.

# Find attribute pairs with correlation > threshold
thresh = 0.8  # Correlation threshold 
high_corr_pairs = []
for col1 in corr_matrix.columns:
  for col2 in corr_matrix.columns:
    if col1 != col2 and corr_matrix.loc[col1, col2] >= thresh:
      correlation = corr_matrix.loc[col1, col2]
      corr_percentage = f"{correlation}"  # Format as percentage
      high_corr_pairs.append((col1, col2, corr_percentage))
      
# Create DataFrame from list of pairs
high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Attribute 1', 'Attribute 2', 'Correlation'])

high_corr_df.nunique()


####################!!!
##dataframe with Row/Column Ordering

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
os.environ["OMP_NUM_THREADS"] = "1"


# Perform hierarchical clustering to get row and column order
row_linkage = linkage(full_matrix, method='ward', metric='euclidean')
col_linkage = linkage(full_matrix.T, method='ward', metric='euclidean')

row_clusters = fcluster(row_linkage, t=4, criterion='maxclust')
col_clusters = fcluster(col_linkage, t=4, criterion='maxclust')

# Create a new DataFrame with ordered rows and columns based on cluster labels 
df_ordered = full_matrix.iloc[dendrogram(row_linkage, no_plot=True)['leaves'], 
                              dendrogram(col_linkage, no_plot=True)['leaves']]

# Generate clustermap with heatmap 
sns.clustermap(df_ordered, method='ward', metric='euclidean', cmap="ocean_r", annot=False, fmt=".2f")
plt.title(f"Clustermap with Ordered Rows and Columns (Estimated Clusters: {n_ordered_clusters})", pad=20)


# Estimate optimal number of clusters using the elbow method
n_ordered_clusters = elbow_method(df_ordered.values)
print("Estimated number of clusters:", n_ordered_clusters)

# Apply KMeans clustering with the estimated number of clusters
kmeans = KMeans(n_clusters=n_ordered_clusters, random_state=56)
kmeans.fit(df_ordered.values)

# Add the KMeans cluster labels to the DataFrame
df_ordered['cluster_name'] = kmeans.labels_

#close connections
close_all_connections()
