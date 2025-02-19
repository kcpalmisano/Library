# -*- coding: utf-8 -*-
""" 
Created on Mon Jun 12 14:41:41 2023

@author: cpalmisano
"""

import numpy as np
import pandas as pd
import gc
import inspect
from datetime import  date
import pyodbc
import sqlalchemy
from sqlalchemy.orm import sessionmaker 

import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

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

import math

def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    radius = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = radius * c

    return distance
###------------------------------------------------------ #!!!

# connection to server -> SERVER   db -> DATABASE
conn, con = connect_to_sql_server('SERVER', 'DATABASE')
cursor = conn.cursor()
conn.autocommit = True 

con = sqlalchemy.create_engine('mssql://SERVER/DATABASE?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server', fast_executemany=True)    
Session = sessionmaker(bind=con)
session = Session()   

# SQL query to fetch data from the database
manh_5stars = pd.read_sql("""
    SELECT DISTINCT latitude, longitude, address1, address2, city, state, fs.five_star_name, practicename, Centername, ZipCode as zip
    FROM dbo.FIVE_STAR_CENTERS fs
    LEFT JOIN main.FiveStarCenter fc ON fc.taxid = fs.prov_tax_id
    WHERE end_dte IS NULL
        AND (latitude IS NOT NULL OR longitude IS NOT NULL)
        AND Latitude < 40.9
        AND Longitude < -73.6
        AND Longitude > -74.046
        AND state = 'NY'
        AND IsRecruitment = 1
    ORDER BY five_star_name, PracticeName
""", con)

manh_5stars['cluster'] = manh_5stars.index


# Define the cluster centers
fivestar_fac = manh_5stars[['latitude', 'longitude', 'practicename', 'five_star_name', 'Centername', 'state', 'city', 'zip']].reset_index()
fivestar_fac = fivestar_fac.rename(columns={'index': 'cluster'})
lalo_fivestar = fivestar_fac.rename(columns={'index': 'cluster'})
#lalo_fivestar.drop(['practicename', 'Centername'], axis=1, inplace=True)

stat_5stars = pd.read_sql('''SELECT DISTINCT latitude, longitude, address1, address2, city, state, fs.five_star_name, practicename, Centername, ZipCode as zip
FROM dbo.FIVE_STAR_CENTERS fs
LEFT JOIN main.FiveStarCenter fc ON fc.taxid = fs.prov_tax_id
WHERE end_dte IS NULL
      AND (latitude IS NOT NULL OR longitude IS NOT NULL)
        AND state = 'NY'
        AND IsRecruitment = 1
		and city LIKE '%staten%'
    ORDER BY five_star_name, PracticeName
    ''', con)

stat_5stars['cluster'] = stat_5stars.index    


# Fetch member information for clustering
members = pd.read_sql("""
    SELECT DISTINCT Person_id, RELATION_CODE AS relation, LAST_NAME + ', ' + FIRST_NAME AS Name, 
        BIRTH_DATE, sex, ADDRESS_1,  ADDRESS_2, city, state, zip, 
        ADDRESS_LATITUDE AS latitude, ADDRESS_LONGITUDE AS longitude
            FROM Main.EligMedical
                WHERE STOP_DATE >= GETDATE()
                AND state = 'NY'
                AND ADDRESS_LATITUDE < 40.9
                AND ADDRESS_LONGITUDE < -73.6
                AND ADDRESS_LONGITUDE > -74.046
""", con)

# Clean up DOB to age
def age(birthdate):
  if pd.isnull(birthdate):
    return None
  else:
    today = date.today()
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    return age

# Apply the function to calculate age from birthdate
members['BIRTH_DATE'] = members['BIRTH_DATE'].apply(age)

# Rename the field
members = members.rename(columns={'BIRTH_DATE': 'Patient_Age'})

# Fill the missing values with an empty string
members['ADDRESS_1'] = members['ADDRESS_1'].fillna('')
members['ADDRESS_2'] = members['ADDRESS_2'].fillna('')
members['city'] = members['city'].fillna('')
members['state'] = members['state'].fillna('')
members['zip'] = members['zip'].fillna('')

# Combine address components into a single column
members['address'] = members['ADDRESS_1'] + ' ' + members['ADDRESS_2'] + ' ' + members['city'] + ' ' + members['state'] + ' ' + members['zip']

# Label encode columns for features
label_encode_columns(members, ['sex', 'city', 'state'])

#####  Staten island HOME #############
staten = pd.read_sql('''
 SELECT DISTINCT Person_id, RELATION_CODE AS relation, LAST_NAME + ', ' + FIRST_NAME AS Name, 
    BIRTH_DATE, sex, ADDRESS_1,  ADDRESS_2, city, state, zip, 
    ADDRESS_LATITUDE AS latitude, ADDRESS_LONGITUDE AS longitude
	    FROM Main.EligMedical
        where city LIKE '%Staten%'
        AND (ADDRESS_LATITUDE IS NOT NULL OR ADDRESS_LONGITUDE IS NOT NULL)
        and STOP_DATE >= GETDATE()
''', con) 

# Apply the function to calculate age from birthdate
staten['BIRTH_DATE'] = staten['BIRTH_DATE'].apply(age)

# Rename the field
staten = staten.rename(columns={'BIRTH_DATE': 'Patient_Age'})

# Fill the missing values with an empty string
staten['ADDRESS_1'] = staten['ADDRESS_1'].fillna('')
staten['ADDRESS_2'] = staten['ADDRESS_2'].fillna('')
staten['city'] = staten['city'].fillna('')
staten['state'] = staten['state'].fillna('')
staten['zip'] = staten['zip'].fillna('')

# Combine address components into a single column
staten['address'] = staten['ADDRESS_1'] + ' ' + staten['ADDRESS_2'] + ' ' + staten['city'] + ' ' + staten['state'] + ' ' + staten['zip']

# Label encode columns for features
label_encode_columns(staten, ['sex', 'city', 'state'])

#####  Staten island WORK #############  
wkstaten = pd.read_sql('''
	 SELECT DISTINCT Person_id, RELATION_CODE AS relation, LAST_NAME + ', ' + FIRST_NAME AS Name, 
    BIRTH_DATE, sex, ADDRESS_1,  ADDRESS_2, WORK_LOCATION_CITY as city, state, WORK_LOCATION_ZIP as zip, 
    WORK_LOCATION_LATITUDE AS latitude, WORK_LOCATION_LONGITUDE AS longitude
	    FROM Main.EligMedical
        where WORK_LOCATION_CITY LIKE '%Staten%'
        AND (WORK_LOCATION_LATITUDE IS NOT NULL OR WORK_LOCATION_LONGITUDE IS NOT NULL)
        and STOP_DATE >= GETDATE()
''', con) 

# Apply the function to calculate age from birthdate
wkstaten['BIRTH_DATE'] = wkstaten['BIRTH_DATE'].apply(age)

# Rename the field
wkstaten = wkstaten.rename(columns={'BIRTH_DATE': 'Patient_Age'})

# Fill the missing values with an empty string
wkstaten['ADDRESS_1'] = wkstaten['ADDRESS_1'].fillna('')
wkstaten['ADDRESS_2'] = wkstaten['ADDRESS_2'].fillna('')
wkstaten['city'] = wkstaten['city'].fillna('')
wkstaten['state'] = wkstaten['state'].fillna('')
wkstaten['zip'] = wkstaten['zip'].fillna('')

# Combine address components into a single column
wkstaten['address'] = wkstaten['ADDRESS_1'] + ' ' + wkstaten['ADDRESS_2'] + ' ' + wkstaten['city'] + ' ' + wkstaten['state'] + ' ' + wkstaten['zip']

# Label encode columns for features
label_encode_columns(wkstaten, ['sex', 'city', 'state'])

###------------------------------------------------------ #!!!
##  NYC
# Load the dataset
mis_lalo = pd.read_csv('C://PATH/Scheduled Code/addresses_with_lat_long.csv')

#Filter those that DO NOT contain 'Staten'
mis_lalo_nyc = mis_lalo[~mis_lalo['address'].str.contains('staten|STATEN|Staten')]

# Merge latitude and longitude with member data
addy = pd.merge(members, mis_lalo_nyc, on='Person_id', how='left')

# Fill missing lat/long values in the members dataframe
addy['latitude_x'].fillna(addy['latitude_y'], inplace=True)
addy['longitude_x'].fillna(addy['longitude_y'], inplace=True)

# Drop unnecessary columns
addy.drop(['address_y', 'latitude_y', 'longitude_y'], axis=1, inplace=True)

# Rename columns for clarity
addy = addy.rename(columns={'address_x': 'address', 'latitude_x': 'latitude', 'longitude_x': 'longitude'})

# Drop unnecessary columns
addy.drop(['Name', 'ADDRESS_1', 'ADDRESS_2', 'address'], axis=1, inplace=True)

# Drop rows with NaN values and infinity values
addy = addy.dropna().replace([np.inf, -np.inf], np.nan).dropna()


##  Staten
#Filter those that DO contain 'Staten'
mis_lalo_stat = mis_lalo[mis_lalo['address'].str.contains('staten|STATEN|Staten')]

# Merge latitude and longitude with member data
stat_addy = pd.merge(staten, mis_lalo_stat, on='Person_id', how='left')

# Fill missing lat/long values in the members dataframe
stat_addy['latitude_x'].fillna(stat_addy['latitude_y'], inplace=True)
stat_addy['longitude_x'].fillna(stat_addy['longitude_y'], inplace=True)

# Drop unnecessary columns
stat_addy.drop(['address_y', 'latitude_y', 'longitude_y'], axis=1, inplace=True)

# Rename columns for clarity
stat_addy = stat_addy.rename(columns={'address_x': 'address', 'latitude_x': 'latitude', 'longitude_x': 'longitude'})

# Drop unnecessary columns
stat_addy.drop(['Name', 'ADDRESS_1', 'ADDRESS_2', 'address'], axis=1, inplace=True)

# Drop rows with NaN values and infinity values
stat_addy = stat_addy.dropna().replace([np.inf, -np.inf], np.nan).dropna()


close_all_connections()

# ==============================

# label encode columns
label_encode_columns(lalo_fivestar, ['practicename', 'five_star_name', 'state', 'city', 'Centername'])

# Initialize cluster centers
init_centers = lalo_fivestar[['latitude', 'longitude']].values
num_clusters = len(fivestar_fac)
stat_numb_clus = len(stat_5stars)

#### ============================= 
# Load zip code lists
manh_zip = pd.read_excel('C:/PATH/NYC_zips.xlsx', sheet_name='Manhattan')
bronx_zip = pd.read_excel('C:/PATH/NYC_zips.xlsx', sheet_name='Bronx')
brk_quen_zip = pd.read_excel('C:/PATH/NYC_zips.xlsx', sheet_name='Brook-Queens')


# Check and convert zip codes in 'members' to strings
if not pd.api.types.is_string_dtype(addy['zip']):
    addy['zip'] = addy['zip'].astype(str)

# Check and convert zip codes in 'manh_zip' to strings
if not pd.api.types.is_string_dtype(manh_zip['Zip']):
    manh_zip['Zip'] = manh_zip['Zip'].astype(str)
    
# Check and convert zip codes in 'bronx_zip' to strings
if not pd.api.types.is_string_dtype(bronx_zip['Zip']):
    bronx_zip['Zip'] = bronx_zip['Zip'].astype(str)
        
# Check and convert zip codes in 'brk_quen_zip' to strings
if not pd.api.types.is_string_dtype(brk_quen_zip['Zip']):
    brk_quen_zip['Zip'] = brk_quen_zip['Zip'].astype(str)
    
# Check and convert zip codes in 'brk_quen_zip' to strings
if not pd.api.types.is_string_dtype(lalo_fivestar['zip']):
     lalo_fivestar['zip'] = lalo_fivestar['zip'].astype(str)   
    

# Make both zip codes lowercase for consistent matching
addy['zip'] = addy['zip'].str.lower()
lalo_fivestar['zip'] = lalo_fivestar['zip'].str.lower()
manh_zip['Zip'] = manh_zip['Zip'].str.lower()
bronx_zip['Zip'] = bronx_zip['Zip'].str.lower()
brk_quen_zip['Zip'] = brk_quen_zip['Zip'].str.lower()
staten['zip'] = staten['zip'].str.lower()
wkstaten['zip'] = wkstaten['zip'].str.lower()


# Handle empty zip codes by replacing with 'NA'
addy['zip'] = addy['zip'].replace('', 'NA')
lalo_fivestar['zip'] = lalo_fivestar['zip'].replace('', 'NA')
manh_zip['Zip'] = manh_zip['Zip'].replace('', 'NA')
bronx_zip['Zip'] = bronx_zip['Zip'].replace('', 'NA')
brk_quen_zip['Zip'] = brk_quen_zip['Zip'].replace('', 'NA')
staten['zip'] = staten['zip'].replace('', 'NA')
wkstaten['zip'] = wkstaten['zip'].replace('', 'NA')


# Filter 'members' dataframe using the cleaned 'manh_zip' dataframe
manhattan_members = addy[addy['zip'].isin(manh_zip['Zip'])]     #24594
bronx_members = addy[addy['zip'].isin(bronx_zip['Zip'])]        #37277
brk_queen_members = addy[addy['zip'].isin(brk_quen_zip['Zip'])] #74971


manhattan_5star = lalo_fivestar[lalo_fivestar['zip'].isin(manh_zip['Zip'])]
bronx_5star = lalo_fivestar[lalo_fivestar['zip'].isin(bronx_zip['Zip'])]
brk_queen_5star = lalo_fivestar[lalo_fivestar['zip'].isin(brk_quen_zip['Zip'])]

# Initialize cluster centers
m_centers = manhattan_5star[['latitude', 'longitude']].values
b_centers = bronx_5star[['latitude', 'longitude']].values
b_q_centers = brk_queen_5star[['latitude', 'longitude']].values
staten_centers = stat_5stars[['latitude', 'longitude']].values


#####################################  Manhattan  ##################################### #!!!

# Function to remove outliers using IQR
def remove_outliers(data, threshold_factor=1.3):    #1.3
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_threshold = Q1 - threshold_factor * IQR
    upper_threshold = Q3 + threshold_factor * IQR
    outliers = (data < lower_threshold) | (data > upper_threshold)
    return data[~outliers.any(axis=1)]

manh_noouts = remove_outliers(manhattan_members[['latitude', 'longitude']])


###---------------------Neighbors 
from scipy.spatial.distance import cdist
import numpy as np


# Ch# Calculate distances between data points and init_centers
distances = cdist(manh_noouts[['latitude', 'longitude']], m_centers)

# Assign each data point to the nearest centroid
cluster_labels = np.argmin(distances, axis=1)

# Add the cluster labels to the dataframe
manh_noouts['cluster'] = cluster_labels

# Create a GeoDataFrame from the dataframe
geometry = gpd.points_from_xy(manh_noouts['longitude'], manh_noouts['latitude'])
gdf = gpd.GeoDataFrame(manh_noouts, geometry=geometry)

# Plot the GeoDataFrame with different colors for each cluster
fig, ax = plt.subplots(figsize=(20, 15))
gdf.plot(column='cluster', categorical=False, markersize=30, cmap='tab20', ax=ax)

# Plot the cluster centers
for center in m_centers:
    plt.scatter(center[1], center[0], c='black', marker='*', s=100)

# Evaluate the clustering with scores
silhouettem = silhouette_score(manh_noouts[['latitude', 'longitude']], cluster_labels)
davies_bouldin = davies_bouldin_score(manh_noouts[['latitude', 'longitude']], cluster_labels)
calinski_harabasz = calinski_harabasz_score(manh_noouts[['latitude', 'longitude']], cluster_labels)

# Set plot labels and aspect ratio
plt.title(f'Manhattan Home 5-Stars, Sil:{silhouettem:.4f}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_aspect('equal')

# Show the plot
plt.tight_layout()
plt.show()

print(f'Silhouette Score Mnh: {silhouettem:.4f}')
print(f'Davies-Bouldin Score Mnh: {davies_bouldin:.4f}')
print(f'Calinski-Harabasz Score Mnh: {calinski_harabasz:.4f}')


### MTA metrobus 
import requests
import zipfile
import io

url = 'https://rrgtfsfeeds.s3.amazonaws.com/gtfs_m.zip'

# Download the ZIP file from the URL
response = requests.get(url)

# Open the ZIP file in memory
zip_file = zipfile.ZipFile(io.BytesIO(response.content))

# Extract and read the specific file 'shapes.txt'
with zip_file.open('stops.txt') as shapes_file:
    mhtn_bus = pd.read_csv(shapes_file)



# Create GeoDataFrames for both datasets (bus stations and Manhattan clustering)
geometry_mta = gpd.points_from_xy(mhtn_bus['stop_lon'], mhtn_bus['stop_lat'])  # Bus stop coordinates
gdf_mta = gpd.GeoDataFrame(mhtn_bus, geometry=geometry_mta)

geometry_members = gpd.points_from_xy(manh_noouts['longitude'], manh_noouts['latitude'])  # Member coordinates
gdf_members = gpd.GeoDataFrame(manh_noouts, geometry=geometry_members)

# Ensure both GeoDataFrames use the same CRS (WGS84 - EPSG:4326 is commonly used for GPS coordinates)
gdf_mta = gdf_mta.set_crs("EPSG:4326")
gdf_members = gdf_members.set_crs("EPSG:4326")

# Create the plot and overlay both GeoDataFrames
fig, ax = plt.subplots(figsize=(20, 15))

# Plot member locations with a colormap
gdf_members.plot(column='cluster', categorical=False, markersize=50, cmap='tab20', ax=ax, label="Member Locations")

# Plot bus stations with a distinct color (red)
gdf_mta.plot(marker='o', color='black', markersize=5, ax=ax, alpha=0.9, label="MTA Bus Stations")

# Plot cluster centers for member locations
for center in m_centers:
    plt.scatter(center[1], center[0], c='red', marker='*', s=400, label="Cluster Centers")

# Set plot labels and aspect ratio
plt.title('Manhattan Member Locations and MTA Bus Stations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_aspect('equal')



# Show the plot
plt.tight_layout()
plt.show()


#####################################  Bronx  #####################################

# Function to remove outliers using IQR
def remove_outliers(data, threshold_factor=1.2):    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_threshold = Q1 - threshold_factor * IQR
    upper_threshold = Q3 + threshold_factor * IQR
    outliers = (data < lower_threshold) | (data > upper_threshold)
    return data[~outliers.any(axis=1)]

bronx_noouts = remove_outliers(bronx_members[['latitude', 'longitude']])

###---------------------Neighbors 
from scipy.spatial.distance import cdist
import numpy as np

#### Calculate distances between data points and init_centers
distances = cdist(bronx_noouts[['latitude', 'longitude']], b_centers)

# Assign each data point to the nearest centroid
cluster_labels = np.argmin(distances, axis=1)

# Add the cluster labels to the dataframe
bronx_noouts['cluster'] = cluster_labels

# Create a GeoDataFrame from the dataframe
geometry = gpd.points_from_xy(bronx_noouts['longitude'], bronx_noouts['latitude'])
gdf = gpd.GeoDataFrame(bronx_noouts, geometry=geometry)

# Plot the GeoDataFrame with different colors for each cluster
fig, ax = plt.subplots(figsize=(20, 15))
gdf.plot(column='cluster', categorical=False, markersize=30, cmap='tab20', ax=ax)

# Plot the cluster centers
for center in b_centers:
    plt.scatter(center[1], center[0], c='black', marker='*', s=150)

# Evaluate the clustering with scores
silhouetteb = silhouette_score(bronx_noouts[['latitude', 'longitude']], cluster_labels)
davies_bouldin = davies_bouldin_score(bronx_noouts[['latitude', 'longitude']], cluster_labels)
calinski_harabasz = calinski_harabasz_score(bronx_noouts[['latitude', 'longitude']], cluster_labels)

# Set plot labels and aspect ratio
plt.title(f'Bronx Home 5-Stars, Sil:{silhouetteb:.4f}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_aspect('equal')

# Show the plot
plt.tight_layout()
plt.show()

print(f'Silhouette Score Bx: {silhouetteb:.4f}')
print(f'Davies-Bouldin Score Bx: {davies_bouldin:.4f}')
print(f'Calinski-Harabasz Score Bx: {calinski_harabasz:.4f}')
    

#####################################  Brooklyn / Queens  #####################################


# Function to remove outliers using IQR
def remove_outliers(data, threshold_factor=1.5):   
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_threshold = Q1 - threshold_factor * IQR
    upper_threshold = Q3 + threshold_factor * IQR
    outliers = (data < lower_threshold) | (data > upper_threshold)
    return data[~outliers.any(axis=1)]

b_q_noouts = remove_outliers(brk_queen_members[['latitude', 'longitude']])

###---------------------Neighbors 
from scipy.spatial.distance import cdist
import numpy as np

# Ch# Calculate distances between data points and init_centers
distances = cdist(b_q_noouts[['latitude', 'longitude']], b_q_centers)

# Assign each data point to the nearest centroid
cluster_labels = np.argmin(distances, axis=1)

# Add the cluster labels to the dataframe
b_q_noouts['cluster'] = cluster_labels

# Create a GeoDataFrame from the dataframe
geometry = gpd.points_from_xy(b_q_noouts['longitude'], b_q_noouts['latitude'])
gdf = gpd.GeoDataFrame(b_q_noouts, geometry=geometry)

# Plot the GeoDataFrame with different colors for each cluster
fig, ax = plt.subplots(figsize=(20, 15))
gdf.plot(column='cluster', categorical=False, markersize=30, cmap='tab20', ax=ax)

# Plot the cluster centers
for center in b_q_centers:
    plt.scatter(center[1], center[0], c='black', marker='*', s=150)

# Evaluate the clustering with scores
silhouetteq = silhouette_score(b_q_noouts[['latitude', 'longitude']], cluster_labels)
davies_bouldin = davies_bouldin_score(b_q_noouts[['latitude', 'longitude']], cluster_labels)
calinski_harabasz = calinski_harabasz_score(b_q_noouts[['latitude', 'longitude']], cluster_labels)

# Set plot labels and aspect ratio
plt.title(f'Bronx Home 5-Stars, Sil:{silhouetteq:.4f}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_aspect('equal')

# Show the plot
plt.tight_layout()
plt.show()

print(f'Silhouette Score BQ: {silhouetteq:.4f}')
print(f'Davies-Bouldin Score BQ: {davies_bouldin:.4f}')
print(f'Calinski-Harabasz Score BQ: {calinski_harabasz:.4f}')


#####################################  Staten Island  #####################################

# Function to remove outliers using IQR
def remove_outliers(data, threshold_factor=1.4):   
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_threshold = Q1 - threshold_factor * IQR
    upper_threshold = Q3 + threshold_factor * IQR
    outliers = (data < lower_threshold) | (data > upper_threshold)
    return data[~outliers.any(axis=1)]

stat_noouts = remove_outliers(stat_addy[['latitude', 'longitude']])

# Ch# Calculate distances between data points and init_centers
statdistances = cdist(stat_noouts[['latitude', 'longitude']], staten_centers)

# Assign each data point to the nearest centroid
statcluster_labels = np.argmin(statdistances, axis=1)

# Add the cluster labels to the dataframe
stat_noouts['cluster'] = statcluster_labels

# Create a GeoDataFrame from the dataframe
geometry = gpd.points_from_xy(stat_noouts['longitude'], stat_noouts['latitude'])
gdf = gpd.GeoDataFrame(stat_noouts, geometry=geometry)

# Plot the GeoDataFrame with different colors for each cluster
fig, ax = plt.subplots(figsize=(20, 15))
gdf.plot(column='cluster', categorical=False, markersize=50, cmap='tab20', ax=ax)

# Plot the cluster centers
for center in staten_centers:
    plt.scatter(center[1], center[0], c='black', marker='*', s=150)

# Evaluate the clustering with scores
silhouettest = silhouette_score(stat_noouts[['latitude', 'longitude']], statcluster_labels)
davies_bouldinst = davies_bouldin_score(stat_noouts[['latitude', 'longitude']], statcluster_labels)
calinski_harabaszst = calinski_harabasz_score(stat_noouts[['latitude', 'longitude']], statcluster_labels)

# Set plot labels and aspect ratio
plt.title(f'Bronx Home 5-Stars, Sil:{silhouetteq:.4f}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_aspect('equal')

# Show the plot
plt.tight_layout()
plt.show()

print(f'Silhouette Score BQ: {silhouettest:.4f}')
print(f'Davies-Bouldin Score BQ: {davies_bouldinst:.4f}')
print(f'Calinski-Harabasz Score BQ: {calinski_harabaszst:.4f}')


############################## Working Model 

# # #---------------------Neighbors 
# from scipy.spatial.distance import cdist
# import numpy as np

# # Ch# Calculate distances between data points and init_centers
# distances = cdist(kaddy_no_outliers[['latitude', 'longitude']], init_centers)

# # Assign each data point to the nearest centroid
# cluster_labels = np.argmin(distances, axis=1)

# # Add the cluster labels to the dataframe
# kaddy_no_outliers['cluster'] = cluster_labels

# # Create a GeoDataFrame from the dataframe
# geometry = gpd.points_from_xy(kaddy_no_outliers['longitude'], kaddy_no_outliers['latitude'])
# gdf = gpd.GeoDataFrame(kaddy_no_outliers, geometry=geometry)

# # Plot the GeoDataFrame with different colors for each cluster
# fig, ax = plt.subplots(figsize=(20, 15))
# gdf.plot(column='cluster', categorical=False, markersize=30, cmap='tab20', ax=ax)

# # Plot the cluster centers
# for center in init_centers:
#     plt.scatter(center[1], center[0], c='black', marker='*', s=100)


# # Set plot labels and aspect ratio
# plt.title('Clustering Around Home 5-Stars')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.gca().set_aspect('equal')

# # Show the plot
# plt.tight_layout()
# plt.show()

# # Evaluate the clustering with scores
# silhouette = silhouette_score(kaddy_no_outliers[['latitude', 'longitude']], cluster_labels)
# davies_bouldin = davies_bouldin_score(kaddy_no_outliers[['latitude', 'longitude']], cluster_labels)
# calinski_harabasz = calinski_harabasz_score(kaddy_no_outliers[['latitude', 'longitude']], cluster_labels)

# print(f'Silhouette Score: {silhouette:.4f}')
# print(f'Davies-Bouldin Score: {davies_bouldin:.4f}')
# print(f'Calinski-Harabasz Score: {calinski_harabasz:.4f}')

# #############--------------------------------------------------------------------------

## Count clusters to see spread
manh_cluster_counts = manh_noouts['cluster'].value_counts().sort_index()
bronx_cluster_counts = bronx_noouts['cluster'].value_counts().sort_index()
bq_cluster_counts = b_q_noouts['cluster'].value_counts().sort_index()
stat_noouts_counts = stat_noouts['cluster'].value_counts().sort_index()

print(manh_cluster_counts)
print(bronx_cluster_counts)
print(bq_cluster_counts)
print(stat_noouts_counts)


####Cluster change for right cluster number
man_cluster = {0:8, 1:9, 2:10, 3:23, 4:29}
manh_noouts['cluster'] = manh_noouts['cluster'].replace(man_cluster)

bronx_cluster = {0:24, 1:25, 2:26, 3:27, 4:28}
bronx_noouts['cluster'] = bronx_noouts['cluster'].replace(bronx_cluster)

b_q_cluster = {8:11, 9:13, 10:14, 11:15, 12:16, 13:18, 14:19, 15:21, 16:22}
b_q_noouts['cluster'] = b_q_noouts['cluster'].replace(b_q_cluster)

staten_cluster = {0:30, 1:31}
stat_noouts['cluster'] = stat_noouts['cluster'].replace(staten_cluster)
stat_5stars['cluster'] = stat_5stars['cluster'].replace(staten_cluster)


#join all  dataframes back into one
home_members = pd.concat([manh_noouts, bronx_noouts, b_q_noouts, stat_noouts ])


###make things pretty
address = pd.merge(
    addy[['Person_id', 'zip', 'latitude', 'longitude', 'relation']],
    home_members[['latitude', 'longitude', 'cluster']],
    on=['latitude', 'longitude'],
    how='left'
)

stataddress = pd.merge(
    stat_addy[['Person_id', 'zip', 'latitude', 'longitude', 'relation']],
    home_members[['latitude', 'longitude', 'cluster']],
    on=['latitude', 'longitude'],
    how='left'
)


address = pd.merge(
        address[['Person_id', 'zip', 'latitude', 'longitude', 'relation', 'cluster']],
        manh_5stars[[ 'latitude', 'longitude', 'cluster', 'Centername', 'practicename']],
        on=['cluster'],
        how='left'
        )

stataddress = pd.merge(
        stataddress[['Person_id', 'zip', 'latitude', 'longitude', 'relation', 'cluster']],
        stat_5stars[[ 'latitude', 'longitude', 'cluster', 'Centername', 'practicename']],
        on=['cluster'],
        how='left'
        )

address_j = pd.concat([address, stataddress])

address_no_dupes = address_j.drop_duplicates()

#fix column names
address_j = address_no_dupes.rename({'latitude_x':'latitude', 'latitude_y':'cluster_latitude', 'longitude_x': 'longitude', 'longitude_y': 'cluster_longitude'}, axis=1)

#keep only non NAN rows
address_no_nan = address_j.dropna(subset=['cluster'])

#join all centroids into one array
all_centers = np.concatenate((init_centers, staten_centers))

###############################  Outlier Member clusters  #######################################

#create a DF that has the missing (NAN) clusters from the home addresses
nan_cluster_df = address_j[address_j['cluster'].isna()][['Person_id', 'latitude', 'longitude']]

#### Calculate distances between data points and init_centers
distances = cdist(nan_cluster_df[['latitude', 'longitude']], all_centers)

# Assign each data point to the nearest centroid
cluster_labels = np.argmin(distances, axis=1)

# Add the cluster labels to the dataframe
nan_cluster_df['cluster'] = cluster_labels

# Create a GeoDataFrame from the dataframe
geometry = gpd.points_from_xy(nan_cluster_df['longitude'], nan_cluster_df['latitude'])
gdf = gpd.GeoDataFrame(nan_cluster_df, geometry=geometry)

# Plot the GeoDataFrame with different colors for each cluster
fig, ax = plt.subplots(figsize=(20, 15))
gdf.plot(column='cluster', categorical=False, markersize=30, cmap='tab20', ax=ax)

# Plot the cluster centers
for center in all_centers:
    plt.scatter(center[1], center[0], c='black', marker='*', s=100)


# Set plot labels and aspect ratio
plt.title('Clustering Around Home 5-Stars')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_aspect('equal')

# Show the plot
plt.tight_layout()
plt.show()

# Evaluate the clustering with scores
silhouette_nan = silhouette_score(nan_cluster_df[['latitude', 'longitude']], cluster_labels)
davies_bouldin_nan = davies_bouldin_score(nan_cluster_df[['latitude', 'longitude']], cluster_labels)
calinski_harabasz_nan = calinski_harabasz_score(nan_cluster_df[['latitude', 'longitude']], cluster_labels)

print(f'Silhouette Score nan: {silhouette_nan:.4f}')
print(f'Davies-Bouldin Score nan: {davies_bouldin_nan:.4f}')
print(f'Calinski-Harabasz Score nan: {calinski_harabasz_nan:.4f}')

###########################

###make things pretty
address_t = pd.merge(
    addy[['Person_id', 'zip', 'latitude', 'longitude', 'relation']],
    nan_cluster_df[['latitude', 'longitude', 'cluster']],
    on=['latitude', 'longitude'],
    how='right'
    )


stataddress_t = pd.merge(
    stat_addy[['Person_id', 'zip', 'latitude', 'longitude', 'relation']],
    nan_cluster_df[['latitude', 'longitude', 'cluster']],
    on=['latitude', 'longitude'],
    how='right'
    )

#put all 5start centers into one 
all_5stars = pd.concat([manh_5stars, stat_5stars])


address_tt = pd.merge(
        address_t[['Person_id', 'zip', 'latitude', 'longitude', 'relation', 'cluster']],
        all_5stars[[ 'latitude', 'longitude', 'cluster', 'Centername', 'practicename']],
        on=['cluster'],
        how='left'
        )


address_t_no_dupes = address_tt.drop_duplicates()

address_t = address_t.rename({'latitude_x':'latitude', 'latitude_y':'cluster_latitude', 'longitude_x': 'longitude', 'longitude_y': 'cluster_longitude'}, axis=1)


#join back the nan into original one
address_hm_full = pd.concat([address_no_nan, address_t])

address_hm_full = address_hm_full.drop_duplicates()



####calculation for distance from centroid and averages
def calculate_ratio_of_distances_above_threshold(df):
    # Calculate the haversine distances and assign them to a new column
    df['distance_to_centroid'] = df.apply(
        lambda row: haversine(row['latitude'], row['longitude'], row['cluster_latitude'], row['cluster_longitude']),
        axis=1
    )

    # Calculate mean and standard deviation once
    mean_distance = df['distance_to_centroid'].mean()
    std_distance = df['distance_to_centroid'].std()

    # Calculate the threshold (num) using mean and standard deviation
    num = mean_distance + std_distance

    # Use a boolean mask to filter rows where the column is greater than the threshold
    filtered_rows = df['distance_to_centroid'] > num

    # Count the number of rows that meet the condition
    num_rows_greater_than_threshold = filtered_rows.sum()

    # Get the total number of rows
    total_rows = len(df)

    # Calculate the ratio
    ratio = num_rows_greater_than_threshold / total_rows

    return ratio


result = calculate_ratio_of_distances_above_threshold(address_hm_full)
print(result)


def add_average_distance_column(df):
    for cluster in df['cluster'].unique():
        if not pd.isna(cluster):
            df.loc[df['cluster'] == cluster, 'average_distance_to_centroid'] = df.groupby('cluster')['distance_to_centroid'].mean()[cluster]
            df.loc[df['cluster'] == cluster, 'cluster_stdev'] = df.groupby('cluster')['distance_to_centroid'].std()[cluster]
    return df

df = add_average_distance_column(address_hm_full.copy())


################
# Get the distinct values of the averages column and cluster_stdev column
distinct_averages = df['average_distance_to_centroid'].unique()
distinct_stds = df['cluster_stdev'].unique()

# Group the dataframe by the 'cluster' column
grouped_df = df.groupby('cluster')

# Create an empty dictionary to store results for each cluster
cluster_results = {}

# Iterate over each group (cluster) in the grouped DataFrame
for cluster, group in grouped_df:
    # Calculate the ratio of distances above threshold for the current cluster
    ratio = calculate_ratio_of_distances_above_threshold(group)
    
    # Store the result in the dictionary with the cluster as the key
    cluster_results[cluster] = ratio

# Print the results for each cluster
for cluster, ratio in cluster_results.items():
    print(f"Cluster {cluster}: Ratio = {ratio}")
  
########################################################  Whole View of Home

# Create a GeoDataFrame from the dataframe
geometry = gpd.points_from_xy(df['longitude'], df['latitude'])
gdf = gpd.GeoDataFrame(df, geometry=geometry)

# Plot the GeoDataFrame with different colors for each cluster
fig, ax = plt.subplots(figsize=(20, 15))
gdf.plot(column='cluster', categorical=False, markersize=50, cmap='tab20', ax=ax)

# Plot the cluster centers
for center in all_centers:
    plt.scatter(center[1], center[0], c='black', marker='*', s=100)


# Set plot labels and aspect ratio
plt.title('Clustering Around Home 5-Stars')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_aspect('equal')

# Show the plot
plt.tight_layout()
plt.show()


# # #### save to csv
# df.to_csv(r'C:\PATH\Data\clusters.csv', index=False)

#df.to_sql('Member_Fivestar1', con, schema='Temp', index=False, chunksize=1000, if_exists='replace')


#### MTA 
#!!!
##Subway plot map

url = 'https://data.ny.gov/api/views/39hk-dx4f/rows.csv?accessType=DOWNLOAD'

mta = pd.read_csv(url)
mta.columns

# ##Double check with a visual 
# # Create a GeoDataFrame from the dataframe
# geometry = gpd.points_from_xy(mta['GTFS Longitude'], mta['GTFS Latitude'])
# gdf = gpd.GeoDataFrame(mta, geometry=geometry)

# # Plot the GeoDataFrame with different colors for each cluster
# fig, ax = plt.subplots(figsize=(20, 15))
# gdf.plot(column = 'Station ID', categorical=False, markersize=50, cmap='tab20', ax=ax)

# # Plot the cluster centers
# for center in all_centers:
#     plt.scatter(center[1], center[0], c='black', marker='*', s=100)


# # Set plot labels and aspect ratio
# plt.title('MTA Subway Stations ')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.gca().set_aspect('equal')

# # Show the plot
# plt.tight_layout()
# plt.show()



#### Load subways stops over clustering 
# Create GeoDataFrames for both datasets
geometry_mta = gpd.points_from_xy(mta['GTFS Longitude'], mta['GTFS Latitude'])
gdf_mta = gpd.GeoDataFrame(mta, geometry=geometry_mta)

geometry_members = gpd.points_from_xy(df['longitude'], df['latitude'])
gdf_members = gpd.GeoDataFrame(df, geometry=geometry_members)

# Ensure both GeoDataFrames use the same CRS ##--  Assuming both datasets use WGS84
gdf_mta = gdf_mta.set_crs("EPSG:4326")  
gdf_members = gdf_members.set_crs("EPSG:4326")

# Create the plot and plot both GeoDataFrames
fig, ax = plt.subplots(figsize=(20, 15))

# Plot member locations
gdf_members.plot(column='cluster', categorical=False, markersize=50, cmap='tab20', ax=ax, label="Member Locations")

# Plot subway stations
gdf_mta.plot(marker='o', color='black', markersize=7, ax=ax, alpha=0.9, label="MTA Subway Stations")

# Plot cluster centers for member locations
for center in all_centers:
    plt.scatter(center[1], center[0], c='red', marker='*', s=200, label="Cluster Centers")

# Set plot labels and aspect ratio
plt.title('Member Locations and MTA Subway Stations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_aspect('equal')

# Show the plot
plt.tight_layout()
plt.show()



### MTA metrobus 
import requests
import zipfile
import io

    
urls = ['https://rrgtfsfeeds.s3.amazonaws.com/gtfs_busco.zip',
        'https://rrgtfsfeeds.s3.amazonaws.com/gtfs_m.zip',
        'https://rrgtfsfeeds.s3.amazonaws.com/gtfs_bx.zip',
        'https://rrgtfsfeeds.s3.amazonaws.com/gtfs_b.zip', 
        'https://rrgtfsfeeds.s3.amazonaws.com/gtfs_q.zip',
        'https://rrgtfsfeeds.s3.amazonaws.com/gtfs_si.zip']


def download_and_extract(urls):
    all_data = []
    
    for url in urls:
        try:
            response = requests.get(url) #get url and download
            response.raise_for_status() #check for successful request
            
            #open zip in memory
            zip_file = zipfile.ZipFile(io.BytesIO(response.content))
            
            #extract and read file
            with zip_file.open('stops.txt') as stops_file:
                bus_stops = pd.read_csv(stops_file)
                all_data.append(bus_stops)
                

        except requests.exceptions.RequestException as e:
            print(f"Failed to download from {url}: {e}")
        except zipfile.BadZipFile:
            print(f"Invalid ZIP file from {url}")
        except KeyError:
            print(f"'stops.txt' not found in the ZIP from {url}")
        except pd.errors.EmptyDataError:
            print(f"'stops.txt' in {url} is empty")
            
        # Check if we have any DataFrames to concatenate
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)  # Combine all DataFrames
    else:
        combined_df = pd.DataFrame()  # Return an empty DataFrame if no data

    return combined_df        
                
bus_stops = download_and_extract(urls)


# #####Double check with a visual 
# # Create a GeoDataFrame from the dataframe
# geometry = gpd.points_from_xy(bus_stops['stop_lon'], bus_stops['stop_lat'])
# gdf = gpd.GeoDataFrame(bus_stops, geometry=geometry)

# # Plot the GeoDataFrame with different colors for each cluster
# fig, ax = plt.subplots(figsize=(20, 15))
# gdf.plot(column='stop_id', categorical=False, markersize=30, cmap='tab20', ax=ax)

# # Plot the cluster centers
# for center in m_centers:
#     plt.scatter(center[1], center[0], c='black', marker='*', s=100)

# # Show the plot
# plt.tight_layout()
# plt.show()    



####Load bus stop data over clustering 
# Create GeoDataFrames for both datasets (bus stations and Manhattan clustering)
geometry_mta = gpd.points_from_xy(bus_stops['stop_lon'], bus_stops['stop_lat'])  # Bus stop coordinates
gdf_mta = gpd.GeoDataFrame(bus_stops, geometry=geometry_mta)

geometry_members = gpd.points_from_xy(df['longitude'], df['latitude'])  # Member coordinates
gdf_members = gpd.GeoDataFrame(df, geometry=geometry_members)

# Ensure both GeoDataFrames use the same CRS (WGS84 - EPSG:4326 is commonly used for GPS coordinates)
gdf_mta = gdf_mta.set_crs("EPSG:4326")
gdf_members = gdf_members.set_crs("EPSG:4326")

# Create the plot and overlay both GeoDataFrames
fig, ax = plt.subplots(figsize=(20, 15))

# Plot member locations with a colormap
gdf_members.plot(column='cluster', categorical=False, markersize=50, cmap='tab20', ax=ax, label="Member Locations")

# Plot bus stations with a distinct color (red)
gdf_mta.plot(marker='o', color='black', markersize=1, ax=ax, alpha=0.9, label="MTA Bus Stations")

# Plot cluster centers for member locations
for center in all_centers:
    plt.scatter(center[1], center[0], c='red', marker='*', s=400, label="Cluster Centers")

# Set plot labels and aspect ratio
plt.title('Manhattan Member Locations and MTA Bus Stations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_aspect('equal')

# Show the plot
plt.tight_layout()
plt.show()


##### MTA Railroad stops
rr_urls = [ 'https://rrgtfsfeeds.s3.amazonaws.com/gtfslirr.zip',
           'https://rrgtfsfeeds.s3.amazonaws.com/gtfsmnr.zip']

rr_stops = download_and_extract(rr_urls)

# ##### Remove stops past  
fltrd_rr = rr_stops[(rr_stops['stop_lon'] > -73.6) | (rr_stops['stop_lat'] > 40.9)]
rr_cond = (rr_stops['stop_lon']  > -73.6) | (rr_stops['stop_lat'] > 40.9)
rr_stops = rr_stops[~rr_cond]

# #####Double check with a visual 
# # Create a GeoDataFrame from the dataframe
# geometry = gpd.points_from_xy(rr_stops['stop_lon'], rr_stops['stop_lat'])
# gdf = gpd.GeoDataFrame(rr_stops, geometry=geometry)

# # Plot the GeoDataFrame with different colors for each cluster
# fig, ax = plt.subplots(figsize=(20, 15))
# gdf.plot(column='stop_id', categorical=False, markersize=30, cmap='tab20', ax=ax)

# # Plot the cluster centers
# for center in all_centers:
#     plt.scatter(center[1], center[0], c='black', marker='*', s=100)

# # Show the plot
# plt.tight_layout()
# plt.show()    

####Load RR stop data over clustering 
# Create GeoDataFrames for both datasets (bus stations and clustering)
geometry_mta = gpd.points_from_xy(rr_stops['stop_lon'], rr_stops['stop_lat'])  # Bus stop coordinates
gdf_mta = gpd.GeoDataFrame(rr_stops, geometry=geometry_mta)

geometry_members = gpd.points_from_xy(df['longitude'], df['latitude'])  # Member coordinates
gdf_members = gpd.GeoDataFrame(df, geometry=geometry_members)

# Ensure both GeoDataFrames use the same CRS (WGS84 - EPSG:4326 is commonly used for GPS coordinates)
gdf_mta = gdf_mta.set_crs("EPSG:4326")
gdf_members = gdf_members.set_crs("EPSG:4326")

# Create the plot and overlay both GeoDataFrames
fig, ax = plt.subplots(figsize=(20, 15))

# Plot member locations with a colormap
gdf_members.plot(column='cluster', categorical=False, markersize=50, cmap='tab20', ax=ax, label="Member Locations")

# Plot bus stations with a distinct color (red)
gdf_mta.plot(marker='o', color='black', markersize=25, ax=ax, alpha=0.9, label="MTA Bus Stations")

# Plot cluster centers for member locations
for center in all_centers:
    plt.scatter(center[1], center[0], c='red', marker='*', s=200, label="Cluster Centers")

# Set plot labels and aspect ratio
plt.title('Manhattan Member Locations and MTA Bus Stations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_aspect('equal')

# Show the plot
plt.tight_layout()
plt.show()

close_all_connections()
#############---------------------------------------------------Work Addys
#!!!
# connection to server -> SERVER   db -> DATABASE
conn, con = connect_to_sql_server('SERVER', 'DATABASE')
cursor = conn.cursor()
conn.autocommit = True 

con = sqlalchemy.create_engine('mssql://SERVER/DATABASE?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server', fast_executemany=True)    
Session = sessionmaker(bind=con)
session = Session()   


# Fetch member information for clustering
wk_members = pd.read_sql("""
    SELECT DISTINCT Person_id, RELATION_CODE AS relation, LAST_NAME + ', ' + FIRST_NAME AS Name,
    BIRTH_DATE, sex, ADDRESS_1,  ADDRESS_2, WORK_LOCATION_CITY AS city, state, WORK_LOCATION_ZIP AS zip,
    WORK_LOCATION_LATITUDE AS wk_latitude, WORK_LOCATION_LONGITUDE AS wk_longitude
    FROM Main.EligMedical
    WHERE STOP_DATE >= GETDATE()
       -- AND medical_hospital = 'HOSPITAL'
        AND state = 'NY'
       -- AND WORK_LOCATION_LATITUDE < 40.9
        --AND WORK_LOCATION_LONGITUDE < -73.6
       -- AND WORK_LOCATION_LONGITUDE > -74.046
""", con)

# Clean up DOB to age
def age(birthdate):
    if pd.isnull(birthdate):
        return None
    else:
        today = date.today()
        age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
        return age

# Apply the function to calculate age from birthdate
wk_members['BIRTH_DATE'] = wk_members['BIRTH_DATE'].apply(age)

# Rename the field
wk_members = wk_members.rename(columns={'BIRTH_DATE': 'Patient_Age'})

# Fill the missing values with an empty string
wk_members['ADDRESS_1'] = wk_members['ADDRESS_1'].fillna('')
wk_members['ADDRESS_2'] = wk_members['ADDRESS_2'].fillna('')
wk_members['city'] = wk_members['city'].fillna('')
wk_members['state'] = wk_members['state'].fillna('')
wk_members['zip'] = wk_members['zip'].fillna('')

# Combine address components into a single column
wk_members['address'] = wk_members['ADDRESS_1'] + ' ' + wk_members['ADDRESS_2'] + ' ' + wk_members['city'] + ' ' + wk_members['state'] + ' ' + wk_members['zip']

# Label encode columns for features
wk_addy = label_encode_columns(wk_members, ['sex', 'city', 'state'])

# Drop unnecessary columns
wk_addy.drop(['Name', 'ADDRESS_1', 'ADDRESS_2', 'address'], axis=1, inplace=True)

# Drop rows with NaN values and infinity values
wk_addy = wk_addy.dropna().replace([np.inf, -np.inf], np.nan).dropna()

# Initialize cluster centers
init_centers = lalo_fivestar[['latitude', 'longitude']].values


close_all_connections()


#Remove outliers from the dataset
wk_no_outliers = remove_outliers(wk_addy[['wk_latitude', 'wk_longitude', 'zip']])

############################## Work Addresses 
#---------------------Neighbors 
from scipy.spatial.distance import cdist
import numpy as np

# Filter 'members' dataframe using the cleaned 'manh_zip' dataframe
wk_manhattan_members = wk_no_outliers[wk_no_outliers['zip'].isin(manh_zip['Zip'])]     #25415
wk_bronx_members = wk_no_outliers[wk_no_outliers['zip'].isin(bronx_zip['Zip'])]        #41276
wk_brk_queen_members = wk_no_outliers[wk_no_outliers['zip'].isin(brk_quen_zip['Zip'])] #76625

#####################################  Manhattan WK  #####################################

# Function to remove outliers using IQR
def remove_outliers(data, threshold_factor=3):    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_threshold = Q1 - threshold_factor * IQR
    upper_threshold = Q3 + threshold_factor * IQR
    outliers = (data < lower_threshold) | (data > upper_threshold)
    return data[~outliers.any(axis=1)]

wkmanh_noouts = remove_outliers(wk_manhattan_members[['wk_latitude', 'wk_longitude']])

wkmanh_noouts_fil = wkmanh_noouts[(wkmanh_noouts['wk_latitude'] >= 40.7) & (wkmanh_noouts['wk_longitude'] <= -73.91)]

wkmanh_noouts = wkmanh_noouts_fil

# # Calculate distances between data points and init_centers
wk_distances = cdist(wkmanh_noouts[['wk_latitude', 'wk_longitude']], m_centers)

# Assign each data point to the nearest centroid
cluster_labels = np.argmin(wk_distances, axis=1)

# Add the cluster labels to the dataframe
wkmanh_noouts['cluster'] = cluster_labels

# Create a GeoDataFrame from the dataframe
geometry = gpd.points_from_xy(wkmanh_noouts['wk_longitude'], wkmanh_noouts['wk_latitude'])
gdf = gpd.GeoDataFrame(wkmanh_noouts, geometry=geometry)

# Plot the GeoDataFrame with different colors for each cluster
fig, ax = plt.subplots(figsize=(20, 15))
gdf.plot(column='cluster', categorical=False, markersize=30, cmap='tab20', ax=ax)

# Plot the cluster centers
for center in m_centers:
    plt.scatter(center[1], center[0], c='black', marker='*', s=100)

# Evaluate the clustering with scores
silhouettemwk = silhouette_score(wkmanh_noouts[['wk_latitude', 'wk_longitude']], cluster_labels)
davies_bouldinmwk = davies_bouldin_score(wkmanh_noouts[['wk_latitude', 'wk_longitude']], cluster_labels)
calinski_harabaszmwk = calinski_harabasz_score(wkmanh_noouts[['wk_latitude', 'wk_longitude']], cluster_labels)

# Set plot labels and aspect ratio
plt.title(f'Manhattan Work 5-Stars, Sil:{silhouettemwk:.4f}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_aspect('equal')

# Show the plot
plt.tight_layout()
plt.show()

print(f'Silhouette Score: {silhouettemwk:.4f}')
print(f'Davies-Bouldin Score: {davies_bouldinmwk:.4f}')
print(f'Calinski-Harabasz Score: {calinski_harabaszmwk:.4f}')


#####################################  Bronx  #####################################

# Function to remove outliers using IQR
def remove_outliers(data, threshold_factor=1.5):    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_threshold = Q1 - threshold_factor * IQR
    upper_threshold = Q3 + threshold_factor * IQR
    outliers = (data < lower_threshold) | (data > upper_threshold)
    return data[~outliers.any(axis=1)]

wkbronx_noouts = remove_outliers(wk_bronx_members[['wk_latitude', 'wk_longitude']])


###---------------------Neighbors 
from scipy.spatial.distance import cdist
import numpy as np

# Ch# Calculate distances between data points and init_centers
distances = cdist(wkbronx_noouts[['wk_latitude', 'wk_longitude']], b_centers)

# Assign each data point to the nearest centroid
cluster_labels = np.argmin(distances, axis=1)

# Add the cluster labels to the dataframe
wkbronx_noouts['cluster'] = cluster_labels

# Create a GeoDataFrame from the dataframe
geometry = gpd.points_from_xy(wkbronx_noouts['wk_longitude'], wkbronx_noouts['wk_latitude'])
gdf = gpd.GeoDataFrame(wkbronx_noouts, geometry=geometry)

# Plot the GeoDataFrame with different colors for each cluster
fig, ax = plt.subplots(figsize=(20, 15))
gdf.plot(column='cluster', categorical=False, markersize=30, cmap='tab20', ax=ax)

# Plot the cluster centers
for center in b_centers:
    plt.scatter(center[1], center[0], c='black', marker='*', s=150)

# Evaluate the clustering with scores
silhouettebwk = silhouette_score(wkbronx_noouts[['wk_latitude', 'wk_longitude']], cluster_labels)
davies_bouldinbwk = davies_bouldin_score(wkbronx_noouts[['wk_latitude', 'wk_longitude']], cluster_labels)
calinski_harabaszbwk = calinski_harabasz_score(wkbronx_noouts[['wk_latitude', 'wk_longitude']], cluster_labels)

# Set plot labels and aspect ratio
plt.title(f'Bronx Home 5-Stars, Sil:{silhouetteb:.4f}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_aspect('equal')

# Show the plot
plt.tight_layout()
plt.show()

print(f'Silhouette Score: {silhouettebwk:.4f}')
print(f'Davies-Bouldin Score: {davies_bouldinbwk:.4f}')
print(f'Calinski-Harabasz Score: {calinski_harabaszbwk:.4f}')
    

#####################################  Brooklyn / Queens  #####################################


# Function to remove outliers using IQR
def remove_outliers(data, threshold_factor=1.2):    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_threshold = Q1 - threshold_factor * IQR
    upper_threshold = Q3 + threshold_factor * IQR
    outliers = (data < lower_threshold) | (data > upper_threshold)
    return data[~outliers.any(axis=1)]

wkb_q_noouts = remove_outliers(wk_brk_queen_members[['wk_latitude', 'wk_longitude']])

###---------------------Neighbors 
from scipy.spatial.distance import cdist
import numpy as np

# Ch# Calculate distances between data points and init_centers
qwkdistances = cdist(wkb_q_noouts[['wk_latitude', 'wk_longitude']], b_q_centers)

# Assign each data point to the nearest centroid
cluster_labels = np.argmin(qwkdistances, axis=1)

# Add the cluster labels to the dataframe
wkb_q_noouts['cluster'] = cluster_labels

# Create a GeoDataFrame from the dataframe
geometry = gpd.points_from_xy(wkb_q_noouts['wk_longitude'], wkb_q_noouts['wk_latitude'])
gdf = gpd.GeoDataFrame(wkb_q_noouts, geometry=geometry)

# Plot the GeoDataFrame with different colors for each cluster
fig, ax = plt.subplots(figsize=(20, 15))
gdf.plot(column='cluster', categorical=False, markersize=30, cmap='tab20', ax=ax)

# Plot the cluster centers
for center in b_q_centers:
    plt.scatter(center[1], center[0], c='black', marker='*', s=150)

# Evaluate the clustering with scores
silhouetteqwk = silhouette_score(wkb_q_noouts[['wk_latitude', 'wk_longitude']], cluster_labels)
davies_bouldinqwk = davies_bouldin_score(wkb_q_noouts[['wk_latitude', 'wk_longitude']], cluster_labels)
calinski_harabaszqwk = calinski_harabasz_score(wkb_q_noouts[['wk_latitude', 'wk_longitude']], cluster_labels)

# Set plot labels and aspect ratio
plt.title(f'Brooklyn/Queens Home 5-Stars, Sil:{silhouetteq:.4f}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_aspect('equal')

# Show the plot
plt.tight_layout()
plt.show()

print(f'Silhouette Score: {silhouetteqwk:.4f}')
print(f'Davies-Bouldin Score: {davies_bouldinqwk:.4f}')
print(f'Calinski-Harabasz Score: {calinski_harabaszqwk:.4f}')

#####################################  Staten Island  #####################################
#!!!
# Function to remove outliers using IQR
def remove_outliers(data, threshold_factor=1.4):   
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_threshold = Q1 - threshold_factor * IQR
    upper_threshold = Q3 + threshold_factor * IQR
    outliers = (data < lower_threshold) | (data > upper_threshold)
    return data[~outliers.any(axis=1)]

wkstat_noouts = remove_outliers(wkstaten[['latitude', 'longitude']])

# Ch# Calculate distances between data points and init_centers
statdistances = cdist(wkstat_noouts[['latitude', 'longitude']], staten_centers)

# Assign each data point to the nearest centroid
wkstatcluster_labels = np.argmin(statdistances, axis=1)

# Add the cluster labels to the dataframe
wkstat_noouts['cluster'] = wkstatcluster_labels

# Create a GeoDataFrame from the dataframe
geometry = gpd.points_from_xy(wkstat_noouts['longitude'], wkstat_noouts['latitude'])
gdf = gpd.GeoDataFrame(wkstat_noouts, geometry=geometry)

# Plot the GeoDataFrame with different colors for each cluster
fig, ax = plt.subplots(figsize=(20, 15))
gdf.plot(column='cluster', categorical=False, markersize=30, cmap='tab20', ax=ax)

# Plot the cluster centers
for center in staten_centers:
    plt.scatter(center[1], center[0], c='black', marker='*', s=150)

# Evaluate the clustering with scores
silhouettestw = silhouette_score(wkstat_noouts[['latitude', 'longitude']], wkstatcluster_labels)
davies_bouldinstw = davies_bouldin_score(wkstat_noouts[['latitude', 'longitude']], wkstatcluster_labels)
calinski_harabaszstw = calinski_harabasz_score(wkstat_noouts[['latitude', 'longitude']], wkstatcluster_labels)

# Set plot labels and aspect ratio
plt.title(f'Staten Work 5-Stars, Sil:{silhouetteq:.4f}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_aspect('equal')

# Show the plot
plt.tight_layout()
plt.show()

print(f'Silhouette Score BQ: {silhouettestw:.4f}')
print(f'Davies-Bouldin Score BQ: {davies_bouldinstw:.4f}')
print(f'Calinski-Harabasz Score BQ: {calinski_harabaszstw:.4f}')

wkstat_noouts = wkstat_noouts.rename(columns={'latitude': 'wk_latitude', 'longitude': 'wk_longitude'})


#############--------------------------------------------------------------------------

## Count clusters to see spread
wkmanh_cluster_counts = wkmanh_noouts['cluster'].value_counts().sort_index()
wkbronx_cluster_counts = wkbronx_noouts['cluster'].value_counts().sort_index()
wkbq_cluster_counts = wkb_q_noouts['cluster'].value_counts().sort_index()
wkstat_cluster_counts = wkstat_noouts['cluster'].value_counts().sort_index()

print(wkmanh_cluster_counts)
print(wkbronx_cluster_counts)
print(wkbq_cluster_counts)
print(wkstat_cluster_counts)


####Cluster change for right cluster number
wkman_cluster = {0:8, 1:9, 2:10, 3:23, 4:29}
wkmanh_noouts['cluster'] = wkmanh_noouts['cluster'].replace(wkman_cluster)

wkbronx_cluster = {0:24, 1:25, 2:26, 3:27, 4:28}
wkbronx_noouts['cluster'] = wkbronx_noouts['cluster'].replace(wkbronx_cluster)

wkb_q_cluster = {8:11, 9:13, 10:14, 11:15, 12:16, 13:18, 14:19, 15:21, 16:22}
wkb_q_noouts['cluster'] = wkb_q_noouts['cluster'].replace(wkb_q_cluster)

wkstat_cluster = {0:30, 1:31}
wkstat_noouts['cluster'] = wkstat_noouts['cluster'].replace(wkstat_cluster)


#!!!
#join all dataframes back into one
work_members = pd.concat([wkmanh_noouts, wkbronx_noouts, wkb_q_noouts, wkstat_noouts])


###make things pretty
wkaddress = pd.merge(
    wk_addy[['Person_id', 'zip', 'wk_latitude', 'wk_longitude', 'relation']],
    work_members[['wk_latitude', 'wk_longitude', 'cluster']],
    on=['wk_latitude', 'wk_longitude'],
    how='left'
)

wkaddress_no_dupes = wkaddress.drop_duplicates()

wkaddress = pd.merge(
        wkaddress_no_dupes[['Person_id', 'zip', 'wk_latitude', 'wk_longitude', 'relation', 'cluster']],
        all_5stars[[ 'latitude', 'longitude', 'cluster', 'Centername', 'practicename']],
        on=['cluster'],
        how='left'
        )

#fix column names
wkaddress = wkaddress.rename({'latitude':'cluster_latitude', 'longitude': 'cluster_longitude'}, axis=1)

#keep only non NAN rows
wkaddress_no_nan = wkaddress.dropna(subset=['cluster'])


###############################  NULL Member clusters  #######################################
wknan_cluster_df = wkaddress[wkaddress['cluster'].isna()][['Person_id', 'wk_latitude', 'wk_longitude']]

wknan_cluster_df_t = wknan_cluster_df[(wknan_cluster_df['wk_latitude'] >= 40.4) & (wknan_cluster_df['wk_longitude'] >= -74.02)  & (wknan_cluster_df['wk_longitude'] <= -73.5) & (wknan_cluster_df['wk_latitude'] <= 40.9)]

wknan_cluster_df = wknan_cluster_df_t

#### Calculate distances between data points and init_centers
wkdistances = cdist(wknan_cluster_df[['wk_latitude', 'wk_longitude']], init_centers)

# Assign each data point to the nearest centroid
wkcluster_labels = np.argmin(wkdistances, axis=1)

# Add the cluster labels to the dataframe
wknan_cluster_df['cluster'] = wkcluster_labels

# Create a GeoDataFrame from the dataframe
geometry = gpd.points_from_xy(wknan_cluster_df['wk_longitude'], wknan_cluster_df['wk_latitude'])
gdf = gpd.GeoDataFrame(wknan_cluster_df, geometry=geometry)

# Plot the GeoDataFrame with different colors for each cluster
fig, ax = plt.subplots(figsize=(20, 15))
gdf.plot(column='cluster', categorical=False, markersize=30, cmap='tab20', ax=ax)

# Plot the cluster centers
for center in all_centers:
    plt.scatter(center[1], center[0], c='black', marker='*', s=100)


# Set plot labels and aspect ratio
plt.title('Clustering Around Home 5-Stars')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_aspect('equal')

# Show the plot
plt.tight_layout()
plt.show()

# Evaluate the clustering with scores
wksilhouette_nan = silhouette_score(wknan_cluster_df[['wk_latitude', 'wk_longitude']], wkcluster_labels)
wkdavies_bouldin_nan = davies_bouldin_score(wknan_cluster_df[['wk_latitude', 'wk_longitude']], wkcluster_labels)
wkcalinski_harabasz_nan = calinski_harabasz_score(wknan_cluster_df[['wk_latitude', 'wk_longitude']], wkcluster_labels)

print(f'Silhouette Score nan: {wksilhouette_nan:.4f}')
print(f'Davies-Bouldin Score nan: {wkdavies_bouldin_nan:.4f}')
print(f'Calinski-Harabasz Score nan: {wkcalinski_harabasz_nan:.4f}')

###########################

###make things pretty
wkaddress_t = pd.merge(
    wk_addy[['Person_id', 'zip', 'wk_latitude', 'wk_longitude', 'relation']],
    wknan_cluster_df[['wk_latitude', 'wk_longitude', 'cluster']],
    on=['wk_latitude', 'wk_longitude'],
    how='right'
)

wkaddress_t_no_dupes = wkaddress_t.drop_duplicates()

wkaddress_t = pd.merge(
        wkaddress_t_no_dupes[['Person_id', 'zip', 'wk_latitude', 'wk_longitude', 'relation', 'cluster']],
        all_5stars[[ 'latitude', 'longitude', 'cluster', 'Centername', 'practicename']],
        on=['cluster'],
        how='left'
        )

wkaddress_t = wkaddress_t.rename(columns={
    'latitude': 'cluster_latitude',
    'longitude': 'cluster_longitude'
})

#join back the nan into original one
address_wk_full = pd.concat([wkaddress_no_nan, wkaddress_t])


address_wk_full = address_wk_full.drop_duplicates()



####calculation for distance from centroid and averages
def calculate_ratio_of_distances_above_threshold_wk(df):
    # Calculate the haversine distances and assign them to a new column
    df['distance_to_centroid'] = df.apply(
        lambda row: haversine(row['wk_latitude'], row['wk_longitude'], row['cluster_latitude'], row['cluster_longitude']),
        axis=1
    )

    # Calculate mean and standard deviation once
    mean_distance = df['distance_to_centroid'].mean()
    std_distance = df['distance_to_centroid'].std()

    # Calculate the threshold (num) using mean and standard deviation
    num = mean_distance + std_distance

    # Use a boolean mask to filter rows where the column is greater than the threshold
    filtered_rows = df['distance_to_centroid'] > num

    # Count the number of rows that meet the condition
    num_rows_greater_than_threshold = filtered_rows.sum()

    # Get the total number of rows
    total_rows = len(df)

    # Calculate the ratio
    ratio = num_rows_greater_than_threshold / total_rows

    return ratio


resultwk = calculate_ratio_of_distances_above_threshold_wk(address_wk_full)
print(resultwk)


def add_average_distance_column_wk(dfwk):
    for cluster in dfwk['cluster'].unique():
        if not pd.isna(cluster):
            dfwk.loc[dfwk['cluster'] == cluster, 'average_distance_to_centroid'] = dfwk.groupby('cluster')['distance_to_centroid'].mean()[cluster]
            dfwk.loc[dfwk['cluster'] == cluster, 'cluster_stdev'] = dfwk.groupby('cluster')['distance_to_centroid'].std()[cluster]
    return dfwk

dfwk = add_average_distance_column_wk(address_wk_full.copy())


################
# Get the distinct values of the averages column and cluster_stdev column
distinct_averages = dfwk['average_distance_to_centroid'].unique()
distinct_stds = dfwk['cluster_stdev'].unique()

# Group the dataframe by the 'cluster' column
grouped_dfwk = dfwk.groupby('cluster')

# Create an empty dictionary to store results for each cluster
cluster_resultswk = {}

# Iterate over each group (cluster) in the grouped DataFrame
for cluster, group in grouped_dfwk:
    # Calculate the ratio of distances above threshold for the current cluster
    ratio = calculate_ratio_of_distances_above_threshold_wk(group)
    
    # Store the result in the dictionary with the cluster as the key
    cluster_resultswk[cluster] = ratio

# Print the results for each cluster
for cluster, ratio in cluster_resultswk.items():
    print(f"Cluster {cluster}: Ratio = {ratio}")
  
########################################################  Whole View of Work

# Create a GeoDataFrame from the dataframe
geometry = gpd.points_from_xy(dfwk['wk_longitude'], dfwk['wk_latitude'])
gdf = gpd.GeoDataFrame(dfwk, geometry=geometry)

# Plot the GeoDataFrame with different colors for each cluster
fig, ax = plt.subplots(figsize=(20, 15))
gdf.plot(column='cluster', categorical=False, markersize=50, cmap='tab20', ax=ax)

# Plot the cluster centers
for center in all_centers:
    plt.scatter(center[1], center[0], c='black', marker='*', s=100)


# Set plot labels and aspect ratio
plt.title('Clustering Around Work 5-Stars')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_aspect('equal')

# Show the plot
plt.tight_layout()
plt.show()






# # #### save to csv
# address_wk_full.to_csv(r'C:\PATH\Data\wk_clusters.csv', index=False)

#dfwk.to_sql('wk_member_Fivestar1', con, schema='Temp', index=False, chunksize=1000, if_exists='replace')


close_all_connections()
