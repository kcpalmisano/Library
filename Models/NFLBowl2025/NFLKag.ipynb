#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:44:17 2024

@author: casey
"""


'''
GOAL - Predict if what happens before snap is successful
success is defined for the offense as a:
    successful yardage gain by run or pass
    the QB not being sacked
    no snap penalty
    no interception
    no fumble

'''

#%% libraries
## Import Libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import time
import gc
import inspect


#ML load
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, chi2, RFE
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance



# ## Load Plotting Libraries
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import plotly.subplots as sp
# from IPython.display import display, HTML


## Timing Start
start_time = time.monotonic()

#%% functions
## Functions for Data Analysis and Visualization

def buildROC(target_test, test_preds, model_name, color):
    """
    Plot Receiver Operating Characteristic (ROC) curve and calculate AUC.
    """
    fpr, tpr, _ = metrics.roc_curve(target_test, test_preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, color, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.legend(loc='lower right')

def plot_combined_roc(y_test, models_preds_colors):
    """
    Plot combined ROC curves for multiple models.
    """
    plt.figure()
    for preds, model_name, color in models_preds_colors:
        buildROC(y_test, preds, model_name, color)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.show()

def label_encode_columns(df, columns):
    """
    Label encode specified columns in a DataFrame.
    """
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def plot_correlation_heatmap(data, target_column):
    """
    Plot heatmap showing feature correlations with the target variable.
    """
    correlation_matrix = data.corr()
    target_corr = correlation_matrix[target_column].drop(target_column)
    sns.heatmap(target_corr.sort_values(ascending=False).to_frame(), annot=True, cmap="coolwarm", cbar=True)
    plt.title(f"Feature Correlation with {target_column}")
    plt.show()

def plot_feature_importance(feature_scores, method_name, score_column='Score', top_n=10):
    """
    Plot feature importance scores for a given method.

    Args:
        feature_scores (pd.DataFrame): DataFrame containing 'Feature' and a score column.
        method_name (str): The name of the feature importance method.
        score_column (str): The name of the column containing the scores.
        top_n (int): Number of top features to plot.
    """
    top_features = feature_scores.nlargest(top_n, score_column)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_features, x=score_column, y='Feature', palette='viridis')
    plt.title(f"Top {top_n} Features by {method_name} Importance")
    plt.xlabel(f"{method_name} Score")
    plt.ylabel("Feature")
    plt.show()


def drop_nulls(df, threshold):
    """
    Drop columns with more than a specified percentage of null values.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        threshold (float): The percentage threshold (0 to 1) for null values.

    Returns:
        pd.DataFrame: The DataFrame with specified columns removed.
    """
    # Calculate the percentage of null values per column
    null_percentages = df.isnull().mean()
    print("Null Percentages:\n", null_percentages)  # Debugging step

    # Identify columns to drop
    columns_to_drop = null_percentages[null_percentages > threshold].index
    print("Columns to Drop:\n", columns_to_drop)  # Debugging step

    # Drop the identified columns
    df_cleaned = df.drop(columns=columns_to_drop)

    # Print summary of removed columns
    print(f"Columns dropped (more than {threshold*100}% nulls): {list(columns_to_drop)}")
    
    return df_cleaned

def chi_squared_feature_importance(X, y, k=10):
    """
    Perform Chi-Squared test for feature importance.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        k (int): Number of top features to select.

    Returns:
        pd.DataFrame: DataFrame of feature names and their Chi-squared scores.
    """
    # Ensure X is non-negative
    if (X < 0).any().any():  # Check for negative values
        X = X + abs(X.min().min()) + 1  # Shift all values to be positive
    
    # Replace zero values with a small positive constant (optional)
    X = X.replace(0, 1)

    # Select top k features using Chi-squared
    chi2_selector = SelectKBest(chi2, k=k)
    chi2_selector.fit(X, y)
    scores = chi2_selector.scores_
    
    # Create a DataFrame of feature scores
    feature_scores = pd.DataFrame({'Feature': X.columns, 'Chi2 Score': scores})
    return feature_scores.sort_values(by='Chi2 Score', ascending=False)


def mutual_info_feature_importance(X, y, classification=True):
    """
    Calculate mutual information for feature importance.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        classification (bool): True for classification, False for regression.

    Returns:
        pd.DataFrame: DataFrame of feature names and their MI scores.
    """
    if classification:
        mi_scores = mutual_info_classif(X, y, random_state=42)
    else:
        mi_scores = mutual_info_regression(X, y, random_state=42)
    
    feature_scores = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
    return feature_scores.sort_values(by='MI Score', ascending=False)


def rfe_feature_importance(X, y, estimator=None, n_features_to_select=10):
    """
    Perform Recursive Feature Elimination for feature importance.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        estimator (model): Base model for RFE. Default is LogisticRegression.
        n_features_to_select (int): Number of top features to select.

    Returns:
        pd.DataFrame: DataFrame of feature names and their ranking.
    """
    if estimator is None:
        estimator = LogisticRegression(random_state=42, max_iter=1000)
    
    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)
    rankings = rfe.ranking_
    feature_scores = pd.DataFrame({'Feature': X.columns, 'RFE Rank': rankings})
    return feature_scores.sort_values(by='RFE Rank', ascending=True)


def permutation_feature_importance(model, X, y, classification=True):
    """
    Perform Permutation Importance to calculate feature importance.

    Args:
        model (fitted model): Pre-trained model to evaluate feature importance.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        classification (bool): True for classification, False for regression.

    Returns:
        pd.DataFrame: DataFrame of feature names and their importance scores.
    """
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    feature_scores = pd.DataFrame({'Feature': X.columns, 'Importance': result.importances_mean})
    return feature_scores.sort_values(by='Importance', ascending=False)


#####Timing 
end_time = time.monotonic()
print("Time for cell was ",timedelta(seconds=end_time - start_time))

#%% data load
## Data Loading
games = pd.read_csv('/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/games.csv')
player_play = pd.read_csv('/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/player_play.csv')
players = pd.read_csv('/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/players.csv')
plays = pd.read_csv('/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/plays.csv')
tracking_wk1 = pd.read_csv('/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/tracking_week_1.csv')
# tracking_wk2 = pd.read_csv('/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/tracking_week_2.csv')
# tracking_wk3 = pd.read_csv('/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/tracking_week_3.csv')
# tracking_wk4 = pd.read_csv('/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/tracking_week_4.csv')
# tracking_wk5 = pd.read_csv('/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/tracking_week_5.csv')
# tracking_wk6 = pd.read_csv('/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/tracking_week_6.csv')
# tracking_wk7 = pd.read_csv('/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/tracking_week_7.csv')
# tracking_wk8 = pd.read_csv('/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/tracking_week_8.csv')
# tracking_wk9 = pd.read_csv('/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/tracking_week_9.csv')

## Merge Datasets
merged_play_data = pd.merge(player_play, plays, on=['gameId', 'playId'], how='inner')

#####Timing 
end_time = time.monotonic()
print("Time for cell was ",timedelta(seconds=end_time - start_time))

#%% Tracking file load 
## List file paths for tracking data
# tracking_files = [
#     '/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/tracking_week_1.csv',
#     '/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/tracking_week_2.csv',
#     '/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/tracking_week_3.csv',
#     '/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/tracking_week_4.csv',
#     '/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/tracking_week_5.csv',
#     '/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/tracking_week_6.csv',
#     '/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/tracking_week_7.csv',
#     '/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/tracking_week_8.csv',
#     '/Users/casey/Desktop/Data/nfl-big-data-bowl-2025/tracking_week_9.csv'
# ]

# ## Initialize an empty DataFrame
# tracking_combined = pd.DataFrame()

# ## Process and append each file in chunks
# for file in tracking_files:
#     for chunk in pd.read_csv(file, chunksize=500000):  
#         tracking_combined = pd.concat([tracking_combined, chunk], ignore_index=True)

# ## Check memory usage and size
# print(f"Tracking Data Rows: {tracking_combined.shape[0]}, Columns: {tracking_combined.shape[1]}")
# print(tracking_combined.info())

#####Timing 
end_time = time.monotonic()
print("Time for cell was ",timedelta(seconds=end_time - start_time))

#%% information
games.info()
player_play.info()
players.info()
plays.info()


#%%  Team W/L home/visitor Visual

def prepare_team_data(games):
    # Count games played and won by home and visitor teams
    home_counts = games.groupby(['season', 'homeTeamAbbr']).size().reset_index(name='games_played')
    home_wins = games[games['homeFinalScore'] > games['visitorFinalScore']].groupby(['season', 'homeTeamAbbr']).size().reset_index(name='games_won')
    visitor_counts = games.groupby(['season', 'visitorTeamAbbr']).size().reset_index(name='games_played')
    visitor_wins = games[games['visitorFinalScore'] > games['homeFinalScore']].groupby(['season', 'visitorTeamAbbr']).size().reset_index(name='games_won')
    return home_counts, home_wins, visitor_counts, visitor_wins

def prepare_total_data(home_counts, home_wins, visitor_counts, visitor_wins):
    # Total games and wins by team
    total_games_home = home_counts.groupby('homeTeamAbbr')['games_played'].sum().reset_index(name='total_games')
    total_wins_home = home_wins.groupby('homeTeamAbbr')['games_won'].sum().reset_index(name='total_wins')
    total_games_visitor = visitor_counts.groupby('visitorTeamAbbr')['games_played'].sum().reset_index(name='total_games')
    total_wins_visitor = visitor_wins.groupby('visitorTeamAbbr')['games_won'].sum().reset_index(name='total_wins')
    return total_games_home, total_wins_home, total_games_visitor, total_wins_visitor

def plot_bar_chart(data, title, ylabel, xlabel):
    teams = data['team']
    played = data['games_played']
    won = data['games_won']
    
    x = range(len(teams))  # X positions for the bars
    width = 0.35          # Width of the bars
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, played, width, label='Games Played', color='skyblue')
    ax.bar([p + width for p in x], won, width, label='Games Won', color='orange')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(teams, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_pie_chart(total_games, total_wins, title, pct_distance=0.8):
    # Merge total games and wins to ensure consistency
    combined = total_games.merge(total_wins, on='team', how='outer').fillna(0)
    
    # Extract labels and values
    labels = combined['team']
    games = combined['total_games']
    wins = combined['total_wins']
    
    # Plot the pie charts
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    axs[0].pie(
        games,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.tab10.colors[:len(games)],
        pctdistance=pct_distance  # Adjust percentage label distance
    )
    axs[0].set_title('Total Games')
    
    axs[1].pie(
        wins,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.tab10.colors[:len(wins)],
        pctdistance=pct_distance  # Adjust percentage label distance
    )
    axs[1].set_title('Total Wins')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()



## Main Execution
home_team_counts, home_team_wins, visitor_team_counts, visitor_team_wins = prepare_team_data(games)
total_games_home, total_wins_home, total_games_visitor, total_wins_visitor = prepare_total_data(
    home_team_counts, home_team_wins, visitor_team_counts, visitor_team_wins
)

## Combine data for bar chart
home_data = home_team_counts.merge(home_team_wins, how='left', on=['season', 'homeTeamAbbr']).fillna(0)
home_data.columns = ['season', 'team', 'games_played', 'games_won']

visitor_data = visitor_team_counts.merge(visitor_team_wins, how='left', on=['season', 'visitorTeamAbbr']).fillna(0)
visitor_data.columns = ['season', 'team', 'games_played', 'games_won']

## Plot bar charts
plot_bar_chart(home_data, "Home Games Played vs Won", "Game Count", "Teams")
plot_bar_chart(visitor_data, "Visitor Games Played vs Won", "Game Count", "Teams")

## Prepare total data for pie charts
total_games_home.columns = ['team', 'total_games']
total_wins_home.columns = ['team', 'total_wins']
total_games_visitor.columns = ['team', 'total_games']
total_wins_visitor.columns = ['team', 'total_wins']

## Plot pie charts
plot_pie_chart(total_games_home, total_wins_home, "Total Home Games and Wins")
plot_pie_chart(total_games_visitor, total_wins_visitor, "Total Visitor Games and Wins")


#####Timing 
end_time = time.monotonic()
print("Time for cell was ",timedelta(seconds=end_time - start_time))

#%%





#####Timing 
end_time = time.monotonic()
print("Time for cell was ",timedelta(seconds=end_time - start_time))

#%% feature and define success

'''
Features to think about: 
Pre-snap Formation Metrics: Analyze offensive and defensive alignments to identify patterns that influence post-snap actions.

Player Motion Indicators: Incorporate data on player movements before the snap, such as shifts and motions, which can provide insights into play intentions.

Personnel Groupings: Evaluate the combination of players on the field, as different groupings can signal specific play types.

Game Context: Include situational factors like down, distance, score, and time remaining, which significantly impact play-calling decisions.
''' 
 
 
## Define Success Criteria
def is_success(row):
    """
    Determines if a play is successful based on yardage, sacks, penalties, interceptions, and fumbles.

    Args:
        row (pd.Series): A single row of the DataFrame.

    Returns:
        int: 1 for success, 0 for failure.
    """
    ## Yardage gain
    if row['down'] == 1 and row['yardsGained'] >= 4:
        yardage_success = True
    elif row['down'] == 2 and row['yardsGained'] >= row['yardsToGo'] / 2:
        yardage_success = True
    elif row['down'] in [3, 4] and row['yardsGained'] >= row['yardsToGo']:
        yardage_success = True
    else:
        yardage_success = False

    ## No sack
    no_sack = row['passResult'] != 'S'

    ## No snap penalty
    no_penalty = row['playNullifiedByPenalty'] == 'N'

    ## No interception
    no_interception = row['passResult'] != 'IN'

    ## No fumble
    no_fumble = row['fumbles'] == 0

    # Overall success
    return int(yardage_success and no_sack and no_penalty and no_interception and no_fumble)

# def define_success(data):  ***Previous 
#     """
#     Add a binary column 'successAfterSnap' based on play outcomes.
#     """
#     data['successAfterSnap'] = (
#         (data['yardsGained'] > 0) |
#         (data['hadPassReception'] == 1) |
#         (data['hadInterception'] == 0) |
#         (data['fumbles'] == 0)
#     ).astype(int)
#     return data

merged_play_data['successAfterSnap'] = merged_play_data.apply(is_success, axis=1)

## Feature Engineering
def create_features(data):
    # Flag if it's a critical down (3rd or 4th down)
    data['isCriticalDown'] = data['down'].isin([3, 4]).astype(int)
    
    # Calculate score differential before the snap
    data['scoreDifferential'] = data['preSnapHomeScore'] - data['preSnapVisitorScore']
    
    # Normalize the yardline (distance from the 50-yard line)
    data['normalizedYardline'] = (data['absoluteYardlineNumber'] - 50).abs()
    
    # Fraction of time left in the current quarter
    data['quarterTimeLeft'] = (
        pd.to_timedelta(data['gameClock'], errors='coerce').dt.total_seconds() / 900
    )
    
    # Offensive tendency: rush attempts vs. dropbacks
    data['offensiveTendency'] = data['hadRushAttempt'].astype(int) - data['hadDropback'].astype(int)
    
    # Handle NaN or inf in 'wasInitialPassRusher' and 'causedPressure'
    data['wasInitialPassRusher'] = data['wasInitialPassRusher'].fillna(0).astype(int)
    data['causedPressure'] = data['causedPressure'].fillna(0).astype(int)
    
    # Pressure rate caused by defenders (pressure per rusher)
    data['pressureRate'] = data['causedPressure'] / (data['wasInitialPassRusher'] + 1e-5)  # Avoid division by zero
    
    # Whether the defensive scheme is Zone coverage
    data['isZoneCoverage'] = (data['pff_manZone'] == 'Zone').astype(int)
    
    # Average time to pressure for pass rushers
    data['avgTimeToPressure'] = data.groupby('nflId')['timeToPressureAsPassRusher'].transform('mean')
    
    # Flag if a player is in motion at the ball snap
    data['isInMotion'] = data['inMotionAtBallSnap'].fillna(0).astype(int)
    
    # Average rushing yards for players
    data['avgRushingYards'] = data.groupby('nflId')['rushingYards'].transform('mean')
    
    # Average passing yards for players
    data['avgPassingYards'] = data.groupby('nflId')['passingYards'].transform('mean')
    
    # Team-level average Expected Points Added (EPA)
    data['teamAvgEPA'] = data.groupby('possessionTeam')['expectedPointsAdded'].transform('mean')
    
    # Defensive blitz rate (pressure rate aggregated at the team level)
    data['defensiveBlitzRate'] = data.groupby('defensiveTeam')['pressureRate'].transform('mean')
    
    # Red zone indicator (inside the 20-yard line)
    data['isRedZone'] = (data['absoluteYardlineNumber'] >= 80).astype(int)
    
    # Goal-to-go situations (distance to first down is 10 yards or less)
    data['isGoalToGo'] = (data['yardsToGo'] <= 10).astype(int)   ##penalty causality issue 
    
    # Urgent play clock situation (5 seconds or less remaining)
    data['isPlayClockCritical'] = (data['playClockAtSnap'] <= 5).astype(int)
    
    return data

merged_play_data = create_features(merged_play_data)


#####Timing 
end_time = time.monotonic()
print("Time for cell was ",timedelta(seconds=end_time - start_time))

#%% correlation
## Correlation Analysis
# Compute correlation matrix
correlation_matrix = merged_play_data.corr()

## Display correlations with the target variable (e.g., 'successAfterSnap')
target_correlation = correlation_matrix['successAfterSnap'].drop('successAfterSnap')
print("Correlation with target variable:\n", target_correlation)

## Plot a heatmap of correlations with the target variable
plot_correlation_heatmap(merged_play_data, target_column='successAfterSnap')



## Plotting the full correlation matrix as a heatmap
plt.figure(figsize=(20, 12))
sns.heatmap(
    correlation_matrix, 
    annot=False, 
    cmap="coolwarm", 
    cbar=True,
    xticklabels=True, 
    yticklabels=True
)
plt.title("Correlation Matrix Heatmap")
plt.show()


## -----------------  ##
## Flatten the correlation matrix and exclude self-correlation (diagonal = 1)
corr_values = correlation_matrix.values.flatten()
corr_values = corr_values[~np.isnan(corr_values)]  # Exclude NaN values
corr_values = corr_values[corr_values != 1]  

## Compute statistics
mean_corr = corr_values.mean()
min_corr = corr_values.min()
max_corr = corr_values.max()

## Display results
print("Mean Correlation:", mean_corr)
print("Min Correlation:", min_corr)
print("Max Correlation:", max_corr)



## Threshold for high correlation
threshold = mean_corr + 2 * np.std(corr_values)  ## Mean + 2 * Standard Deviation
print("Suggested Threshold:", threshold)



## Find pairs of highly correlated features
high_corr_pairs = correlation_matrix.abs().stack().reset_index()
high_corr_pairs.columns = ['Feature1', 'Feature2', 'Correlation']
high_corr_pairs = high_corr_pairs[
    (high_corr_pairs['Feature1'] != high_corr_pairs['Feature2']) & 
    (high_corr_pairs['Correlation'] > threshold)
]

print("Highly Correlated Feature Pairs:")
print(high_corr_pairs)

## Drop one feature from each pair (second non-ID feature currently)
fields_to_drop = [
    "quarter", "preSnapHomeScore", "preSnapVisitorScore", "rushingYards", "offensiveTendency", 
    "avgRushingYards", "hadRushAttempt", "avgPassingYards", "hadDropback", "passingYards", 
    "receivingYards", "wasTargettedReceiver", "yardageGainedAfterTheCatch", "hadPassReception", 
    "fumbleLost", "fumbleOutOfBounds", "fumbleRecoveries", "fumbles", "hadInterception", 
    "interceptionYards", "sackYardsAsOffense", "sackYardsAsDefense", "tackleForALoss", 
    "tackleForALossYardage", "causedPressure", "quarterbackHit", "soloTackle", "passDefensed", 
    "timeToPressureAsPassRusher", "getOffTimeAsPassRusher", "isDropback", "timeToThrow", 
    "timeInTackleBox", "timeToSack", "pressureRate", "avgTimeToPressure", "playClockAtSnap", 
    "normalizedYardline", "absoluteYardlineNumber", "expectedPoints", 
    "scoreDifferential", "isZoneCoverage", "defensiveBlitzRate", "isPlayClockCritical", 
    "yardsToGo", "down", "isCriticalDown", "isGoalToGo", "isRedZone", "playAction", 
    "dropbackDistance", "expectedPointsAdded", "prePenaltyYardsGained", "yardsGained", 
    "homeTeamWinProbabilityAdded", "visitorTeamWinProbilityAdded", "penaltyYards_y", 
    "pff_runPassOption"
]


reduced_data = merged_play_data.drop(columns=fields_to_drop)

print(f"Dropped features: {fields_to_drop}")
print(f"Reduced data shape: {reduced_data.shape}")


reduced_data.head(5)


## Compute reduced correlation matrix
red_correlation_matrix = reduced_data.corr()

## Plot a heatmap of correlations with the target variable
plot_correlation_heatmap(reduced_data, target_column='successAfterSnap')


## Plotting the full correlation matrix as a heatmap
plt.figure(figsize=(20, 12))
sns.heatmap(
    red_correlation_matrix, 
    annot=False, 
    cmap="coolwarm", 
    cbar=True,
    xticklabels=True, 
    yticklabels=True
)
plt.title("Correlation Matrix Heatmap")
plt.show()

#####Timing 
end_time = time.monotonic()
print("Time for cell was ",timedelta(seconds=end_time - start_time))


#%% Data exploration & Label encoding and data fixing

## General data overview
print("Shape of the DataFrame:", reduced_data.shape)
print("Column Data Types:\n", reduced_data.dtypes)
print("Null Values in Each Column:\n", reduced_data.isnull().sum())
print("Descriptive Statistics:\n", reduced_data.describe(include='all'))

## Check for unique values in categorical columns
for col in reduced_data.columns:
    if reduced_data[col].dtype == 'int64' or reduced_data[col].dtype == 'float64':
        continue
    print(f"Unique values in {col}:\n", reduced_data[col].value_counts())

## Plot distributions for numerical columns
numerical_columns = reduced_data.select_dtypes(include=['int64', 'float64']).columns

for col in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(reduced_data[col], kde=True, bins=30)  ##KDE for density check
    plt.title(f"Distribution of {col}")
    plt.show()

### ---------  ### Dealing with nulls

###reduced_data.info()

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import cross_validate

# # Function for comparing different approaches
# def score_dataset(X_train, X_test, y_train, y_test):
#     model = RandomForestRegressor(n_estimators=10, random_state=0)
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     return mean_absolute_error(y_test, preds)


# ####USing above function to use imputation to see score and output of function
# from sklearn.impute import SimpleImputer

# # Imputation
# my_imputer = SimpleImputer()
# imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
# imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))

# # Imputation removed column names; put them back
# imputed_X_train.columns = X_train.columns
# imputed_X_test.columns = X_test.columns

# print("MAE from Approach 2 (Imputation):")
# print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))


## Verify that nulls are handled
print("Null Values BEFORE Handling:\n", reduced_data.isnull().sum())

new_data = drop_nulls(reduced_data, .6)
new_data.info()

## Replace all NaN values in the DataFrame with 0
new_data.fillna(0, inplace=True)

## Verify that nulls are handled
print("Null Values AFTER Handling:\n", new_data.isnull().sum())

### --------- ### Scaling

# from sklearn.preprocessing import MinMaxScaler

# # Scale numerical columns using MinMaxScaler
# scaler = MinMaxScaler()
# reduced_data[numerical_columns] = scaler.fit_transform(reduced_data[numerical_columns])

# # Display scaled data
# print("Data after scaling:\n", reduced_data.head())

### --------- ### label encoding

## Identify non-integer columns
non_int_columns = [col for col in new_data.columns if new_data[col].dtypes not in ['int64', 'float64']]

## Display non-integer columns
print("Non-integer columns for label encoding:", non_int_columns)

## label encode all non int columns 
label_encode_columns(new_data, non_int_columns)

## double check for actual 
test = new_data[new_data['teamAbbr'] == 0] 

#%% Feature Importance with Random Forest

## Prepare data for feature importance analysis
predictive_columns = [  
    col for col in new_data.columns if col not in ['gameId', 'playId', 'nflId', 'successAfterSnap']
]
target_column = 'successAfterSnap'

X = new_data[predictive_columns]
y = new_data[target_column]

## Train a RandomForestClassifier for feature importance
clf = RandomForestClassifier(random_state=56)
clf.fit(X, y)

## Extract and sort feature importances
feature_importance = pd.DataFrame({
    'Feature': predictive_columns,
    'Importance': clf.feature_importances_
}).sort_values(by='Importance', ascending=False)

## Display top features
print("Feature Importance:\n", feature_importance)

## Plot feature importance
plot_feature_importance(feature_importance,  "Random Forest", 'Importance', top_n=15)


#####Timing 
end_time = time.monotonic()
print("Time for cell was ",timedelta(seconds=end_time - start_time))

#%%  CHI ^2

chi_scores = chi_squared_feature_importance(X, y, k=15)


plot_feature_importance(chi_scores, "Chi-Square", "Chi2 Score", top_n=15)


#####Timing 
end_time = time.monotonic()
print("Time for cell was ",timedelta(seconds=end_time - start_time))

#%%  MI

mi_scores = mutual_info_feature_importance(X, y) 


plot_feature_importance(mi_scores, "Mutual Information", 'MI Score', top_n=15)


#####Timing 
end_time = time.monotonic()
print("Time for cell was ",timedelta(seconds=end_time - start_time))


#%%  RFE   ***Time consuming

rfe_scores = rfe_feature_importance(X, y, n_features_to_select=15)

plot_feature_importance( rfe_scores, "Recursive Feature Elimination", 'RFE Rank' , top_n=15 )


#####Timing 
end_time = time.monotonic()
print("Time for cell was ",timedelta(seconds=end_time - start_time))



#%%  Permutation  (****  TOO Much time to run  ****))

# perm_scores = permutation_feature_importance(clf, X, y, classification=True)

# plot_feature_importance(perm_scores, "Permutation" , '', top_n=15)

# #####Timing 
# end_time = time.monotonic()
# print("Time for cell was ",timedelta(seconds=end_time - start_time))

'''
ALL ABOVE is just on games and player_play / plays
NO weekly tracking involved yet and score for model is built on the below:

    # Yardage gain
    if row['down'] == 1 and row['yardsGained'] >= 4:
        yardage_success = True
    elif row['down'] == 2 and row['yardsGained'] >= row['yardsToGo'] / 2:
        yardage_success = True
    elif row['down'] in [3, 4] and row['yardsGained'] >= row['yardsToGo']:
        yardage_success = True
    else:
        yardage_success = False

    # No sack
    no_sack = row['passResult'] != 'S'

    # No snap penalty
    no_penalty = row['playNullifiedByPenalty'] == 'N'

    # No interception
    no_interception = row['passResult'] != 'IN'

    #  No fumble
    no_fumble = row['fumbles'] == 0
    
**Snap data is in weekly data!!! ['frameType'] == 'BEFORE_SNAP' / 'AFTER_SNAP' (also 'DURING')

'''

#%%  minimize tracking


### JUST WEEK 1 FOR TESTING!!! ------------------------------------------------
tracking_combined = tracking_wk1

## Select key columns
columns_to_keep = ['gameId', 'playId', 'nflId', 'x', 'y', 's', 'a', 'dis', 'o', 'dir', 'event']
filtered_data = tracking_combined[columns_to_keep]


before_snap_df = tracking_combined[tracking_combined['frameType'] == "BEFORE_SNAP"].copy()


## Aggregation for 'beforethesnap' data
aggregated_before_snap = before_snap_df.groupby(['gameId', 'playId']).agg({
    'x': ['mean', 'max', 'min'],       # Average, max, and min x-coordinates
    'y': ['mean', 'max', 'min'],       # Average, max, and min y-coordinates
    's': ['mean', 'max'],              # Average and max speed
    'a': ['mean', 'max'],              # Average and max acceleration
    'dis': 'sum',                      # Total distance traveled
    'o': ['mean'],                     # Average orientation
    'dir': ['mean']                    # Average direction
}).reset_index()

## Rename columns for clarity
aggregated_before_snap.columns = [
    'gameId', 'playId', 
    'mean_x', 'max_x', 'min_x',
    'mean_y', 'max_y', 'min_y',
    'mean_speed', 'max_speed', 
    'mean_acceleration', 'max_acceleration',
    'total_distance', 
    'mean_orientation', 'mean_direction'
]


##Merge aggregated features with main data
merged_data = pd.merge(new_data, aggregated_before_snap, on=['gameId', 'playId'], how='left')




#####Timing 
end_time = time.monotonic()
print("Time for cell was ",timedelta(seconds=end_time - start_time))


#%%
## Correlation Analysis
plot_correlation_heatmap(merged_play_data, target_column='yardsGained')


#####Timing 
end_time = time.monotonic()
print("Time for cell was ",timedelta(seconds=end_time - start_time))

#%%
## Model Training and Evaluation
predictive_columns = [
    'isCriticalDown', 'scoreDifferential', 'normalizedYardline', 'quarterTimeLeft', 'isZoneCoverage', 'isRedZone'
]
target_column = 'successAfterSnap'

X = merged_play_data[predictive_columns]
y = merged_play_data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=56)


#####Timing 
end_time = time.monotonic()
print("Time for cell was ",timedelta(seconds=end_time - start_time))

#%%
model = RandomForestClassifier(random_state=56)
model.fit(X_train, y_train)

## Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#####Timing 
end_time = time.monotonic()
print("Time for cell was ",timedelta(seconds=end_time - start_time))

#%% AFTER Feature Importance
feature_importance = pd.DataFrame({
    'Feature': predictive_columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plot_feature_importance(feature_importance)

## Timing End
end_time = time.monotonic()
print("Elapsed Time: ", timedelta(seconds=end_time - start_time))


#####Timing 
end_time = time.monotonic()
print("Time for cell was ",timedelta(seconds=end_time - start_time))


#%% SHAP

import shap

## Initialize SHAP explainer
explainer = shap.Explainer(model, X_train)

## Calculate SHAP values
shap_values = explainer(X_test)

## Visualize SHAP summary plot
shap.summary_plot(shap_values, X_test)

## Feature contribution
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

## Single feature dependance
shap.dependence_plot("yardsToGo", shap_values, X_test)





#####Timing 
end_time = time.monotonic()
print("Time for cell was ",timedelta(seconds=end_time - start_time))

