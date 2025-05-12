##Pipeline
def clean_data(
    df,
    missing_threshold=0.6,
    drop_missing=None,
    log_scale=False,
    drop_constant=True
):
    """
    Clean the dataset with optional handling for:
    - High missingness
    - Constant (zero variance) columns
    - Smart date parsing
    - Optional log1p scaling

    Parameters:
    - missing_threshold: % missing to consider high
    - drop_missing: True = drop, False = keep, None = prompt user
    """
    df = df.copy()

    ## Handle missingness
    missing_pct = df.isnull().mean()
    high_missing = missing_pct[missing_pct > missing_threshold].sort_values(ascending=False)

    if drop_missing is None:
        print(f"\n{len(high_missing)} columns have >{int(missing_threshold*100)}% missing values:")
        print(high_missing)
        user_choice = input("\n💡 Drop these columns? (y/n): ").strip().lower()
        drop_missing = user_choice == 'y'

    if drop_missing:
        df.drop(columns=high_missing.index, inplace=True)
        print(f"🧹 Dropped {len(high_missing)} high-missing columns.")
    else:
        print(f"Kept all columns (including {len(high_missing)} high-missing ones).")

    ## Drop constant columns
    if drop_constant:
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        df.drop(columns=constant_cols, inplace=True)
        if constant_cols:
            print(f"Dropped {len(constant_cols)} constant columns.")

    ## Improve dtypes (with safer datetime inference)
    for col in df.select_dtypes(include="object").columns:
        if "date" in col.lower() or "dt" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                continue
            except Exception:
                pass
        if df[col].nunique() < 20:
            df[col] = df[col].astype("category")

    ## Log-scale numeric columns
    if log_scale:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        for col in numeric_cols:
            if (df[col] > 0).all():
                df[col] = np.log1p(df[col])
        print("Applied log1p scaling to numeric columns.")

    ## Clean any leftover infs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def correlation_matrix(
    df,
    methods=["pearson", "spearman", "kendall"],
    top_n=5,
    mask_upper=True,
    figsize=(10, 8)
):
    """
    Show correlation heatmaps for multiple methods and return top correlated pairs.

    Parameters:
    - method: "pearson", "kendall", or "spearman"
        * pearson - linear relation between two continuous variables
        * kendall - monotonic relationships (using ranks, not raw values)
        * spearman - strenght of association between rankings 
    - top_n: number of top pairs to show
    - mask_upper: whether to mask top triangle
    - figsize: heatmap size

    Returns:
    - top_corrs: dict of top pairs for each method
    - top_corr_preferred: Series of top pairs for selected method
    - df_reduced: DataFrame with correlated features dropped (if drop_high_corr=True)
    """
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    top_corrs = {}

    for method in methods:
        print(f"\nCorrelation Matrix using {method.title()} method:")
        corr = numeric_df.corr(method=method)

        # Mask upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else None

        # Plot
        plt.figure(figsize=figsize)
        sns.heatmap(
            corr,
            mask=mask,
            cmap="coolwarm",
            annot=False,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        plt.title(f"{method.title()} Correlation Matrix")
        plt.tight_layout()
        plt.show()

        # Get top correlations
        corr_pairs = corr.abs().unstack().sort_values(ascending=False)
        corr_pairs = corr_pairs[corr_pairs < 1.0].drop_duplicates()
        top_corrs[method] = corr_pairs.head(top_n)

        print(f"\nTop {top_n} Correlated Feature Pairs ({method.title()}):")
        print(top_corrs[method])

    return top_corrs

def drop_correlated_features_by_variance(df, correlation_series, threshold=0.9):
    """
    Drop one of each highly correlated pair, keeping the feature with lower variance.
    """
    drop_cols = set()

    for (feat1, feat2), corr_value in correlation_series.items():
        if abs(corr_value) > threshold:
            var1 = df[feat1].var()
            var2 = df[feat2].var()

            print(f"\nCorrelation between '{feat1}' and '{feat2}': {corr_value:.2f}")
            print(f"   Variance — {feat1}: {var1:.4f}, {feat2}: {var2:.4f}")

            if var1 > var2:
                drop_cols.add(feat1)
                print(f"Dropping '{feat1}' (higher variance)")
            else:
                drop_cols.add(feat2)
                print(f"Dropping '{feat2}' (higher variance)")

    df_reduced = df.drop(columns=list(drop_cols), errors='ignore')
    print(f"\nDropped {len(drop_cols)} highly correlated features.")
    return df_reduced


def scatter_top_correlations(df, top_corr_pairs):
    """
    Scatter plots for top correlated numeric feature pairs.
    """
    for (col1, col2) in top_corr_pairs.index:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df[col1], y=df[col2])
        plt.title(f"{col1} vs {col2}")
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def subset_data(df, filter_dict):
    """
    Create subset of data based on column filters.
    Example: {"region": "EU", "status": ["active", "pending"]}
    """
    df_filtered = df.copy()
    for col, val in filter_dict.items():
        if isinstance(val, list):
            df_filtered = df_filtered[df_filtered[col].isin(val)]
        else:
            df_filtered = df_filtered[df_filtered[col] == val]
    return df_filtered


def join_data(df1, df2, on="id", how="inner", suffixes=("_left", "_right")):
    """
    Join two dataframes on a key or keys.
    """
    df_joined = df1.merge(df2, on=on, how=how, suffixes=suffixes)
    print(f"Joined dataframes on '{on}' using '{how}' join.")
    return df_joined


def plot_categorical_distribution(df, cat_col, rotate=True):
    """
    Plot a bar chart for a categorical column, with smart label rotation.
    """
    plt.figure(figsize=(10, 5))
    ax = df[cat_col].value_counts().plot(kind="bar")

    plt.title(f"Distribution of {cat_col}")
    plt.ylabel("Count")

    if rotate:
        plt.xticks(rotation=45, ha='right')  ## tilted and aligned

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()



def plot_numeric_distributions(df, numeric_cols=None, min_unique=10):
    """
    Plot histograms for numeric columns with enough unique values.
    
    Parameters:
    - min_unique: Minimum number of unique values to consider as "continuous"
    """
    if numeric_cols is None:
        # Default: All numeric columns
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    # Filter by unique value count
    selected_cols = [col for col in numeric_cols if df[col].nunique() >= min_unique]

    if not selected_cols:
        print("No numeric columns met the unique-value threshold for distribution plots.")
        return

    print(f"Plotting {len(selected_cols)} numeric columns with ≥{min_unique} unique values...")

    df[selected_cols].hist(bins=30, figsize=(16, len(selected_cols)//3 * 3 + 3))
    plt.tight_layout()
    plt.show()




def detect_outliers_iqr(df, columns=None, top_n=5, drop_outliers=False):
    """
    Count and optionally drop outliers using the IQR method.

    Parameters:
    - columns: list of numeric columns to check (default = all numeric)
    - top_n: number of columns to display
    - drop_outliers: if True, returns df with outliers removed

    Returns:
    - (outlier_counts, cleaned_df if drop_outliers else original df)
    """
    outlier_counts = {}
    df_clean = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=["int64", "float64"]).columns

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

        outlier_mask = (df[col] < lower) | (df[col] > upper)
        outlier_counts[col] = outlier_mask.sum()

        if drop_outliers:
            df_clean = df_clean[~outlier_mask]

    sorted_outliers = dict(sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True))

    print(f"\nTop {top_n} Outlier Columns (IQR Method):")
    for i, (col, count) in enumerate(sorted_outliers.items()):
        if i >= top_n:
            break
        print(f"{col}: {count} outliers")

    return (sorted_outliers, df_clean if drop_outliers else df)


def audit_column_types(df):
    """
    Print summary of column types and counts.
    """
    dtype_counts = df.dtypes.value_counts()
    #print("\n Column Type Summary:")
    #print(dtype_counts)
    return dtype_counts


## ========== USAGE ========== ##
## Load data / Clean data
df_filtered = clean_data(
    df_filtered,     ###  <------ Data frame
    missing_threshold=0.6,
    drop_missing=None,     ## ask me
    log_scale=False
)

##Explore structure
audit_column_types(df_filtered)

## Correlations show all and top correlated paris
top_corrs = correlation_matrix(df_filtered, top_n=10)

#Pick a method and drop higher variance features
preferred_method = 'pearson'  ## "pearson", "spearman", "kendall"
top_corr_preferred = top_corrs[preferred_method]
df_corr_pruned = drop_correlated_features_by_variance(df_filtered, top_corr_preferred, threshold=0.9)

## Scatter plot top correlated pairs (before dropping is fine too)
scatter_top_correlations(df_filtered, top_corr_preferred)

## Outlier detection (optional drop)
_, df_no_outliers = detect_outliers_iqr(df_corr_pruned, top_n=10, drop_outliers=True)

## Distributions
plot_numeric_distributions(df_filtered) # (df_no_outliers)

## Categorical Distributions
cat_cols = ['city']  # add more if needed
for col in cat_cols:
    plot_categorical_distribution(df_filtered, col)  # (df_no_outliers)   ##cat column comparison or single column

### Subset / join if needed
##df_subset = subset_data(df_filtered, {"region": "NA"})
##df_merged = join_data(df_subset, demographics_df, on="member_id", how="left")



##Select ALL numeric columns
numeric_columns = df_optimized.select_dtypes(include=['number'])

###Full data frame Cor matrix
corr_matrix = numeric_columns.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(60, 20))
sns.heatmap(corr_matrix.corr(), mask= mask, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, square=True)
plt.title("Correlation Matrix with ALL Data")
plt.show()



##Select all DAX numeric columns
numeric_columns_daxf = df_filtered.select_dtypes(include=['number'])

###Full data frame Cor matrix
corr_matrix_daxf = numeric_columns_daxf.corr()

mask_daxf = np.triu(np.ones_like(corr_matrix_daxf, dtype=bool))

plt.figure(figsize=(60, 20))
sns.heatmap(corr_matrix_daxf.corr(), mask= mask_daxf, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, square=True)
plt.title("Correlation Matrix with DAx Data")
plt.show()


## Function to remove outliers using IQR (Interquartile Range)
#def remove_outliers(df, column):
#    Q1 = df[column].quantile(0.25)
#    Q3 = df[column].quantile(0.75)
#    IQR = Q3 - Q1
#    lower_bound = Q1 - 1.5 * IQR
#    upper_bound = Q3 + 1.5 * IQR     
#    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


## Apply outlier removal iteratively
#for col in numeric_columns:
#    numeric_columns_opt = remove_outliers(numeric_columns, col)

###Check after outlier removal

#corr_matrix_opt = df_optimized_opt.corr()

#mask = np.triu(np.ones_like(corr_matrix_opt, dtype=bool))

#plt.figure(figsize=(42, 20))
#sns.heatmap(corr_matrix_opt.corr(), mask= mask, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, square=True)
#plt.title("Correlation Between Income, Spending, and Check-in Behavior")
#plt.show()



## Regression Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import shap
shap.initjs()



def preprocess_data_regression(df: pd.DataFrame, target_column: str, drop_columns: list = None):
    df = df.copy()

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col], _ = pd.factorize(df[col])

    df = df.drop(columns=df.select_dtypes(include=["datetime64"]).columns)
    df = df.fillna(0)

    drop_cols = [target_column]
    if drop_columns:
        drop_cols += [col for col in drop_columns if col in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[target_column]

    ## Fix inf / nan in target
    y = pd.to_numeric(y, errors='coerce')
    y.replace([np.inf, -np.inf], np.nan, inplace=True)
    y = y.fillna(y.median())  ## Can also use dropna() if appropriate

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=56
    )

    return X_train, X_test, y_train, y_test, X.columns



def train_regressor(X_train, y_train, model_type="rf", use_grid_search=True, custom_params=None):
    """
    Train either a Random Forest Regressor or Linear Regression model.

    Parameters:
    - model_type: 'rf' for Random Forest, 'linear' for Linear Regression
    - use_grid_search: Only applies to Random Forest
    """
    if model_type == "linear":
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    elif model_type == "rf":
        if use_grid_search:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'min_samples_split': [2, 4],
                'min_samples_leaf': [1, 2],
            }

            rf = RandomForestRegressor(random_state=56)
            grid_search = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                scoring='neg_mean_squared_error',
                cv=3,
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            print("Best params:", grid_search.best_params_)
            return grid_search.best_estimator_
        else:
            if custom_params is None:
                custom_params = {"n_estimators": 200}

            rf = RandomForestRegressor(random_state=56, **custom_params)
            rf.fit(X_train, y_train)
            return rf
    else:
        raise ValueError("Invalid model_type. Use 'rf' or 'linear'.")


def evaluate_regressor(model, X_test, y_test, feature_names=None, show_shap=False):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.grid(True)
    plt.show()

    if feature_names is not None:
        plot_feature_importance(model, feature_names)

    if show_shap:
        explain_model(model, X_test)


def plot_feature_importance(model, feature_names, top_n=10):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)  # take absolute value of coefficients
    else:
        print("Model does not support feature importance.")
        return

    indices = np.argsort(importances)[-top_n:][::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(range(top_n), importances[indices], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.title("Top Predictive Features")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()

'''
def explain_model(model, X_sample):
    print("Generating SHAP explanations...")
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, X_sample)
'''

def cross_validate_regressor(model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=2):  ## set folds here
    print(f"\nCross-validation ({cv}-fold, scoring={scoring})")
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
    print(f"{scoring.upper()} Mean: {scores.mean():.4f} | Std: {scores.std():.4f}")
    return scores


def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel("Residual")
    plt.title("Distribution of Residuals")
    plt.grid(True)
    plt.show()


## ========== USAGE ========== ##
## Choose model type: "rf" for Random Forest, "linear" for Linear Regression
model_type = "linear"  ## or "rf"

X_train, X_test, y_train, y_test, feature_names = preprocess_data_regression(
    df_filtered,                               ### <----- dataframe 
    target_column="ancel_spend_per_day"        ### <----- target feature
)

## Train
#model = train_regressor(X_train, y_train, use_grid_search=False)

### Custom Parameters 
custom_params = {"n_estimators": 100, "max_depth": 6, "max_features": "sqrt"}
model = train_regressor(X_train, y_train, use_grid_search=False, custom_params=custom_params)


## Evaluate
evaluate_regressor(model,
                  X_test,
                  y_test, 
                   feature_names=feature_names,
                   show_shap=False
                  )

## Cross-validation
cross_validate_regressor(model, X_train, y_train)

## Plot residuals
y_pred = model.predict(X_test)
plot_residuals(y_test, y_pred)


## Stand Alone Model Eval Function
def evaluate_model(y_test, y_pred, y_train, title="Actual vs Predicted Values"):
    ## Calculate Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    var = y_train.var()

    ## Print Metrics
    print(f"Model Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"Variance: {var:.2f}")

    ## Create Scatter Plot
    plt.figure(figsize=(12, 6))
    #sns.scatterplot(x=y_test, y=y_test, color="blue", alpha=0.5, label="Actual Values")
    sns.scatterplot(x=y_test, y=y_pred, color="orange", alpha=0.5, label="Predicted Values")

    plt.xlabel("Actual Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()



##Select all numeric columns, fix inf and nan values 
numeric_data = df_optimized.select_dtypes(include=['number']).copy()
numeric_data.replace([np.inf, -np.inf], np.nan, inplace=True)
numeric_data.fillna(0, inplace=True)

##use median instead of zero ???
##numeric_data.fillna(numeric_data.median(), inplace=True)


### view skewed features

## Compute skewness for all numeric columns
skewness = df_non_zero.skew().sort_values(ascending=False)

## Define threshold for high skewness (|skew| > 1)
highly_skewed_features = skewness[abs(skewness) > 1].index.tolist()

print("Highly Skewed Features:", highly_skewed_features)

## Set up the plot 
num_features = len(highly_skewed_features)
rows = (num_features + 1) // 2  # adjust as needed
plt.figure(figsize=(14, rows * 4))

## Loop through features and plot
for i, feature in enumerate(highly_skewed_features):
    plt.subplot(rows, 2, i + 1)
    sns.histplot(df_non_zero[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()




merged['age'] = pd.to_numeric(merged['age'], errors='coerce')
merged['eqx_age'] = pd.to_numeric(merged['eqx_age'], errors='coerce')


match_counts = merged['age_match'].value_counts().rename({True: 'Match', False: 'Mismatch'})

sns.barplot(x=match_counts.index, y=match_counts.values)
plt.title('Age Match vs Mismatch')
plt.xlabel('Age Comparison')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


## Create age bins (optional: adjust bin edges based on your data)
bins = list(range(0, 101, 10))  ## 0-9, 10-19, ..., 90-99 etm
labels = [f'{b}-{b+9}' for b in bins[:-1]]

merged['age_group'] = pd.cut(merged['age'], bins=bins, labels=labels)
merged['eqx_age_group'] = pd.cut(merged['eqx_age'], bins=bins, labels=labels)

## Confusion matrix-style heatmap
age_heatmap = merged.groupby(['age_group', 'eqx_age_group']).size().unstack(fill_value=0)

sns.heatmap(age_heatmap, annot=True, fmt='d', cmap='coolwarm')
plt.title('Age Group Comparison: DAx vs Eqx')
plt.xlabel('Eqx_age_group')
plt.ylabel('DAx_age_group')
plt.tight_layout()
plt.show()



from scipy.stats import gaussian_kde
 ###takes a long time
plt.figure(figsize=(12, 6))
merged['age_match'] = merged['age_match'].astype(object)
merged['age_match'] = merged['age_match'].fillna("Missing")

# Scatter plot
sns.scatterplot(
    data=merged,
    x='age',
    y='eqx_age',
    hue='age_match',
    palette={True: 'green', False: 'red', "Missing": 'grey'},
    alpha=0.6
)

# Add labels and title
plt.title("Self-Reported vs. EQX Age by Match Status", fontsize=14, fontweight="bold")
plt.xlabel("Self-Reported Age", fontsize=12)
plt.ylabel("EQX Estimated Age", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="Age Match")
plt.tight_layout()
plt.show()


# Remove NaNs for plotting
subset = merged.dropna(subset=['age', 'eqx_age'])

# Get point positions
x = subset['age']
y = subset['eqx_age']

# Calculate density
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

# Plot
plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(x=x, y=y, hue=z, palette='viridis', alpha=0.7, edgecolor=None)
plt.plot([0, 100], [0, 100], linestyle='--', color='gray')

plt.title('Age Comparison with Density Overlay')
plt.xlabel('DAX Ages')
plt.ylabel('EQX Ages')
#plt.colorbar(label='Density')
plt.grid(True)
plt.tight_layout()
plt.show()





import joblib
import shap
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay
                            )
                            
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


def preprocess_data(df: pd.DataFrame, target_column: str, spend_column: str = None, use_smote: bool = False):
    """
    Preprocesses the dataset: encodes categorical variables, fills missing values,
    removes datetime columns, and returns train/test splits.

    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - target_column (str): Column to use as the prediction target (must already exist)
    - spend_column (str): Optional column to drop (e.g., raw spend column)
    - use_smote (bool): Whether to apply SMOTE to balance the training data

    Returns:
    - X_train, X_test, y_train, y_test, feature_names
    """
    df = df.copy()

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # Encode categorical features
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Drop datetime columns
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns
    df = df.drop(columns=datetime_cols)

    # Fill missing values
    df = df.fillna(0)

    # Drop spend column if provided
    drop_cols = [target_column]
    if spend_column and spend_column in df.columns:
        drop_cols.append(spend_column)

    X = df.drop(columns=drop_cols)
    y = df[target_column]

    # Standard (non-stratified) split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=56  # no stratify
    )

    # Apply SMOTE
    if use_smote:
        if y_train.nunique() < 2:
            raise ValueError("Cannot apply SMOTE: target has only one class in training set.")
        sm = SMOTE(random_state=56)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, X.columns


def train_classifier(X_train, y_train, use_grid_search=True, custom_params=None):
    """
    Train a RandomForestClassifier with optional grid search or custom parameters.
    """
    if use_grid_search:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'min_samples_split': [2, 4],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced']
        }

        rf = RandomForestClassifier(random_state=56)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            scoring='f1',
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        print("Best params:", grid_search.best_params_)
        return grid_search.best_estimator_
    else:
        if custom_params is None:
            custom_params = {"n_estimators": 200, "class_weight": "balanced"}

        rf = RandomForestClassifier(random_state=56, **custom_params)
        rf.fit(X_train, y_train)
        return rf


def evaluate_model(model, X_test, y_test, show_roc=True, show_precision_recall=True, show_shap=False, feature_names=None):
    """
    Evaluate classification performance and optionally show ROC, PR curve, SHAP values.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    auc_score = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC: {auc_score:.4f}")

    if show_roc:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

    if show_precision_recall:
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label="Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision vs Recall")
        plt.grid(True)
        plt.legend()
        plt.show()

    if feature_names is not None:
        plot_feature_importance(model, feature_names)




def plot_feature_importance(model, feature_names, top_n=10):
    """
    Bar plot of top N feature importances.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:][::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(range(top_n), importances[indices], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.title("Top Predictive Features")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()


def explain_model(model, X_sample):
    print("Generating SHAP explanations...")
    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        shap.summary_plot(shap_values, X_sample)
    except Exception as e:
        print("SHAP explanation failed:", e)


def plot_confusion_matrix(model, X_test, y_test):
    """
    Plot confusion matrix.
    """
    print("\n Confusion Matrix:")
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()


def cross_validate_model(model, X_train, y_train, scoring="f1", cv=5):
    """
    Evaluate model using cross-validation.
    """
    print(f"\n Cross-validation ({cv}-fold, scoring={scoring})")
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
    print(f"{scoring.upper()} Mean: {scores.mean():.4f} | Std: {scores.std():.4f}")
    return scores


def describe_features(X):
    print("\nFeature Summary:")
    try:
        from IPython.display import display
        display(X.describe(include='all').T)
    except ImportError:
        print(X.describe(include='all').T)




## ========== USAGE ========== ##
##add target variable (if needed)
df_dax_reduced["will_spend"] = (df_dax_reduced["ancel_spend_per_day"] > 0).astype(int)

# Preprocess your dataset
X_train, X_test, y_train, y_test, feature_names = preprocess_data(
    df_dax_reduced,
    target_column="will_spend",
    spend_column="ancel_spend_per_day",
    use_smote=False
)

#Best params: {'class_weight': 'balanced', 'max_depth': 4, 'min_samples_leaf': 2, 'min_samples_split': 8, 'n_estimators': 100}
#custom_rf_params = {
 #   "n_estimators": 100,
 #   "max_depth": 4,
 #   "min_samples_split": 8,
 #   "min_samples_leaf": 2,
 #   "class_weight": "balanced"
#}

# Train the model
model = train_classifier(X_train, y_train, use_grid_search=False)

## View feature summary
#describe_features(pd.DataFrame(X_train, columns=feature_names))

## Cross-validation
##cross_validate_model(model, X_train, y_train)

# Evaluate the model
evaluate_model(
    model,
    X_test,
    y_test,
    show_roc=True,
    show_precision_recall=True,
    feature_names=feature_names
)

## Confusion matrix
##plot_confusion_matrix(model, X_test, y_test)
