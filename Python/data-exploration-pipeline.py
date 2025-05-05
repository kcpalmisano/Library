## Function to optimize data types
def optimize_dtypes(df):
    for col in df.columns:
        ## Convert date columns
        if 'date' in col or 'updated_on' in col or 'dts' in col:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        ## Convert categorical text columns
        elif df[col].dtype == 'object' and df[col].nunique() < 0.5 * len(df):
            df[col] = df[col].astype('category')
        
        ## Convert numerical columns (integers first)
        elif pd.api.types.is_numeric_dtype(df[col]):
            if pd.api.types.is_integer_dtype(df[col].dropna()):  
                df[col] = df[col].astype('Int64')  ## Keeps NaNs
            else:
                df[col] = df[col].astype('float32')  ## Optimized float
    
    return df

## Apply optimization
df = optimize_dtypes(df)



#Pipeline
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
            annot=True,
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
                print(f"🧹 Dropping '{feat1}' (higher variance)")
            else:
                drop_cols.add(feat2)
                print(f"🧹 Dropping '{feat2}' (higher variance)")

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
### Load data / Clean data
df = clean_data(
    df,   #### <----- Data frame
    missing_threshold=0.6,
    drop_missing=None,  ## function will ask 
    log_scale=False 
)

###Explore structure
audit_column_types(df)

### Correlations show all and top correlated paris
top_corrs = correlation_matrix(df, top_n=10)

##Pick a method and drop higher variance features
preferred_method = 'pearson'  ## "pearson", "spearman", "kendall"
top_corr_preferred = top_corrs[preferred_method]
df_corr_pruned = drop_correlated_features_by_variance(df, top_corr_preferred, threshold=0.9)

### Scatter plot top correlated pairs (before dropping is fine too)
scatter_top_correlations(df, top_corr_preferred)

### Outlier detection (optional drop)
_, df_no_outliers = detect_outliers_iqr(df_corr_pruned, top_n=10, drop_outliers=True)

### Distributions
plot_numeric_distributions(df_no_outliers)  

### Categorical Distributions
cat_cols = ['group_action']  ## add more if needed
for col in cat_cols:
    plot_categorical_distribution(df, col)     ##cat column comparison or single column

### Subset / join if needed
##df_subset = subset_data(df, {"region": "NA"})
##df_merged = join_data(df_subset, demographics_df, on="member_id", how="left")





