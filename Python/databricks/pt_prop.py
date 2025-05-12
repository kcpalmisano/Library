
#***The data is loaded in first and changed to a pandas dataframe. 
#**This was a databricks notebook so forgive the formatting

## List of columns to convert
pt_cols = ['pt_usage_1m', 'pt_usage_2m', 'pt_usage_3m', 
           'pt_usage_4m', 'pt_usage_5m', 'pt_usage_6m']

spend_cols = ['spend_month_1m', 'spend_month_2m', 'spend_month_3m', 'spend_month_4m', 'spend_month_5m', 'spend_month_6m']

# Convert pt_cols and spend_cols to numeric more efficiently
data[pt_cols + spend_cols] = data[pt_cols + spend_cols].apply(pd.to_numeric, errors='coerce')
data['usage_mtd_paid_count'] = pd.to_numeric(data['usage_mtd_paid_count'], errors='coerce')

# Convert dates using vectorized functions
data['last_used_paid_date'] = pd.to_datetime(data['last_used_paid_date'], errors='coerce')
data['summary_month_key'] = pd.to_datetime(data['summary_month_key'].astype(str), format='%Y%m%d', errors='coerce')

# Use NumPy for faster boolean operations
data['pt_6m_usage_flag'] = (data[pt_cols].to_numpy() > 0).any(axis=1).astype(np.uint8)
data['pt_usage_flag'] = (data['usage_mtd_paid_count'].to_numpy() > 0).astype(np.uint8)
data['spend_usage_flag'] = (data[spend_cols].to_numpy() > 0).any(axis=1).astype(np.uint8)

##make a copy
all_modeling_df = data.copy()

###Pipeline to get numbers for each month and timeframe
class PTConversionAnalyzer:
    def __init__(self, df, manual_mode=False, manual_date=None, lookback_months=6):   ###  <--- lookback period 
        self.df = df.copy()
        self.manual_mode = manual_mode
        self.manual_date = pd.to_datetime(manual_date) if manual_date else None
        self.lookback_months = lookback_months
        self._setup_dates()

    def _setup_dates(self):
        today = pd.Timestamp.today().normalize()
        if self.manual_mode and self.manual_date:
            self.target_month_start = self.manual_date.replace(day=1)
        else:
            self.target_month_start = (today - pd.DateOffset(months=1)).replace(day=1)

        self.target_month_end = self.target_month_start + pd.offsets.MonthEnd(0)
        self.lookback_start = self.target_month_start - pd.DateOffset(months=self.lookback_months)
        self.lookback_end = self.target_month_start - pd.DateOffset(days=1)

    def analyze(self):
        df = self.df.copy()

        ## Flag usage
        df['used_in_target_month'] = (
            (df['summary_month_key'] >= self.target_month_start) &
            (df['summary_month_key'] <= self.target_month_end) &
            (df['pt_usage_flag'] == 1)
        )

        df['used_in_lookback'] = (
            (df['summary_month_key'] >= self.lookback_start) &
            (df['summary_month_key'] <= self.lookback_end) &
            (df['pt_usage_flag'] == 1)
        )

        ## Member-level usage flags
        member_flags = (
            df.groupby('member_id').agg({
                'used_in_target_month': 'max',
                'used_in_lookback': 'max'
            }).reset_index()
        )

        ## PT conversion flag
        member_flags['pt_conversion'] = (
            (member_flags['used_in_target_month'] == 1) &
            (member_flags['used_in_lookback'] == 0)
        ).astype(int)

        ## Bring in latest metadata (from target month only)
        latest_metadata = df[
            df['summary_month_key'] == self.target_month_end
        ][['member_id', 'member_status_desc', 'tenure_in_months']].drop_duplicates('member_id')

        member_flags = member_flags.merge(latest_metadata, on='member_id', how='left')

        ## Filter to active members only
        active_flags = member_flags[member_flags['member_status_desc'] == 'Active']

        ## === Metric calculations ===
        ## All unique users (still full dataset)
        total_users = df['member_id'].nunique()

        ## Active members (now used everywhere below where appropriate) will change to monthly
        active_members = active_flags['member_id'].nunique()
        
        ## active mebers total
        total_active_users = active_flags['member_id'].nunique()

        ## PT usage in target month — among active only
        used_in_target = active_flags['used_in_target_month'].sum()

        ## Used PT in lookback (active only)
        used_pt_lookback_active = active_flags[active_flags['used_in_lookback'] == 1]['member_id'].nunique()

        ## Did NOT use PT in lookback (active only)
        did_not_use_pt_lookback_active = active_flags[active_flags['used_in_lookback'] == 0]['member_id'].nunique()

        ## Converted users = used in target but NOT in lookback (active only)
        converted_users = active_flags[active_flags['pt_conversion'] == 1]['member_id'].nunique()

        ## Never used PT (no use in lookback or target — active only)
        never_used_pt = active_flags[
            (active_flags['used_in_target_month'] == 0) &
            (active_flags['used_in_lookback'] == 0)
        ]['member_id'].nunique()

        ## Conversion rate among eligible active users
        conversion_rate = (
            converted_users / did_not_use_pt_lookback_active
            if did_not_use_pt_lookback_active > 0 else 0
        )

        ## New members = tenure < 6 months (among active)
        new_members = active_flags[active_flags['tenure_in_months'] < 6]['member_id'].nunique()

        ## Converted new members (among active)
        converted_new_members = active_flags[
            (active_flags['pt_conversion'] == 1) &
            (active_flags['tenure_in_months'] < 6)
        ]['member_id'].nunique()

        ## New member conversion rate
        new_members_conversion = (
            converted_new_members / new_members
            if new_members > 0 else 0
        )


        results = {
            "Total users in dataset": total_active_users,
            "Active members in target month": active_members,
            "Used PT in target month": used_in_target,
            "Used PT in lookback (among active)": used_pt_lookback_active,
            "Did NOT use PT in lookback (among active)": did_not_use_pt_lookback_active,
            "Converted users": converted_users,
            "Never used PT": never_used_pt,
            "Conversion rate": conversion_rate,
            "New members (< 6 months)": new_members,
            "New member conversions": converted_new_members,
            "New member conversion rate": new_members_conversion
        }

        self.member_flags = member_flags 

        return results

    def print_results(self, results):
        #print(f"-Total users in dataset: {results['Total users in dataset']:,}")
        print(f"-Active members in {self.target_month_start.strftime('%B %Y')}: {results['Active members in target month']:,}")
        print(f"-Used PT in target month: {results['Used PT in target month']:,}")
        print(f"-Used PT in lookback (among active): {results['Used PT in lookback (among active)']:,}")
        print(f"-Did NOT use PT in lookback (among active): {results['Did NOT use PT in lookback (among active)']:,}")
        print(f"-Converted users (used PT in target but not in lookback): {results['Converted users']:,}")
        print(f"-Never used PT (in target or lookback): {results['Never used PT']:,}")
        print(f"-Conversion rate (conversions / eligible base): {results['Conversion rate']:.2%}")
        print(f"-New members (tenure < 6 months): {results['New members (< 6 months)']:,}")
        print(f"-New member conversions: {results['New member conversions']:,}")
        print(f"-New Member Conversion rate (conversions / new members): {results['New member conversion rate']:.2%}")


target_month = '2025-03-01'

analyzer = PTConversionAnalyzer(eqx_data, manual_mode=True, manual_date=target_month)
results = analyzer.analyze()
member_flags = analyzer.member_flags
analyzer.print_results(results)

def build_modeling_df_from_analyzer_base(all_users_df, member_flags, target_month):
    """
    Constructs a modeling DataFrame based on the analyzer's definition of active members
    in the target month, with a single row per member including metadata and pt_conversion label.
    
    Parameters:
        all_users_df (pd.DataFrame): Full input dataset.
        member_flags (pd.DataFrame): Analyzer's member-level summary (includes pt_conversion, flags, etc.).
        target_month (str or pd.Timestamp): Target month (e.g., '2025-02').

    Returns:
        pd.DataFrame: Cleaned, aligned modeling DataFrame with one row per member.
    """
    # Ensure target_month is timestamp
    target_month = pd.to_datetime(target_month)
    target_month_end = target_month + pd.offsets.MonthEnd(0)

    # Get active members from the analyzer
    active_members = member_flags[member_flags['member_status_desc'] == 'Active']['member_id'].unique()

    # Filter to latest metadata rows for those members in target month
    modeling_df = all_users_df[
        (all_users_df['summary_month_key'] == target_month_end) &
        (all_users_df['member_id'].isin(active_members))
    ].drop_duplicates(subset='member_id')

    # Join with member_flags to bring in pt_conversion and usage flags
    modeling_df = modeling_df.merge(
        member_flags[['member_id', 'pt_conversion', 'used_in_target_month', 'used_in_lookback']],
        on='member_id',
        how='left'
    )

    return modeling_df

## various dataframe set ups

##Set up flags and data
member_flags_clean = member_flags.drop(columns=['tenure_in_months', 'member_status_desc'], errors='ignore')
all_users_df = member_flags_clean.copy()

## copy for merged all data
all_users_df = all_users_df.merge(all_modeling_df, on='member_id', how='inner')

## Filter to target month only
target_month_df = all_modeling_df[all_modeling_df['summary_month_key'] == analyzer.target_month_end]

## Drop duplicate member records if needed
latest_month_features = target_month_df.drop_duplicates(subset='member_id')
all_users_df = all_users_df.drop_duplicates(subset='member_id')


modeling_df = latest_month_features.merge(member_flags_clean, on='member_id', how='inner')

## Set up ACTIVE MEMBERS from member data set
active_df = modeling_df[modeling_df['member_status_desc'] == 'Active']

print("member_flags:", member_flags.shape)
print("latest_month_features:", latest_month_features.shape)
print("Merged modeling_df:", active_df.shape)
print("Conversion rate in model data:", active_df['pt_conversion'].mean())
# Based on earlier stats
print("Eligible users in member_flags:", active_df[active_df['used_in_lookback'] == 0].shape[0])
print("Modeled users with features and no lookback usage:", active_df[active_df['used_in_lookback'] == 0].shape[0])
pos = modeling_df['pt_conversion'].sum()
total = len(modeling_df)
print(f"Conversion rate: {pos / total:.2%}")

## Feature Engineering for PT Conversion Model
def add_engineered_features(df):

    float_cols = [
        'gf_usage_avg', 'gf_usage_avg_l3m', 'pl_usage_avg', 'pl_usage_avg_l3m',
        'vod_usage_avg_l3m', 'vod_usage_avg_l6m',
        'pref_club_av_checkin_perc_l6m', 'pref_club_age'
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Pre-define column groups for reuse
    gf_cols = [f'gf_usage_{i}m' for i in range(1, 7)]
    pl_cols = [f'pl_usage_{i}m' for i in range(1, 7)]
    checkin_cols = [f'checkin_perc_{i}m' for i in range(1, 7)]

    # Trend-Based Features
    df['gf_usage_momentum'] = df[gf_cols[:3]].mean(axis=1) - df[gf_cols[3:]].mean(axis=1)
    df['pl_usage_momentum'] = df[pl_cols[:3]].mean(axis=1) - df[pl_cols[3:]].mean(axis=1)
    df['vod_usage_momentum'] = df[[f'vod_usage_{i}m' for i in range(1, 4)]].mean(axis=1) - df[[f'vod_usage_{i}m' for i in range(4, 7)]].mean(axis=1)

    # Engagement Intensity / Variability
    df['checkin_std'] = df[checkin_cols].std(axis=1)
    df['gf_recent_ratio'] = df['gf_usage_avg_l3m'] / (df['gf_usage_avg'] + 0.01)
    df['pl_recent_ratio'] = df['pl_usage_avg_l3m'] / (df['pl_usage_avg'] + 0.01)

    # Behavioral Signals
    df['checkin_perc_per_club_age'] = df['pref_club_av_checkin_perc_l6m'] / (df['pref_club_age'] + 1)

    gf_low_thresh = df['gf_usage_avg'].quantile(0.25)
    gf_recent_thresh = df['gf_usage_avg_l3m'].quantile(0.75)
    df['low_gf_high_recent'] = ((df['gf_usage_avg'] <= gf_low_thresh) & (df['gf_usage_avg_l3m'] >= gf_recent_thresh)).astype(int)

    # Recent usage binary flags
    df['used_gf_recent'] = (df['gf_usage_1m'] > 0).astype(int)
    df['used_pl_recent'] = (df['pl_usage_1m'] > 0).astype(int)
    df['used_vod_recent'] = (df['vod_usage_1m'] > 0).astype(int)
    df['checked_in_recent'] = (df['checkin_perc_1m'] > 0).astype(int)

    # Vectorized recency function
    def recency_vectorized(mat):
        mask = (mat > 0).values
        reverse_idx = mask.shape[1] - np.argmax(mask[:, ::-1], axis=1)
        reverse_idx[~mask.any(axis=1)] = 7
        return reverse_idx

    df['months_since_gf'] = recency_vectorized(df[gf_cols])
    df['months_since_pl'] = recency_vectorized(df[pl_cols])
    df['months_since_checkin'] = recency_vectorized(df[checkin_cols])

    # Drop-off slope (decay)
    df['gf_decay'] = df['gf_usage_1m'] - df['gf_usage_6m']
    df['pl_decay'] = df['pl_usage_1m'] - df['pl_usage_6m']
    df['vod_decay'] = df['vod_usage_1m'] - df['vod_usage_6m']

    # Momentum vs Total
    df['gf_momentum_ratio'] = df['gf_usage_avg_l3m'] / (df['gf_usage_avg'] + 0.01)
    df['pl_momentum_ratio'] = df['pl_usage_avg_l3m'] / (df['pl_usage_avg'] + 0.01)

    # Trend direction
    df['gf_trend'] = (df['gf_usage_1m'] > df['gf_usage_6m']).astype(int)
    df['pl_trend'] = (df['pl_usage_1m'] > df['pl_usage_6m']).astype(int)

    # Session diversity (entropy)
    def entropy_vectorized(mat):
        values = mat.values
        row_sums = np.sum(values, axis=1, keepdims=True)
        probs = values / (row_sums + 1e-9)
        log_probs = np.log2(probs + 1e-6)
        return -np.sum(probs * log_probs, axis=1)

    df['visit_type_entropy'] = entropy_vectorized(df[['gf_usage_avg_l3m', 'pl_usage_avg_l3m', 'vod_usage_avg_l3m']])

    # Lifecycle buckets
    df['tenure_bin_new'] = (df['tenure_in_months'] <= 3).astype(int)
    df['tenure_bin_mid'] = ((df['tenure_in_months'] > 3) & (df['tenure_in_months'] <= 12)).astype(int)
    df['tenure_bin_veteran'] = (df['tenure_in_months'] > 12).astype(int)

    # Interaction features
    df['checkin_std_x_gf_recent'] = df['checkin_std'] * df['used_gf_recent']
    df['gf_momentum_x_recency'] = df['gf_usage_momentum'] * df['gf_momentum_ratio']
    df['pl_decay_x_recent'] = df['pl_decay'] * df['used_pl_recent']

    # === New Interaction Features ===
    df['checkin_std_x_tenure'] = df['checkin_std'] * df['tenure_in_months']
    df['checkin_perc_age_ratio'] = df['pref_club_av_checkin_perc_l6m'] / (df['age'] + 1)
    df['spend_momentum'] = df['av_spend_l3m'] / (df['av_spend_l6m'] + 0.01)
    df['gf_recent_x_momentum'] = df['gf_usage_avg_l3m'] * df['gf_usage_momentum']
    df['months_since_gf_x_avg'] = df['months_since_gf'] * df['gf_usage_avg_l3m']
    df['tenure_checkin_combo'] = df['tenure_in_months'] * df['pref_club_av_checkin_perc_l6m']






    # Cap infinite/NaN values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df
'''
def safely_merge_features(base_df, new_df, keys, new_cols, drop_existing=True):
    """
    Merge engineered features into the base_df without creating _x/_y suffixes.

    Parameters:
        base_df (pd.DataFrame): The main dataset to merge into.
        new_df (pd.DataFrame): The source of new features.
        keys (list or str): Column(s) to merge on.
        new_cols (list): The feature columns to bring over from new_df.
        drop_existing (bool): If True, drop any overlapping columns from base_df before merging.

    Returns:
        pd.DataFrame: A cleanly merged DataFrame.
    """
    # Ensure keys is a list
    keys = [keys] if isinstance(keys, str) else keys

    if drop_existing:
        # Drop only if the column exists in base_df, and it is not a key
        drop_cols = [col for col in new_cols if col in base_df.columns and col not in keys]
        base_df = base_df.drop(columns=drop_cols, errors='ignore')

    # Only keep the keys and new feature columns from new_df
    temp = new_df[keys + new_cols]

    # Perform the clean merge
    merged_df = base_df.merge(temp, on=keys, how='left')

    return merged_df
'''

##add the features
all_users_df = add_engineered_features(all_users_df)

## Filter first
modeling_df = all_users_df[all_users_df['used_in_lookback'] == 0].copy()

## Drop unwanted features
features_to_drop = [
    'used_in_target_month', 'used_in_lookback', 'usage_mtd_paid_count', 'pt_4wk_count', 'pt_usage_flag',
    'purchase_1yr_rolling_amt', 'purchase_tenure_amt', 
    'pt_12wk_count', 'pt_usage_1m', 'pt_usage_2m', 'pt_usage_3m',
    'pt_usage_4m', 'pt_usage_5m', 'pt_usage_6m', 'lm_pt_last_used_facility',
    'has_taken_equifit', 'purchase_lifetime_amt', 'first_purchase_pack_size',
    'purchase_mtd_amt', 'purchase_ytd_amt', 'has_taken_comp_pt',
    'pt_6m_usage_flag', 'used_in_target_month'
]
modeling_df.drop(columns=features_to_drop, errors='ignore', inplace=True)

## Build feature list
feature_list = [
    col for col in modeling_df.columns
    if col not in ['pt_conversion', 'member_id', 'summary_month_key']
]

##sanity check
modeling_df['pt_conversion'].value_counts()

#modeling_df['used_in_lookback'].value_counts()

### Need to do this 
modeling_df = modeling_df[modeling_df['member_status_desc'] == 'Active'].copy()

(modeling_df['member_status_desc'] == 'Active').value_counts()

##Sanity Time frame check 
# Ensure 'summary_month_key' is datetime
all_users_df['summary_month_key'] = pd.to_datetime(all_users_df['summary_month_key'])

# Total distinct members in entire dataset
eligible_users = all_users_df['member_id'].nunique()

# Members in training + test masks (from X) — deduplicated
eligible_in_time_window = modeling_df[['member_id']].drop_duplicates()['member_id'].nunique()

# Members active from lookback_start to target_month (inclusive) — deduplicated
active_users = all_users_df[
    #(all_users_df['summary_month_key'] >= lookback_start) &
    (all_users_df['summary_month_key'] <= target_month)
][['member_id']].drop_duplicates()['member_id'].nunique()

print("\n===== Sanity Check: Time Frame Coverage =====")
print(f"Total eligible users: {eligible_users:,}")
print(f"Eligible users active during training or test window: {eligible_in_time_window:,}")
#print(f"Active users from {lookback_start.date()} to {target_month.date()}: {active_users:,}")
print(f"Coverage rate: {active_users / eligible_users:.2%}")


## --- Setup ---
target_dt = pd.to_datetime(target_month).replace(day=1)
lookback_start = target_dt - pd.DateOffset(months=6)
lookback_end = target_dt - pd.DateOffset(days=1)
target_month_end = target_dt + pd.offsets.MonthEnd(0)

## --- Eligible users all-time (no PT in lookback, active members only) ---
eligible_all_time = active_df[active_df['used_in_lookback'] == 0]
print(f"Total eligible users (active, no PT in lookback): {eligible_all_time.shape[0]:,}")

## --- Train/Test masks ---
train_window = (
    (modeling_df['summary_month_key'] >= lookback_start) &
    (modeling_df['summary_month_key'] <= lookback_end)
)
test_window = modeling_df['summary_month_key'] == target_month_end

## --- Training Stats ---
train_df = modeling_df[train_window]
train_rows = train_df.shape[0]
train_conversions = train_df['pt_conversion'].sum()
train_rate = train_conversions / train_rows if train_rows > 0 else 0

## --- Testing Stats ---
test_df = modeling_df[test_window]
test_rows = test_df.shape[0]
test_conversions = test_df['pt_conversion'].sum()
test_rate = test_conversions / test_rows if test_rows > 0 else 0

## --- Output ---
print(f"\nTraining window: {lookback_start.date()} to {lookback_end.date()}")
print(f"Training rows: {train_rows:,} | Conversions: {train_conversions:,} | Rate: {train_rate:.2%}")

print(f"\nTesting month: {target_month_end.date()}")
print(f"Testing rows: {test_rows:,} | Conversions: {test_conversions:,} | Rate: {test_rate:.2%}")

modeling_df.groupby(['summary_month_key', 'pt_conversion'])['member_id'].nunique().unstack(fill_value=0)

target_dt = pd.to_datetime("2025-02-01")
target_end = target_dt + pd.offsets.MonthEnd(0)

# Show number of ACTIVE users in the target month
print(all_users_df[
    (all_users_df['summary_month_key'] == target_end) &
    (all_users_df['member_status_desc'] == 'Active')
]['member_id'].nunique())


modeling_df[
    (modeling_df['summary_month_key'] == target_end) &
    (modeling_df['member_status_desc'] == 'Active')
]['member_id'].nunique() 

##base pipeline for model 

def train_catboost_pipeline(
    dataframe,
    target_col='pt_conversion',
    feature_list=None,
    best_params=None,
    do_search=False,
    test_size=0.2,
    random_state=56,
    threshold=0.5,
    target_month=None,
    calibrate=False,
    top_k_fraction=0.1
):
    class ThresholdedPipeline(Pipeline, ClassifierMixin):
        def __init__(self, steps, threshold=0.5):
            super().__init__(steps)
            self.threshold = threshold

        def predict(self, X):
            probs = self.predict_proba(X)[:, 1]
            return (probs >= self.threshold).astype(int)

    # --- Dates --- #
    dataframe = dataframe.copy()
    dataframe['summary_month_key'] = pd.to_datetime(dataframe['summary_month_key'])
    target_dt = pd.to_datetime(target_month).replace(day=1)
    lookback_start = target_dt - pd.DateOffset(months=6)
    lookback_end = target_dt - pd.DateOffset(days=1)
    print(f"\n[INFO] Training on full 6-month window: {lookback_start.date()} to {lookback_end.date()}")
    print(f"[INFO] Testing on target month: {target_dt.date()}")

    train_df = dataframe[
        (dataframe['summary_month_key'] >= lookback_start) &
        (dataframe['summary_month_key'] <= lookback_end)
    ].copy()
    test_df = dataframe[
        dataframe['summary_month_key'].dt.to_period('M') == target_dt.to_period('M')
    ].copy()

    if feature_list is None:
        feature_list = [
            col for col in train_df.columns
            if col not in [target_col, 'member_id', 'summary_month_key']
            and not np.issubdtype(train_df[col].dtype, np.datetime64)
        ]

    for df_ in [train_df, test_df]:
        datetime_cols = df_.select_dtypes(include='datetime64[ns]').columns.tolist()
        if datetime_cols:
            print(f"[INFO] Dropping datetime columns: {datetime_cols}")
            df_.drop(columns=datetime_cols, inplace=True, errors='ignore')

    X_train = train_df[feature_list]
    y_train = train_df[target_col]
    X_test = test_df[feature_list]
    y_test = test_df[target_col]

    print(f"\n[INFO] Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")
    if y_train.nunique() < 2:
        raise ValueError("[ERROR] y_train contains only one class.")

    for df_ in [X_train, X_test]:
        df_.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_.fillna(0, inplace=True)

    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    print(f"pos_weight: {pos_weight:.2f} | positives: {y_train.sum():,.0f}, negatives: {(~y_train.astype(bool)).sum():,.0f}")

    preprocessor = Pipeline(steps=[
        ('convert_numeric', FunctionTransformer(
            lambda df: df.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtype == 'object' else col),
            validate=False)),
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ])

    base_model = CatBoostClassifier(
        eval_metric='AUC', class_weights=[1, pos_weight],
        random_seed=random_state, verbose=0
    )

    pipeline = ThresholdedPipeline(steps=[
        ('preprocessing', preprocessor),
        ('model', base_model)
    ], threshold=threshold)

    if do_search:
        param_grid = {
            'model__iterations': [100, 300, 500, 1000],
            'model__depth': [4, 6, 8, 10],
            'model__learning_rate': [0.01, 0.03, 0.06, 0.1],
            'model__l2_leaf_reg': [3, 5, 7, 9],
            'model__random_strength': [ 0.5, 1, 2],
            'model__bagging_temperature': [0.0, 0.3, 0.5, 0.7, 1.0],
            'model__border_count': [32, 64]
        }
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=75,
            scoring='roc_auc',
            cv=3,
            verbose=2,
            n_jobs=-1,
            random_state=random_state
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        print("\n[INFO] Best hyperparameters found:")
        print(search.best_params_)
    else:
        pipeline.fit(X_train, y_train)

    if calibrate:
        pipeline.named_steps['model'] = CalibratedClassifierCV(pipeline.named_steps['model'], cv=3, method='isotonic')
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print(f"\nClassification Report @ Threshold = {threshold}")
    print(classification_report(y_test, y_pred))
    print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.title(f"Precision-Recall Curve (AUC={pr_auc:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.grid(); plt.tight_layout(); plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("ROC Curve"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

    model = pipeline.named_steps['model']
    if hasattr(model, 'feature_importances_'):
        fi_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        print("\nTop 25 Features:")
        display(fi_df.head(25))

    # --- Business Metrics ---
    top_k = int(len(y_test) * top_k_fraction)
    top_indices = np.argsort(y_prob)[-top_k:][::-1]
    lift = y_test.iloc[top_indices].mean() / y_test.mean()
    recall_at_top = y_test.iloc[top_indices].sum() / y_test.sum()

    print(f"\nLift@Top {int(top_k_fraction * 100)}%: {lift:.2f}")
    print(f"Recall@Top {int(top_k_fraction * 100)}%: {recall_at_top:.2%}")

    return pipeline, X_test, y_test, y_pred, y_prob

## Decile / lift / probability
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV

def evaluate_lift_and_recall(y_true, y_scores, top_percents=[0.01, 0.05, 0.10]):
    df = pd.DataFrame({'y_true': y_true, 'y_score': y_scores})
    df = df.sort_values('y_score', ascending=False).reset_index(drop=True)
    total_positives = df['y_true'].sum()

    print(f"\nTotal Positives in Data: {total_positives:,}\n")
    print("Top % | Users | Positives | Recall | Lift")
    print("------|--------|-----------|--------|------")
    for p in top_percents:
        cutoff = int(len(df) * p)
        selected = df.iloc[:cutoff]
        positives = selected['y_true'].sum()
        recall = positives / total_positives
        lift = (positives / cutoff) / (total_positives / len(df))
        print(f"{int(p*100):>4d}% | {cutoff:6,} | {positives:9,} | {recall:6.2%} | {lift:5.2f}")


def calibrate_model(pipeline, X_val, y_val, method='isotonic'):
    base_model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessing']

    X_val_prepped = preprocessor.transform(X_val)

    calibrated_model = CalibratedClassifierCV(base_model, method=method, cv='prefit')
    calibrated_model.fit(X_val_prepped, y_val)

    ## Return calibrated pipeline
    return make_pipeline(preprocessor, calibrated_model)


#### Call the model 
## Define feature list
feature_list = [
    col for col in modeling_df.columns
    if col not in ['pt_conversion', 'member_id', 'summary_month_key']
    and not np.issubdtype(modeling_df[col].dtype, np.datetime64)
]

## Best parameters
best_params = {
    'model__random_strength': 1,
    'model__learning_rate': 0.01,
    'model__l2_leaf_reg': 5,
    'model__iterations': 100,
    'model__depth': 4,
    'model__border_count': 64,
    'model__bagging_temperature': 0.0,
    'od_type': 'Iter',
    'od_wait': 50
}

'''
Best hyperparameters found:
{'model__random_strength': 1, 'model__learning_rate': 0.01, 'model__l2_leaf_reg': 5, 'model__iterations': 100, 'model__depth': 4, 'model__border_count': 64, 'model__bagging_temperature': 1.0}
'''

# Call the pipeline (no need to manually calibrate or call extra functions)
pipeline_model, X_test, y_test, y_pred, y_prob = train_catboost_pipeline(
    dataframe=modeling_df,
    target_col='pt_conversion',
    feature_list=feature_list,
    best_params=best_params,  #None, 
    threshold=0.5,    #0.3
    do_search=False,
    target_month=target_month,
    calibrate=True  ## ← this triggers Platt/Isotonic scaling inside the function
)


## After training and splitting:
calibrated_pipeline = calibrate_model(pipeline_model, X_test, y_test)
y_calibrated_prob = calibrated_pipeline.predict_proba(X_test)[:, 1]

## Lift-based evaluation
evaluate_lift_and_recall(y_test, y_calibrated_prob)

### view thresholds
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_thresholds(y_true, y_scores, thresholds=np.arange(0.1, 1.0, 0.05)):
    print("Threshold | Precision | Recall | F1 Score")
    print("---------------------------------------------")
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        #print(y_pred.sum())
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"{thresh:9.2f} | {precision:9.2f} | {recall:6.2f} | {f1:8.2f}")

evaluate_thresholds(y_test, y_prob)


# --------------------------
# Custom Lift Scorer
# --------------------------

def lift_at_k(y_true, y_scores, top_k=0.10):
    k = int(len(y_scores) * top_k)
    top_k_indices = np.argsort(y_scores)[-k:]
    lift = y_true[top_k_indices].mean() / y_true.mean()
    return lift

def make_lift_scorer(top_k=0.10):
    def scorer(estimator, X, y):
        prob = estimator.predict_proba(X)[:, 1]
        return lift_at_k(y, prob, top_k=top_k)
    return make_scorer(scorer, greater_is_better=True)

# --------------------------
# Feature Separation by Class Means
# --------------------------

def get_top_separating_features(df, target_col='pt_conversion', features=None, top_n=10, plot=False):
    if features is None:
        features = df.select_dtypes(include='number').drop(columns=[target_col]).columns.tolist()

    class_means = df.groupby(target_col)[features].mean().T
    class_means['abs_diff'] = abs(class_means[1] - class_means[0])
    ranked = class_means.sort_values('abs_diff', ascending=False)

    if plot:
        for feature in ranked.index[:top_n]:
            plt.figure(figsize=(6, 3))
            sns.histplot(data=df, x=feature, hue=target_col, bins=30, kde=False, stat='density', common_norm=False)
            plt.title(f"Feature: {feature}")
            plt.tight_layout()
            plt.show()

    return ranked.index[:top_n].tolist()

# --------------------------
# CatBoost Pipeline
# --------------------------

def train_catboost_pipeline(
    dataframe,
    target_col='pt_conversion',
    feature_list=None,
    best_params=None,
    do_search=False,
    test_size=0.2,
    random_state=56,
    threshold=0.3,
    target_month=None,
    calibrate=False,
    calibration_method='isotonic',
    use_ranking=True,
    top_k_fraction=0.10,
    extra_features_func=None
):
    class ThresholdedPipeline(Pipeline):
        def __init__(self, steps, threshold=0.5):
            super().__init__(steps)
            self.threshold = threshold

        def predict(self, X):
            probs = self.predict_proba(X)[:, 1]
            return (probs >= self.threshold).astype(int)

    df = dataframe.copy()
    df['summary_month_key'] = pd.to_datetime(df['summary_month_key'])
    if extra_features_func:
        df = extra_features_func(df)

    target_dt = pd.to_datetime(target_month).replace(day=1)
    lookback_start = target_dt - pd.DateOffset(months=6)
    lookback_end = target_dt - pd.DateOffset(days=1)

    print(f"\n[INFO] Training on full 6-month window: {lookback_start.date()} to {lookback_end.date()}")
    print(f"[INFO] Testing on target month: {target_dt.date()}")

    train_df = df[(df['summary_month_key'] >= lookback_start) & (df['summary_month_key'] <= lookback_end)].copy()
    test_df = df[df['summary_month_key'].dt.to_period('M') == target_dt.to_period('M')].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("[ERROR] Train or test data is empty.")

    if feature_list is None:
        feature_list = [
            col for col in train_df.columns
            if col not in [target_col, 'member_id', 'summary_month_key']
            and not np.issubdtype(train_df[col].dtype, np.datetime64)
        ]

    datetime_cols = train_df.select_dtypes(include='datetime64[ns]').columns.tolist()
    feature_list = [col for col in feature_list if col not in datetime_cols]

    X_train, y_train = train_df[feature_list], train_df[target_col]
    X_test, y_test = test_df[feature_list], test_df[target_col]

    for X in [X_train, X_test]:
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)

    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    print(f"[INFO] Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")
    print(f"pos_weight: {pos_weight:.2f}")

    preprocessor = Pipeline([
        ('convert_numeric', FunctionTransformer(
            lambda df: df.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtype == 'object' else col),
            validate=False
        )),
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ])

    base_model = CatBoostClassifier(
        eval_metric='AUC',
        class_weights=[1, pos_weight],
        random_seed=random_state,
        verbose=1
    )

    if best_params:
        base_model.set_params(**best_params)

    pipeline = ThresholdedPipeline([
        ('preprocessing', preprocessor),
        ('model', base_model)
    ], threshold=threshold)

    if do_search:
        param_grid = {
            'model__iterations': [300, 500, 1000],
            'model__depth': [4, 6, 8],
            'model__learning_rate': [0.01, 0.03, 0.1],
            'model__l2_leaf_reg': [3, 5, 7],
            'model__random_strength': [0.5, 1, 2],
            'model__bagging_temperature': [0.0, 0.5, 1.0],
            'model__border_count': [32, 64]
        }

        scorer = make_lift_scorer(top_k=top_k_fraction) if use_ranking else 'roc_auc'
        search = RandomizedSearchCV(
            pipeline, param_grid, n_iter=100,
            scoring=scorer, cv=3, verbose=2, n_jobs=-1,
            random_state=random_state
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        print("[INFO] Best Params:")
        print(search.best_params_)

    else:
        pipeline.fit(X_train, y_train)

    if calibrate:
        pipeline.named_steps['model'] = CalibratedClassifierCV(pipeline.named_steps['model'], cv=3, method=calibration_method)
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print(f"\nClassification Report @ Threshold = {threshold}")
    print(classification_report(y_test, y_pred))
    print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(); plt.plot(recall, precision); plt.title("PR Curve"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.grid(); plt.tight_layout(); plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], 'k--'); plt.title("ROC Curve"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.grid(); plt.tight_layout(); plt.show()

    model = pipeline.named_steps['model']
    if hasattr(model, 'feature_importances_'):
        fi_df = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_}).sort_values(by='importance', ascending=False)
        print("\nTop 25 Features:"); display(fi_df.head(25))

    top_k = int(len(y_test) * top_k_fraction)
    top_indices = np.argsort(y_prob)[-top_k:][::-1]
    lift = y_test.iloc[top_k_indices].mean() / y_test.mean()
    recall_at_top = y_test.iloc[top_k_indices].sum() / y_test.sum()
    print(f"\nLift@Top {int(top_k_fraction*100)}%: {lift:.2f}")
    print(f"Recall@Top {int(top_k_fraction*100)}%: {recall_at_top:.2%}")

    return pipeline, X_test, y_test, y_pred, y_prob, (search.best_params_ if do_search else best_params)


best_params = {
    'model__random_strength': 0.5,
    'model__learning_rate': 0.01,
    'model__l2_leaf_reg': 5,
    'model__iterations': 1000,
    'model__depth': 4,
    'model__border_count': 32,
    'model__bagging_temperature': 0.5
}


pipeline, X_test, y_test, y_pred, y_prob, best_params = train_catboost_pipeline(
    dataframe=modeling_df,
    target_col='pt_conversion',
    feature_list=[col for col in feature_list if col != 'member_since_date'],
    threshold=0.35,    
    calibrate=True,
    calibration_method='isotonic',
    use_ranking=True,
    do_search=False,
    best_params=best_params, 
    target_month=target_month,
    extra_features_func=add_engineered_features
)

print(best_params)


def evaluate_model_diagnostics(y_test, y_prob, threshold=0.5, top_k_percents=[0.01, 0.05, 0.10]):
    y_test = pd.Series(y_test).reset_index(drop=True)
    y_prob = pd.Series(y_prob).reset_index(drop=True)
    
    ## Basic info
    print(f"Total test samples: {len(y_test):,}")
    print(f"Total positives: {y_test.sum():,} ({y_test.mean():.2%})")

    ## Classification report
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\nClassification Report @ Threshold = {threshold}")
    print(classification_report(y_test, y_pred))
    print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    ## Threshold sweep
    print("\nThreshold | Precision | Recall | F1 Score")
    print("---------------------------------------------")
    for t in np.arange(0.1, 1.0, 0.05):
        yp = (y_prob >= t).astype(int)
        prec = precision_score(y_test, yp, zero_division=0)
        rec = recall_score(y_test, yp, zero_division=0)
        f1 = f1_score(y_test, yp, zero_division=0)
        print(f"{t:9.2f} | {prec:9.2f} | {rec:6.2f} | {f1:8.2f}")

    ## Lift table
    print("\nTop % | Users | Positives | Recall | Lift")
    print("------|--------|-----------|--------|------")
    for top_k in top_k_percents:
        k = int(len(y_test) * top_k)
        top_idx = y_prob.sort_values(ascending=False).head(k).index
        positives = y_test.loc[top_idx].sum()
        recall = positives / y_test.sum()
        lift = (positives / k) / y_test.mean()
        print(f"{int(top_k*100):>5}% | {k:6,} | {positives:9,} | {recall:6.2%} | {lift:5.2f}")

    ## Score distribution plot
    plt.figure(figsize=(6, 4))
    plt.hist(y_prob[y_test == 0], bins=30, alpha=0.6, label='Negatives')
    plt.hist(y_prob[y_test == 1], bins=30, alpha=0.6, label='Positives')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold}')
    plt.title("Score Distribution")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

'''
    ## PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    ## ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, y_prob):.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    '''

from sklearn.metrics import precision_score, recall_score, f1_score


evaluate_model_diagnostics(y_test, y_prob, threshold=0.35)

def plot_feature_distributions(df, target_col='pt_conversion', features=None, top_n=5):
    """
    Plots histograms of top_n features split by target class.
    """
    if features is None:
        # Auto-detect top numeric features (excluding target)
        features = df.select_dtypes(include='number').drop(columns=[target_col]).columns.tolist()
    
    # rank by variance between classes
    class_means = df.groupby(target_col)[features].mean().T
    class_means['abs_diff'] = abs(class_means[1] - class_means[0])
    ranked_features = class_means.sort_values('abs_diff', ascending=False).index.tolist()
    selected = ranked_features[:top_n]

    # Plot
    for feature in selected:
        plt.figure(figsize=(6, 3))
        sns.histplot(data=df, x=feature, hue=target_col, kde=False, bins=30, palette='Set1', stat='density', common_norm=False)
        plt.title(f"Feature: {feature}")
        plt.tight_layout()
        plt.show()


## Plot top separating features
plot_feature_distributions(modeling_df, target_col='pt_conversion', top_n=10)

###prediction pipeline
def run_monthly_scoring_pipeline(
    df, 
    model, 
    feature_names, 
    month_str='2025-02-01', 
    show_lift=True, 
    bins=10,
    return_scored=False
):
    def predict_on_same_month(df, trained_model, feature_names, month):
        month = pd.to_datetime(month).replace(day=1)
        month_end = month + pd.offsets.MonthEnd(0)
        month_df = df[df['summary_month_key'] == month_end].copy()

        member_ids = month_df['member_id'].copy()
        month_df['gender'] = month_df['gender'].fillna('Unknown')

        # Drop any datetime columns
        datetime_cols = month_df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        if datetime_cols:
            print(f"[INFO] Dropping datetime columns for scoring: {datetime_cols}")
            month_df.drop(columns=datetime_cols, inplace=True)

        # === Ensure only training-time features are used ===
        available_cols = set(month_df.columns)
        used_features = [col for col in feature_names if col in available_cols]

        # Ensure missing columns are added as 0
        for col in feature_names:
            if col not in month_df.columns:
                month_df[col] = 0

        X = month_df[used_features].copy()
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)


        month_df['pt_conversion_score'] = trained_model.predict_proba(X)[:, 1]
        month_df['member_id'] = member_ids.values
        return month_df[['member_id', 'pt_conversion_score']]

    def merge_actuals(df, scored_df, month):
        month_end = pd.to_datetime(month).replace(day=1) + pd.offsets.MonthEnd(0)
        actuals = df[df['summary_month_key'] == month_end][['member_id', 'pt_conversion']]
        merged = scored_df.merge(actuals, on='member_id', how='left')
        merged['pt_conversion'] = merged['pt_conversion'].fillna(0).astype(int)
        return merged

    def get_decile_analysis(scored_df, score_col='pt_conversion_score', actual_col='pt_conversion', bins=10):
        scored_df = scored_df.sort_values(by=score_col, ascending=False).reset_index(drop=True)

        try:
            scored_df['decile'] = pd.qcut(scored_df.index, q=bins, labels=False)
        except:
            scored_df['decile'] = pd.cut(scored_df.index, bins=bins, labels=False)

        grouped = scored_df.groupby('decile')[actual_col].agg(['sum', 'count']).reset_index()
        grouped['non_converters'] = grouped['count'] - grouped['sum']
        grouped['conversion_rate'] = grouped['sum'] / grouped['count']
        overall_rate = scored_df[actual_col].mean()
        grouped['lift'] = grouped['conversion_rate'] / overall_rate
        grouped['cumulative_converters'] = grouped['sum'].cumsum()
        grouped['cumulative_pct'] = grouped['cumulative_converters'] / scored_df[actual_col].sum()
        grouped['decile_size_pct'] = grouped['count'] / scored_df.shape[0]

        return scored_df, grouped.sort_values(by='decile', ascending=False).reset_index(drop=True)

    def plot_lift_chart(lift_df, month_str='2025-02-01'):
        lift_df = lift_df.sort_values(by='decile')  # Decile 0 = highest score on left

        plt.figure(figsize=(10, 6))
        plt.bar(lift_df['decile'], lift_df['lift'], color='skyblue', edgecolor='black')
        plt.axhline(1, color='gray', linestyle='--')
        plt.xlabel('Decile (0 = Highest Score)', fontsize=12)
        plt.ylabel('Lift vs Average', fontsize=12)
        plt.title(f'Lift Chart: {month_str}', fontsize=14)
        plt.xticks(lift_df['decile'])
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def print_converter_stats(scored_df):
        actual_converters = scored_df[scored_df['pt_conversion'] == 1]
        print("\n===== Converter Score Summary =====")
        print(f"Total actual converters: {len(actual_converters)}")
        print(f"Avg predicted score (converters): {actual_converters['pt_conversion_score'].mean():.3f}")
        print(f"Score range: {actual_converters['pt_conversion_score'].min():.3f} – {actual_converters['pt_conversion_score'].max():.3f}")

        top10 = scored_df.sort_values(by='pt_conversion_score', ascending=False).head(int(len(scored_df) * 0.10))
        top25 = scored_df.sort_values(by='pt_conversion_score', ascending=False).head(int(len(scored_df) * 0.25))
        total = scored_df['pt_conversion'].sum()

        print(f"\nTop 10% captured: {top10['pt_conversion'].sum()}/{total} ({top10['pt_conversion'].sum() / total:.2%})")
        print(f"Top 25% captured: {top25['pt_conversion'].sum()}/{total} ({top25['pt_conversion'].sum() / total:.2%})")     

    # === RUN PIPELINE === #
    print("\n===== Monthly Scoring =====")
    print(f"Target month: {month_str}")
    print(f"Model type: {type(model).__name__}")

    scored = predict_on_same_month(df, model, feature_names, month_str)
    merged = merge_actuals(df, scored, month_str)

    print_converter_stats(merged)
    scored_with_deciles, lift_table = get_decile_analysis(merged, bins=bins)

    if show_lift:
        plot_lift_chart(lift_table, month_str=month_str)

    return (scored_with_deciles, lift_table) if return_scored else lift_table

def plot_converters_with_percentages(lift_table, month_str='2025-02-01'):
    ## Sort by decile (low = top scoring)
    lift_table = lift_table.sort_values(by='decile')
    deciles = lift_table['decile'].astype(str)
    converters = lift_table['sum']
    total = lift_table['count']
    conv_pct = (converters / total * 100).round(1)

    ## Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(deciles, converters, color='royalblue', edgecolor='black')

    ## Set y-limit with padding to avoid cutting off top labels
    y_max = converters.max() * 1.15
    plt.ylim(0, y_max)

    for idx, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            label = f"{int(height):,} ({conv_pct.iloc[idx]}%)"
            plt.text(bar.get_x() + bar.get_width() / 2, height + (y_max * 0.01), 
                    label, ha='center', va='bottom', fontsize=9)   


scored_df, decile_table = run_monthly_scoring_pipeline(
    df=all_users_df,
    model=pipeline_model,
    feature_names=feature_list,
    month_str=target_month,
    show_lift=True,
    bins=10,
    return_scored=True
)

# Plot raw converters
plot_converters_with_percentages(decile_table, month_str=target_month)

# Display decile table
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2%}'.format)
display(decile_table)

##sanity check
all_users_df[
    (all_users_df['pt_conversion'] == 1) &
    (all_users_df['summary_month_key'] == '2025-02-28')
].shape[0]    ###double check for the exact numbers of converters by month


## Focus on high-confidence predictions only
threshold = 0.75
filtered = scored_df[scored_df['pt_conversion_score'] > threshold]

converted_scores = filtered[filtered['pt_conversion'] == 1]['pt_conversion_score']
non_converted_scores = filtered[filtered['pt_conversion'] == 0]['pt_conversion_score']

plt.figure(figsize=(10, 6))
plt.hist(non_converted_scores, bins=16, alpha=0.6, label='Did Not Convert', color='orange', edgecolor='black')
plt.hist(converted_scores, bins=16, alpha=0.7, label='Converted', color='steelblue', edgecolor='black')

plt.xlabel('PT Conversion Score (> 0.75 only)')
plt.ylabel('Member Count')
plt.title('High-Confidence Score Distribution: Converted vs Not')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nTotal members with score > {threshold}: {len(filtered)}")
print(f"Actual converters in this group: {filtered['pt_conversion'].sum()}")
print(f"Conversion rate in high-confidence group: {filtered['pt_conversion'].mean():.2%}")


precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision[:-1], label='Precision', color='blue')
plt.plot(thresholds, recall[:-1], label='Recall', color='orange')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs Threshold')
plt.axhline(y=0.5, color='gray', linestyle='--')
plt.legend()
plt.grid(True)
plt.show()
