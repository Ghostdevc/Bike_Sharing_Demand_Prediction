import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data_and_utils.utils import load_data_daily, load_data_hourly, check_df, grab_col_names, target_summary_with_cat, target_summary_with_cat, high_correlated_cols, correlation_matrix, check_outlier, remove_outlier, outlier_thresholds, replace_with_thresholds
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
df_daily = load_data_daily()
df_hourly = load_data_hourly()

#### Dataset characteristics
	
#Both hour.csv and day.csv have the following fields, except hr which is not available in day.csv
	
	#- instant: record index
	#- dteday : date
	#- season : season (1:springer, 2:summer, 3:fall, 4:winter)
	#- yr : year (0: 2011, 1:2012)
	#- mnth : month ( 1 to 12)
	#- hr : hour (0 to 23)
	#- holiday : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
	#- weekday : day of the week
	#- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
	#+ weathersit : 
		#- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
		#- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
		#- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
		#- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
	#- temp : Normalized temperature in Celsius. The values are divided to 41 (max)
	#- atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
	#- hum: Normalized humidity. The values are divided to 100 (max)
	#- windspeed: Normalized wind speed. The values are divided to 67 (max)
	#- casual: count of casual users
	#- registered: count of registered users
	#- cnt: count of total rental bikes including both casual and registered
# EDA
check_df(df_daily)
check_df(df_hourly)
## Visualizations

    #- In order to be able to visualize the columns we first need to capture cat and num feature names.
    #- Utilizing grab_col_names() function with certain parameters will yield us the column names accordingly. These parameters are chosen based on considerations over unique values of the features.
cat_cols, num_cols, cat_but_car = grab_col_names(df_hourly, cat_th=25, car_th=20)
print(num_cols)
print(cat_cols)
print(cat_but_car)
### Feature Visualizations
#### Categorical Features
for col in cat_cols:
    ax = sns.catplot(kind='count', data=df_hourly, x=col, hue=col, height=6, aspect=2)
#**OBSERVATIONS**
#- Time based features are uniformly distributed as expected.
#- Class imbalance is observed in weathersit, workingday and holiday features. 
#### Numerical Features
sns.pairplot(df_hourly[num_cols])
g = sns.relplot(
    data=df_hourly, 
    x="dteday", 
    y="cnt", 
    kind="line", 
    height=6, 
    aspect=4,
    errorbar=None
)

g.figure.suptitle("Daily Rent Trend (2011-2012)", y=1.03)
g.set_axis_labels("Date", "Total Rent Count (cnt)")

plt.xticks(rotation=45)

plt.show()
#**OBSERVATIONS**
#- non-stagnant
#- Seasonal
#- Trendy
#- Rent count increases during spring and summer
for col in num_cols:
    g = sns.displot(kind='kde', data=df_hourly, x=col, fill=True, height=6, aspect=2)
    g.figure.suptitle(col + " " + "KDE", y=1.03)
    plt.xticks(rotation=45)
    plt.show()
#### Target Guided Feature Visualizations
for col in cat_cols:
    sns.displot(kind='kde', data=df_hourly, x='cnt', hue=col, fill=True, palette='tab10', height=6, aspect=4)
    plt.show()
    #sns.kdeplot(data=df_hourly, x='cnt', hue=col)
    target_summary_with_cat(df_hourly, 'cnt', col)
for catcol in cat_cols:
    target_nums = [col for col in num_cols if col not in ['cnt']]
    
    for numcol in target_nums:
        g = sns.jointplot(
            data=df_hourly,
            x=numcol, 
            y="cnt", 
            hue=catcol,
            kind="scatter",  
            alpha=0.6,       
            height=6,
            palette='tab10'         
        )
        
        g.figure.suptitle(f"Analysis: {numcol} vs cnt (Category: {catcol})", y=1.02)

        plt.xticks(rotation=45)
        
        plt.show()

        target_summary_with_cat(df_hourly, 'cnt', catcol)
#### Feature Correlations
#correlation_matrix(df_hourly, [col for col in df_hourly.columns if col not in ['dteday']])
correlation_matrix(df_hourly, df_hourly.columns)
high_correlated_cols(df_hourly, True, 0.40)
### Outlier Detection
# Firstly transforming all features to raw forms to normalize as a whole later on
	#- temp : Normalized temperature in Celsius. The values are divided to 41 (max)
	#- atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
	#- hum: Normalized humidity. The values are divided to 100 (max)
	#- windspeed: Normalized wind speed. The values are divided to 67 (max)

df_hourly['temp'] = df_hourly['temp'] * 41
df_hourly['atemp'] = df_hourly['atemp'] * 50
df_hourly['hum'] = df_hourly['hum'] * 100
df_hourly['windspeed'] = df_hourly['windspeed'] * 67
for col in [col for col in num_cols if col not in ['dteday']]:
    print(col + ' has outliers: ', check_outlier(df_hourly, col))
for col in [col for col in num_cols if col not in ['dteday', 'cnt']]:
    replace_with_thresholds(df_hourly, col)
for col in [col for col in num_cols if col not in ['dteday']]:
    print(col + ' has outliers: ', check_outlier(df_hourly, col))
df_hourly['cnt'] = df_hourly['registered'] + df_hourly['casual']
for col in [col for col in num_cols if col not in ['dteday']]:
    print(col + ' has outliers: ', check_outlier(df_hourly, col))
for col in num_cols:
    g = sns.displot(kind='kde', data=df_hourly, x=col, fill=True, height=6, aspect=2)
    g.figure.suptitle(col + " " + "KDE", y=1.03)
    plt.xticks(rotation=45)
    plt.show()
### Feature Engineering
# Discretization of hr column
hour_target_mean = df_hourly.groupby('hr')['cnt'].mean().sort_values()
bins = pd.qcut(hour_target_mean.values, q=4, labels=False, duplicates='drop')
bin_mapping = dict(zip(hour_target_mean.index, bins))
#bin_labels = {
    #0: 'low_activity_hours',
    #1: 'medium_activity_hours',
    #2: 'high_activity_hours',
    #3: 'very_high_activity_hours'
#}
    
df_hourly['NEW_hourly_activity_level'] = df_hourly['hr'].map(bin_mapping)
# month
month_target_mean = df_hourly.groupby('mnth')['cnt'].mean().sort_values()
bins_mnth = pd.qcut(month_target_mean.values, q=4, labels=False, duplicates='drop')
bin_mapping_mnth = dict(zip(month_target_mean.index, bins_mnth))
#bin_labels_mnth = {
    #0: 'low_activity_months',
    #1: 'medium_activity_months',
    #2: 'high_activity_months',
    #3: 'very_high_activity_months'
#}
df_hourly['NEW_monthly_activity_level'] = df_hourly['mnth'].map(bin_mapping_mnth)
# season
season_target_mean = df_hourly.groupby('season')['cnt'].mean()

# Rank 1: Spring - Average CNT: 111.12 - LOWEST ACTIVITY
# Rank 2: Winter - Average CNT: 198.87 - MEDIUM-LOW ACTIVITY
# Rank 3: Summer - Average CNT: 208.34 - MEDIUM-HIGH ACTIVITY
# Rank 4: Fall - Average CNT: 236.02 - HIGHEST ACTIVITY

# assign new values based on rank (rank based)
# rank(method='dense') kullanarak: en düşük mean = 1, en yüksek mean = 4
season_rank_mapping = season_target_mean.rank(method='dense').astype(int).to_dict()
df_hourly['NEW_season_ranked'] = df_hourly['season'].map(season_rank_mapping)
# weathersit
weathersit_reverse_mapping = {
    4: 1,  # Heavy Rain/Snow -> Rank 1 (worst weather, lowest cnt: 74.33)
    3: 2,  # Light Snow/Rain -> Rank 2 (bad weather, low cnt: 111.58)
    2: 3,  # Mist + Cloudy -> Rank 3 (moderate weather, medium cnt: 175.17)
    1: 4   # Clear -> Rank 4 (best weather, highest cnt: 204.87)
}

df_hourly['NEW_weathersit_ranked'] = df_hourly['weathersit'].map(weathersit_reverse_mapping)
# temp
# will be discretized into 2 bins since 2 density regions are visible in the distribution chart
df_hourly['temp_binned'] = pd.qcut(df_hourly['temp'], 
                                    q=2, 
                                    labels=['low_temp', 'high_temp'],
                                    duplicates='drop')
# atemp
# will be discretized into 3 bins since 3 density regions are visible in the distribution chart
df_hourly['atemp_binned'] = pd.qcut(df_hourly['atemp'], 
                                     q=3, 
                                     labels=['low_atemp', 'medium_atemp', 'high_atemp'],
                                     duplicates='drop')
# windspeed
# will be discretized into 4 bins since 4 density regions are visible in the distribution chart
df_hourly['windspeed_binned'] = pd.qcut(df_hourly['windspeed'], 
                                         q=4, 
                                         labels=['very_low_wind', 'low_wind', 
                                                'medium_wind', 'high_wind'],
                                         duplicates='drop')
discretized_cols = ['temp_binned', 'atemp_binned', 'windspeed_binned']
# target means of discretized_cols
for col in discretized_cols:
    target_summary_with_cat(df_hourly, 'cnt', col)
temp_rank_mapping = {
    'low_temp': 1,   # Lowest CNT
    'high_temp': 2   # Highest CNT
}

df_hourly['NEW_temp_ranked'] = df_hourly['temp_binned'].map(temp_rank_mapping).astype(int)

atemp_rank_mapping = {
    'low_atemp': 1,      # Lowest CNT
    'medium_atemp': 2,   # Medium CNT
    'high_atemp': 3      # Highest CNT
}

df_hourly['NEW_atemp_ranked'] = df_hourly['atemp_binned'].map(atemp_rank_mapping).astype(int)


windspeed_target_means = {
    'very_low_wind': 152.114,
    'low_wind': 182.128,
    'medium_wind': 200.412,
    'high_wind': 197.390
}

sorted_windspeed = sorted(windspeed_target_means.items(), key=lambda x: x[1])
windspeed_rank_mapping = {name: idx+1 for idx, (name, _) in enumerate(sorted_windspeed)}


windspeed_rank_mapping = {
    'very_low_wind': 1,  # Lowest CNT (152.114)
    'low_wind': 2,       # Medium-low CNT (182.128)
    'high_wind': 3,      # Medium-high CNT (197.390)
    'medium_wind': 4     # Highest CNT (200.412)
}

df_hourly['NEW_windspeed_ranked'] = df_hourly['windspeed_binned'].map(windspeed_rank_mapping).astype(int)
check_df(df_hourly)
import scipy.stats as stats

y1 = df_hourly['cnt']
plt.figure(2); plt.title('Normal')
sns.distplot(y1, kde=False, fit=stats.norm)
plt.show()
plt.figure(3); plt.title('Log Normal')
sns.distplot(y1, kde=False, fit=stats.lognorm)
plt.show()
#**OBSERVATION**
#- Distribution of target fits log normal rather than normal curve.
#- Transformation can be considered
# Model Construction
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
### Feature Selection
#- Our aim is to model the data linearly and spectating the linear model's (Ridge, Lasso) capability to select features.
#- Thus Feature Engineering section was focused on creating linearly definable features
#- Non-linear features and the features with less or no correlations with the target will be pre-eliminated.
#- 'Registered' and 'Casual' columns will be removed as they are components of the target variable
final_features = [col for col in df_hourly.columns if 'NEW' in col]
final_features.extend(['hum', 'workingday', 'yr', 'cnt'])
print(final_features)
check_df(df_hourly[final_features])
correlation_matrix(df_hourly, final_features)
#**OBSERVATIONS**
#- Compared to previous correlation matrix, a matrix packed with more related features can be observed.
X = df_hourly[final_features].drop(columns=['cnt'])
y = np.log(df_hourly['cnt'])
y_original = df_hourly['cnt']
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_poly, y, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.176, random_state=42
)

# Original (non-log) split
y_original_train_val, y_original_test = train_test_split(
    y_original, test_size=0.15, random_state=42
)
y_original_train, y_original_val = train_test_split(
    y_original_train_val, test_size=0.176, random_state=42
)

print(f"Training:   {X_train.shape[0]:>5} samples ({X_train.shape[0]/len(X)*100:>5.1f}%)")
print(f"Validation: {X_val.shape[0]:>5} samples ({X_val.shape[0]/len(X)*100:>5.1f}%)")
print(f"Test:       {X_test.shape[0]:>5} samples ({X_test.shape[0]/len(X)*100:>5.1f}%)")
# Data standardized (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
results = {}
#**OLS**
ols_model = LinearRegression()
ols_model.fit(X_train_scaled, y_train)

y_train_pred_ols = ols_model.predict(X_train_scaled)
y_val_pred_ols = ols_model.predict(X_val_scaled)
results['OLS'] = {
    'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_ols)),
    'train_mae': mean_absolute_error(y_train, y_train_pred_ols),
    'train_r2': r2_score(y_train, y_train_pred_ols),
    'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred_ols)),
    'val_mae': mean_absolute_error(y_val, y_val_pred_ols),
    'val_r2': r2_score(y_val, y_val_pred_ols),
    'model': ols_model
}
y_train_pred_ols_real = np.exp(y_train_pred_ols)
y_val_pred_ols_real = np.exp(y_val_pred_ols)
# Real-space metrics
results['OLS']['train_rmse_real'] = np.sqrt(mean_squared_error(y_original_train, y_train_pred_ols_real))
results['OLS']['train_mae_real'] = mean_absolute_error(y_original_train, y_train_pred_ols_real)
results['OLS']['train_r2_real'] = r2_score(y_original_train, y_train_pred_ols_real)
results['OLS']['val_rmse_real'] = np.sqrt(mean_squared_error(y_original_val, y_val_pred_ols_real))
results['OLS']['val_mae_real'] = mean_absolute_error(y_original_val, y_val_pred_ols_real)
results['OLS']['val_r2_real'] = r2_score(y_original_val, y_val_pred_ols_real)
print(f"LOG-SPACE:")
print(f"Training   - RMSE: {results['OLS']['train_rmse']:>7.4f}, MAE: {results['OLS']['train_mae']:>7.4f}, R²: {results['OLS']['train_r2']:>6.4f}")
print(f"Validation - RMSE: {results['OLS']['val_rmse']:>7.4f}, MAE: {results['OLS']['val_mae']:>7.4f}, R²: {results['OLS']['val_r2']:>6.4f}")
print(f"\nREAL-SPACE (gerçek cnt değerleri):")
print(f"Training   - RMSE: {results['OLS']['train_rmse_real']:>7.2f}, MAE: {results['OLS']['train_mae_real']:>7.2f}, R²: {results['OLS']['train_r2_real']:>6.4f}")
print(f"Validation - RMSE: {results['OLS']['val_rmse_real']:>7.2f}, MAE: {results['OLS']['val_mae_real']:>7.2f}, R²: {results['OLS']['val_r2_real']:>6.4f}")
#**RIDGE**
alphas_ridge = np.logspace(-3, 3, 50)
ridge_cv = GridSearchCV(Ridge(), {'alpha': alphas_ridge}, cv=5, 
                        scoring='neg_root_mean_squared_error', n_jobs=-1)
ridge_cv.fit(X_train_scaled, y_train)
best_ridge = ridge_cv.best_estimator_
print(f"Best alpha: {ridge_cv.best_params_['alpha']:.4f}")
y_train_pred_ridge = best_ridge.predict(X_train_scaled)
y_val_pred_ridge = best_ridge.predict(X_val_scaled)
# Log-space metrics
results['Ridge'] = {
    'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_ridge)),
    'train_mae': mean_absolute_error(y_train, y_train_pred_ridge),
    'train_r2': r2_score(y_train, y_train_pred_ridge),
    'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred_ridge)),
    'val_mae': mean_absolute_error(y_val, y_val_pred_ridge),
    'val_r2': r2_score(y_val, y_val_pred_ridge),
    'model': best_ridge,
    'best_alpha': ridge_cv.best_params_['alpha']
}
# Real-space metrics
y_train_pred_ridge_real = np.exp(y_train_pred_ridge)
y_val_pred_ridge_real = np.exp(y_val_pred_ridge)
results['Ridge']['train_rmse_real'] = np.sqrt(mean_squared_error(y_original_train, y_train_pred_ridge_real))
results['Ridge']['train_mae_real'] = mean_absolute_error(y_original_train, y_train_pred_ridge_real)
results['Ridge']['train_r2_real'] = r2_score(y_original_train, y_train_pred_ridge_real)
results['Ridge']['val_rmse_real'] = np.sqrt(mean_squared_error(y_original_val, y_val_pred_ridge_real))
results['Ridge']['val_mae_real'] = mean_absolute_error(y_original_val, y_val_pred_ridge_real)
results['Ridge']['val_r2_real'] = r2_score(y_original_val, y_val_pred_ridge_real)
print(f"LOG-SPACE:")
print(f"Training   - RMSE: {results['Ridge']['train_rmse']:>7.4f}, MAE: {results['Ridge']['train_mae']:>7.4f}, R²: {results['Ridge']['train_r2']:>6.4f}")
print(f"Validation - RMSE: {results['Ridge']['val_rmse']:>7.4f}, MAE: {results['Ridge']['val_mae']:>7.4f}, R²: {results['Ridge']['val_r2']:>6.4f}")
print(f"\nREAL-SPACE (gerçek cnt değerleri):")
print(f"Training   - RMSE: {results['Ridge']['train_rmse_real']:>7.2f}, MAE: {results['Ridge']['train_mae_real']:>7.2f}, R²: {results['Ridge']['train_r2_real']:>6.4f}")
print(f"Validation - RMSE: {results['Ridge']['val_rmse_real']:>7.2f}, MAE: {results['Ridge']['val_mae_real']:>7.2f}, R²: {results['Ridge']['val_r2_real']:>6.4f}")
#**LASSO**
alphas_lasso = np.logspace(-3, 2, 50)
lasso_cv = GridSearchCV(Lasso(max_iter=10000), {'alpha': alphas_lasso}, cv=5,
                        scoring='neg_root_mean_squared_error', n_jobs=-1)
lasso_cv.fit(X_train_scaled, y_train)
best_lasso = lasso_cv.best_estimator_
print(f"Best alpha: {lasso_cv.best_params_['alpha']:.4f}")
y_train_pred_lasso = best_lasso.predict(X_train_scaled)
y_val_pred_lasso = best_lasso.predict(X_val_scaled)
n_features_selected = np.sum(best_lasso.coef_ != 0)
# Log-space metrics
results['Lasso'] = {
    'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_lasso)),
    'train_mae': mean_absolute_error(y_train, y_train_pred_lasso),
    'train_r2': r2_score(y_train, y_train_pred_lasso),
    'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred_lasso)),
    'val_mae': mean_absolute_error(y_val, y_val_pred_lasso),
    'val_r2': r2_score(y_val, y_val_pred_lasso),
    'model': best_lasso,
    'best_alpha': lasso_cv.best_params_['alpha'],
    'n_features_selected': n_features_selected
}
# Real-space metrics
y_train_pred_lasso_real = np.exp(y_train_pred_lasso)
y_val_pred_lasso_real = np.exp(y_val_pred_lasso)
results['Lasso']['train_rmse_real'] = np.sqrt(mean_squared_error(y_original_train, y_train_pred_lasso_real))
results['Lasso']['train_mae_real'] = mean_absolute_error(y_original_train, y_train_pred_lasso_real)
results['Lasso']['train_r2_real'] = r2_score(y_original_train, y_train_pred_lasso_real)
results['Lasso']['val_rmse_real'] = np.sqrt(mean_squared_error(y_original_val, y_val_pred_lasso_real))
results['Lasso']['val_mae_real'] = mean_absolute_error(y_original_val, y_val_pred_lasso_real)
results['Lasso']['val_r2_real'] = r2_score(y_original_val, y_val_pred_lasso_real)
print(f"LOG-SPACE:")
print(f"Training   - RMSE: {results['Lasso']['train_rmse']:>7.4f}, MAE: {results['Lasso']['train_mae']:>7.4f}, R²: {results['Lasso']['train_r2']:>6.4f}")
print(f"Validation - RMSE: {results['Lasso']['val_rmse']:>7.4f}, MAE: {results['Lasso']['val_mae']:>7.4f}, R²: {results['Lasso']['val_r2']:>6.4f}")
print(f"Features selected: {n_features_selected}/{X_train_scaled.shape[1]} ({n_features_selected/X_train_scaled.shape[1]*100:.1f}%)")
print(f"\nREAL-SPACE (gerçek cnt değerleri):")
print(f"Training   - RMSE: {results['Lasso']['train_rmse_real']:>7.2f}, MAE: {results['Lasso']['train_mae_real']:>7.2f}, R²: {results['Lasso']['train_r2_real']:>6.4f}")
print(f"Validation - RMSE: {results['Lasso']['val_rmse_real']:>7.2f}, MAE: {results['Lasso']['val_mae_real']:>7.2f}, R²: {results['Lasso']['val_r2_real']:>6.4f}")
#**ELASTICNET**
elasticnet_cv = GridSearchCV(
    ElasticNet(max_iter=10000),
    {'alpha': np.logspace(-3, 2, 20), 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]},
    cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1
)
elasticnet_cv.fit(X_train_scaled, y_train)
best_elasticnet = elasticnet_cv.best_estimator_
print(f"Best alpha: {elasticnet_cv.best_params_['alpha']:.4f}")
print(f"Best l1_ratio: {elasticnet_cv.best_params_['l1_ratio']:.2f}")
y_train_pred_en = best_elasticnet.predict(X_train_scaled)
y_val_pred_en = best_elasticnet.predict(X_val_scaled)
n_features_selected_en = np.sum(best_elasticnet.coef_ != 0)
results['ElasticNet'] = {
    'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_en)),
    'train_mae': mean_absolute_error(y_train, y_train_pred_en),
    'train_r2': r2_score(y_train, y_train_pred_en),
    'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred_en)),
    'val_mae': mean_absolute_error(y_val, y_val_pred_en),
    'val_r2': r2_score(y_val, y_val_pred_en),
    'model': best_elasticnet,
    'best_alpha': elasticnet_cv.best_params_['alpha'],
    'best_l1_ratio': elasticnet_cv.best_params_['l1_ratio'],
    'n_features_selected': n_features_selected_en
}
# Real-space metrics
y_train_pred_en_real = np.exp(y_train_pred_en)
y_val_pred_en_real = np.exp(y_val_pred_en)
results['ElasticNet']['train_rmse_real'] = np.sqrt(mean_squared_error(y_original_train, y_train_pred_en_real))
results['ElasticNet']['train_mae_real'] = mean_absolute_error(y_original_train, y_train_pred_en_real)
results['ElasticNet']['train_r2_real'] = r2_score(y_original_train, y_train_pred_en_real)
results['ElasticNet']['val_rmse_real'] = np.sqrt(mean_squared_error(y_original_val, y_val_pred_en_real))
results['ElasticNet']['val_mae_real'] = mean_absolute_error(y_original_val, y_val_pred_en_real)
results['ElasticNet']['val_r2_real'] = r2_score(y_original_val, y_val_pred_en_real)
print(f"LOG-SPACE:")
print(f"Training   - RMSE: {results['ElasticNet']['train_rmse']:>7.4f}, MAE: {results['ElasticNet']['train_mae']:>7.4f}, R²: {results['ElasticNet']['train_r2']:>6.4f}")
print(f"Validation - RMSE: {results['ElasticNet']['val_rmse']:>7.4f}, MAE: {results['ElasticNet']['val_mae']:>7.4f}, R²: {results['ElasticNet']['val_r2']:>6.4f}")
print(f"Features selected: {n_features_selected_en}/{X_train_scaled.shape[1]} ({n_features_selected_en/X_train_scaled.shape[1]*100:.1f}%)")
print(f"\nREAL-SPACE (gerçek cnt değerleri):")
print(f"Training   - RMSE: {results['ElasticNet']['train_rmse_real']:>7.2f}, MAE: {results['ElasticNet']['train_mae_real']:>7.2f}, R²: {results['ElasticNet']['train_r2_real']:>6.4f}")
print(f"Validation - RMSE: {results['ElasticNet']['val_rmse_real']:>7.2f}, MAE: {results['ElasticNet']['val_mae_real']:>7.2f}, R²: {results['ElasticNet']['val_r2_real']:>6.4f}")
#**TEST SET EVALUATION**
test_results = {}
for model_name in ['OLS', 'Ridge', 'Lasso', 'ElasticNet']:
    model = results[model_name]['model']
    y_test_pred = model.predict(X_test_scaled)
    
    test_results[model_name] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred),
        'R2': r2_score(y_test, y_test_pred)
    }

    y_test_pred_real = np.exp(y_test_pred)
    test_results[model_name]['RMSE_real'] = np.sqrt(mean_squared_error(y_original_test, y_test_pred_real))
    test_results[model_name]['MAE_real'] = mean_absolute_error(y_original_test, y_test_pred_real)
    test_results[model_name]['R2_real'] = r2_score(y_original_test, y_test_pred_real)
comparison_df_log = pd.DataFrame({
    'Model': ['OLS', 'Ridge', 'Lasso', 'ElasticNet'],
    'Train_RMSE': [results[m]['train_rmse'] for m in ['OLS', 'Ridge', 'Lasso', 'ElasticNet']],
    'Val_RMSE': [results[m]['val_rmse'] for m in ['OLS', 'Ridge', 'Lasso', 'ElasticNet']],
    'Test_RMSE': [test_results[m]['RMSE'] for m in ['OLS', 'Ridge', 'Lasso', 'ElasticNet']],
    'Test_MAE': [test_results[m]['MAE'] for m in ['OLS', 'Ridge', 'Lasso', 'ElasticNet']],
    'Test_R2': [test_results[m]['R2'] for m in ['OLS', 'Ridge', 'Lasso', 'ElasticNet']]
})
print("\n" + comparison_df_log.to_string(index=False))
comparison_df_real = pd.DataFrame({
    'Model': ['OLS', 'Ridge', 'Lasso', 'ElasticNet'],
    'Train_RMSE': [results[m]['train_rmse_real'] for m in ['OLS', 'Ridge', 'Lasso', 'ElasticNet']],
    'Val_RMSE': [results[m]['val_rmse_real'] for m in ['OLS', 'Ridge', 'Lasso', 'ElasticNet']],
    'Test_RMSE': [test_results[m]['RMSE_real'] for m in ['OLS', 'Ridge', 'Lasso', 'ElasticNet']],
    'Test_MAE': [test_results[m]['MAE_real'] for m in ['OLS', 'Ridge', 'Lasso', 'ElasticNet']],
    'Test_R2': [test_results[m]['R2_real'] for m in ['OLS', 'Ridge', 'Lasso', 'ElasticNet']],
    'N_Features': [X_train_scaled.shape[1], X_train_scaled.shape[1],
                   results['Lasso']['n_features_selected'],
                   results['ElasticNet']['n_features_selected']]
})
print("\n" + comparison_df_real.to_string(index=False))
# Best Model (real-space RMSE)
best_model = min(test_results, key=lambda x: test_results[x]['RMSE_real'])
print(f"\n BEST MODEL (Real-space RMSE): {best_model}")
print(f"   Test RMSE: {test_results[best_model]['RMSE_real']:.2f} bikes")
print(f"   Test MAE:  {test_results[best_model]['MAE_real']:.2f} bikes")
print(f"   Test R²:   {test_results[best_model]['R2_real']:.4f}")
def visualize_predictions(y_true_real, y_pred_log, model_name="Model"):
    """
    Tahminleri görselleştir
    """
    import matplotlib.pyplot as plt
    
    y_pred_real = np.exp(y_pred_log)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Actual vs Predicted
    axes[0].scatter(y_true_real, y_pred_real, alpha=0.5)
    axes[0].plot([y_true_real.min(), y_true_real.max()], 
                 [y_true_real.min(), y_true_real.max()], 
                 'r--', linewidth=2)
    axes[0].set_xlabel('Actual cnt')
    axes[0].set_ylabel('Predicted cnt')
    axes[0].set_title(f'{model_name}: Actual vs Predicted')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_true_real - y_pred_real
    axes[1].scatter(y_pred_real, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted cnt')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'{model_name}: Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()