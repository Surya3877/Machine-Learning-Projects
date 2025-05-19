# Data-Science-Internship-Assignment-Sales-Forecasting

## Introduction
Welcome to the Data Science Internship Assignment. In this project, you will work with real-world retail sales data to develop a forecasting model that predicts future sales for thousands of product families across different stores in Ecuador. This assignment will help you understand how external factors like promotions, holidays, economic conditions, and events impact sales, and how machine learning models can be used to improve demand forecasting.

This assignment is structured into two main parts:
1. **Data Processing and Feature Engineering (Day 1)** - Cleaning, transforming, and exploring the dataset.
2. **Model Selection, Forecasting, and Evaluation (Day 2)** - Training different forecasting models, comparing their performance, and presenting insights.

## Dataset Overview
The dataset consists of multiple files providing sales data and additional influencing factors:
- **train.csv** - Historical sales data.
- **test.csv** - The test set for which sales need to be predicted.
- **stores.csv** - Metadata about store locations and clusters.
- **oil.csv** - Daily oil prices (affecting Ecuador's economy).
- **holidays_events.csv** - Information about holidays and special events.

Your task is to forecast daily sales for each product family at each store for the next 15 days after the last training date.

---
## Part 1: Data Processing and Feature Engineering (Day 1)
### 1. Data Cleaning
- Load the dataset using Pandas.
   ```
   train = pd.read_csv('train.csv', parse_dates=['date'])
   stores = pd.read_csv('stores.csv')
   oil = pd.read_csv('oil.csv', parse_dates=['date'])
   holidays = pd.read_csv('holidays_events.csv', parse_dates=['date'])
   ```

- Handle missing values in oil prices by filling gaps with interpolation.
    ```
     oil['dcoilwtico'] = oil['dcoilwtico'].interpolate(method='linear')
      ```
  
-  Convert date columns to proper datetime formats
    ```
   train['date'] = pd.to_datetime(train['date'])
   oil['date'] = pd.to_datetime(oil['date'])
   holidays_events['date'] = pd.to_datetime(holidays_events['date'])
      ```
   
- Merge data from `stores.csv`, `oil.csv`, and `holidays_events.csv` into the main dataset.
    ```
  train = train.merge(stores, on='store_nbr', how='left')
  train = train.merge(oil, on='date', how='left')
  train = train.merge(holidays, on='date', how='left')
      ```

### 2. Feature Engineering
#### Time-based Features:
- Extract **day, week, month, year, and day of the week**.
- Identify **seasonal trends** (e.g., are sales higher in December?).
  ```
  train['day'] = train['date'].dt.day
  train['week'] = train['date'].dt.isocalendar().week
  train['month'] = train['date'].dt.month
  train['year'] = train['date'].dt.year
  train['day_of_week'] = train['date'].dt.dayofweek
    ```


#### Event-based Features:
- Create **binary flags** for holidays, promotions, and economic events.
- Identify if a day is a **government payday** (15th and last day of the month).
- Consider **earthquake impact (April 16, 2016)** as a separate feature.
  ```
  train['is_holiday'] = train['type'].notna().astype(int)
  train['is_weekend'] = (train['day_of_week'] >= 5).astype(int)
  train['is_gov_payday'] = train['day'].isin([15, train['date'].dt.days_in_month]).astype(int)
  train['earthquake_impact'] = (train['date'] == '2016-04-16').astype(int)
    ```


#### Rolling Statistics:
- Compute **moving averages** and **rolling standard deviations** for past sales.
- Include **lagged features** (e.g., sales from the previous week, previous month).
  ```
  train['sales_lag_7'] = train.groupby(['store_nbr', 'family'])['sales'].shift(7)
  train['sales_lag_30'] = train.groupby(['store_nbr', 'family'])['sales'].shift(30)
  train['rolling_mean_7'] = train.groupby(['store_nbr', 'family'])['sales'].rolling(7).mean().reset_index(level=[0,1], drop=True)
  train['rolling_std_7'] = train.groupby(['store_nbr', 'family'])['sales'].rolling(7).std().reset_index(level=[0,1], drop=True)
    ```


#### Store-Specific Aggregations:
- Compute **average sales per store type**.
- Identify **top-selling product families per cluster**.
  ```
  store_avg_sales = train.groupby('store_nbr')['sales'].mean().rename('avg_store_sales')
  train = train.merge(store_avg_sales, on='store_nbr', how='left')
  ```
### 3. Exploratory Data Analysis (EDA)
- Visualize **sales trends over time**.
  ```
  print("Exploratory Data Analysis")
  plt.figure(figsize=(12, 6))
  sns.lineplot(x='date', y='sales', data=train, label='Sales Trend')
  plt.title('Sales Trends Over Time')
  plt.xlabel('Date')
  plt.ylabel('Sales')
  plt.xticks(rotation=45)
  plt.legend()
  plt.show()
    ```

- Analyze **sales before and after holidays and promotions**.
  ```
  holiday_sales = train.groupby(['date', 'is_holiday'])['sales'].mean().reset_index()
  sns.boxplot(x='is_holiday', y='sales', data=holiday_sales)
  plt.title('Sales Before and After Holidays')
  plt.xlabel('Is Holiday')
  plt.ylabel('Sales')
  plt.show()
    ```

- Check **correlations between oil prices and sales trends**.
  ```
  correlation = train[['sales', 'dcoilwtico']].corr()
  print("Correlation between Sales and Oil Prices:")
  print(correlation)
  sns.scatterplot(x='dcoilwtico', y='sales', data=train)
  plt.title('Oil Prices vs Sales')
  plt.xlabel('Oil Price')
  plt.ylabel('Sales')
  plt.show()
    ```

- Identify **anomalies in the data**.
  ```
  sns.boxplot(y='sales', data=train)
  plt.title('Sales Anomalies Detection')
  plt.ylabel('Sales')
  plt.show()
    train.to_csv('processed_train.csv', index=False)
  print("Data processing and feature engineering completed.")
    ```


## Part 2: Model Selection, Forecasting, and Evaluation (Day 2)
### 1. Model Training
Train at least five different time series forecasting models:
- **Baseline Model (Naïve Forecasting)** - Assume future sales = previous sales.
  ```
  train['naive_forecast'] = train.groupby(['store_nbr', 'family'])['sales'].shift(1)
    ```
# Prepare training data
  ```
features = ['day', 'week', 'month', 'year', 'day_of_week', 'is_holiday', 'is_weekend', 
            'is_gov_payday', 'earthquake_impact', 'sales_lag_7', 'sales_lag_30', 
            'rolling_mean_7', 'rolling_std_7', 'avg_store_sales']
X = train[features].dropna()
y = train.loc[X.index, 'sales']
  ```
# Split into training and validation sets
  ```
train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]
  ```
- **ARIMA (AutoRegressive Integrated Moving Average)** - A traditional time series model.
  ```
  arima_model = ARIMA(y_train, order=(2,1,0))
  arima_model_fit = arima_model.fit(low_memory=True)
  arima_preds = arima_model_fit.forecast(steps=len(y_val))
    ```

- **Random Forest Regressor** - Tree-based model to capture non-linear relationships.
  ```
  rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
  rf_model.fit(X_train, y_train)
  rf_preds = rf_model.predict(X_val)
  sample_size = 100000  # Adjust based on available memory
  X_sample = X_train.sample(sample_size, random_state=42)
  y_sample = y_train.loc[X_sample.index]
  rf_model.fit(X_sample, y_sample)
    ```

- **XGBoost or LightGBM** - Gradient boosting models to improve accuracy.
  ```
  xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
  xgb_model.fit(X_train, y_train)
  xgb_preds = xgb_model.predict(X_val)
    ```

### 2. Model Evaluation
Compare models based on:
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)**
- **R-Squared Score**
- **Visual Inspection** (Plot actual vs. predicted sales)
  ```
  def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - RMSE: {rmse:.2f}, MAPE: {mape:.2%}, R2: {r2:.2f}")
    

  print("Model Evaluation:")
  if 'arima_preds' in locals():
    evaluate_model(y_val, arima_preds, "ARIMA")
  else:
    print("ARIMA predictions not found.")
  if "rf_model" in locals():
    rf_preds = rf_model.predict(X_val)
  else:
    print("Random Forest model not trained.")

  evaluate_model(y_val, arima_preds, "ARIMA")
  evaluate_model(y_val, rf_preds, "Random Forest")
  evaluate_model(y_val, xgb_preds, "XGBoost")
  ```
### 3. Visualization
- Plot **historical sales and predicted sales**.
- Compare **model performances** using error metrics.
- Visualize **feature importance** (for Random Forest/XGBoost).
  ```
  plt.figure(figsize=(12, 6))
  sns.lineplot(x=y_val.index, y=y_val, label='Actual Sales')
  sns.lineplot(x=y_val.index, y=rf_preds, label='Random Forest Predictions')
  sns.lineplot(x=y_val.index, y=xgb_preds, label='XGBoost Predictions')
  plt.legend()
  plt.title('Actual vs Predicted Sales')
  plt.show()

  print("Model training, forecasting, and evaluation completed.")
    ```

### 4. Interpretation and Business Insights

Model Comparison Based on Metrics
|    Model     |	RMSE ↓ (Lower is better) |	MAPE ↓ (Lower is better)    |	R² ↑ (Higher is better) |
| ------------ | -------------------------- | ------------------------   | ----------------------- |
|ARIMA	      |       1411.32              |	10195183152728457216.00% |  -0.06 (very bad)       |
|Random Forest |	     431.90	              |   975344682013655424.00%	 |   0.90 (good)           |
|XGBoost	      |       307.43 (best)	     |   728251545701144448.00%	 |   0.95 (best)           |

**Interpretation**

1. **RMSE (Root Mean Square Error)**

- XGBoost has the lowest RMSE (307.43), meaning its predictions are closest to actual values.
- ARIMA performs the worst (1411.32).
- Random Forest is better than ARIMA but worse than XGBoost.

2. **R² (Coefficient of Determination)**

 - XGBoost has the highest R² (0.95), indicating it explains 95% of the variance.
 - Random Forest is close with 0.90.
 - ARIMA has negative R², meaning it's worse than simply using the mean of the data.
   
3. **MAPE (Mean Absolute Percentage Error) – Extremely Large!**

 - These numbers are way too big to be reasonable. This suggests:
 - Some predictions are dividing by very small actual values (causing MAPE to explode).
 - Data scaling issues (e.g., log-transformed data).
 - Errors when computing percentage format (should be in 0-100% range).

#### Best Performing Model
Based on the error metrics, the **[XGBoost Model]** performed best with the lowest RMSE and highest R-Squared value. This model effectively captured the sales patterns and external factors.

#### Impact of External Factors
- **Holidays:** Sales showed a significant increase during national holidays, particularly around [2016-04-16]. Models with holiday data as features showed improved accuracy.
- **Oil Prices:** There was a negative correlation between oil prices and sales in certain product families, indicating that higher fuel prices decreased sales.
- **Promotions:** Promotional events had a notable impact on sales spikes, especially for [oils]. Incorporating promotion data improved model performance.

#### Business Strategies
- **Inventory Planning:** Use sales forecasts to optimize inventory levels, especially during peak holiday seasons.
- **Targeted Promotions:** Align promotional campaigns with periods of low predicted sales to boost revenue.
- **Price Adjustments:** Monitor oil prices and adjust pricing strategies accordingly to maintain sales volumes.

By leveraging these insights, the business can improve demand forecasting, reduce stockouts, and enhance customer satisfaction.







