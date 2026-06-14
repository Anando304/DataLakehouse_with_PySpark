"""
Problem: PySpark ML Pipeline
Author: Anando Zaman
Date: June 13, 2026

Context:
---------------------
From the previous exercise and considering how sessions are defined, we would like to apply an
ML use case to forecast a metric. You may choose one of the following:
• Average duration of sessions
• Number of sessions

Data
• Same as previous exercise.

Task:
---------------------
Select the top 1 user who has the highest number of sessions. Forecast the next 3 months of
your selected metric, starting from the last available record for that user



SOLUTION:
------------
1. Load and Preprocess the data
2. Determine the user with the largest number of sessions
3. Use PROPHET or ARIMAX/SARIMAX algorithm and Pandas to forecast.
    - Prophet/ARIMAX/SARIMAX works great with time series data and seasonality change. No manual lags or vector assembly feature engineering required.
    - Trend + Seasonality/Holiday Effects

    Why use SARISMA and not Prophet?
        - Installing Prophet in the docker image takes a long time since it is a native C++ lib behind the scenes
        - Faster to install statsmodel library and use ARIMA or SARIMAX


    Why not use PySpark MLib or RandomForest?
        - Decision trees are not suited for extrapolation since it cannot predict a value higher or lower than its minima/maxima
        
        - PySpark's native MLlib library (VectorAssembler, LinearRegression, GBTRegressor, etc.) excels at traditional tabular machine learning (like regression and classification). It lacks native algorithmic support for classical time-series forecasting (eg; PROPHET)
        - Example of tabular data usecases: (Predicting house prices based on square footage, bedrooms, and location)
            - MLib assumes each data row as independent rather than having some time-dependent relation, which further complicates our prediction.
            - The number of sessions a user plays today is heavily dependent on how many sessions they played yesterday, the day before, and last week.
            - Extensive feature engineering would be required (get day/month, session cnt lags, etc).

"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import DataFrame, types as T # Import types as T
from pyspark.sql.types import StructType, StructField, StringType
from utils.metrics import get_top_user_by_session_count, get_daily_session_counts
from utils.utils import load_dataset, generate_session_info, preprocess_music_data
from utils.constants import SPARK_MASTER, MUSIC_ANALYTICS_PIPELINE, MUSIC_USER_TIMESTAMPS_DATASET_DIR, USER_PROFILES_DATASET_DIR, DATASET_OUTPUT_DIRECTORY
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

if __name__ == "__main__":
    # Initialize Spark session
    spark_session = SparkSession.builder.master(SPARK_MASTER).appName(MUSIC_ANALYTICS_PIPELINE).getOrCreate()

    music_user_timestamps_schema = StructType([
        StructField("userid", StringType(), True),
        StructField("timestamp", T.TimestampType(), True), # Use TimestampType for proper sessionization
        StructField("musicbrainz-artist-id", StringType(), True),
        StructField("artist-name", StringType(), True),
        StructField("musicbrainz-track-id", StringType(), True),
        StructField("track-name", StringType(), True)
    ])
    
    # STEP 1: Load datasets
    music_user_timestamps_df = load_dataset(spark_session=spark_session, data_path=MUSIC_USER_TIMESTAMPS_DATASET_DIR, schema=music_user_timestamps_schema)
    user_profiles_df = load_dataset(spark_session=spark_session, data_path=USER_PROFILES_DATASET_DIR).withColumnRenamed("#id", "userid")

    # STEP 2: Data Cleaning - Filtering out nulls and invalid users (non-overlapping userIDs)
    filtered_music_user_timestamps_df = preprocess_music_data(music_df=music_user_timestamps_df, profiles_df=user_profiles_df)

    # STEP 3: Sessionize the data - define a global_session_key
    music_user_sessions_df = generate_session_info(df=filtered_music_user_timestamps_df, gap_threshold=1200)

    # STEP 4: derive session data for user with highest number of sessions
    top_user_df = get_top_user_by_session_count(music_user_sessions_df)
    target_userid = top_user_df.collect()[0]["userid"]
    daily_series_df = get_daily_session_counts(music_user_sessions_df, target_userid)

    # STEP 5 [ML]: Forecast the next 3 month of sessions
    # ==========================================================================
    # 5.1: PREPARE DATASET FOR ML CONSUMPTION
    # ==========================================================================
    daily_series_pdf = daily_series_df.toPandas()
    daily_series_pdf['date'] = pd.to_datetime(daily_series_pdf['date'])
    daily_series_pdf.set_index('date', inplace=True)
    
    # Ensure a clean daily frequency index
    daily_series_pdf_clean = daily_series_pdf.asfreq('D', fill_value=0) # D=daily

    # ==========================================================================
    # 5.2: SARIMAX MODEL BACKTEST/HOLDOUT VALIDATION TEST
    # ==========================================================================
    eval_days = 30
    # Guard check to verify user dataset length supports a 30-day temporal split
    if len(daily_series_pdf_clean) > eval_days:
        train_data = daily_series_pdf_clean['total_session_count'].iloc[:-eval_days]
        test_data = daily_series_pdf_clean['total_session_count'].iloc[-eval_days:]

        # Train a mock model strictly on historical training partitions
        # (1,1,1) accounts for auto-regressive and moving average trends
        # setting seasonal_order to 7 accounts for weekly seasonality
        eval_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        eval_fit = eval_model.fit(disp=False)
        eval_forecast = eval_fit.get_forecast(steps=eval_days).predicted_mean

        # Build side-by-side comparison DataFrame
        eval_matrix_pdf = pd.DataFrame({
            "actual_sessions": test_data.values,
            "predicted_sessions": eval_forecast.values,
            "absolute_error": (test_data.values - eval_forecast.values)
        }, index=test_data.index)
        
       # Mean Absolute Error (MAE) Eval
        eval_matrix_pdf['absolute_error'] = eval_matrix_pdf['absolute_error'].abs()
        mae = eval_matrix_pdf['absolute_error'].mean()

        print("\n==================================================")
        print("             TIME-SERIES MODEL EVALUATION          ")
        print("==================================================")
        print(f"Model : SARIMAX")
        print(f"Validation Strategy : Backtest Holdout (Last {eval_days} Days)")
        print(f"Target Evaluation User : {target_userid}")
        print(f"Mean Absolute Error (MAE): {mae:.4f} sessions/day")
        # Forcing a clean text output format for the comparison print
        with pd.option_context('display.max_rows', 10, 'display.float_format', lambda x: '%.4f' % x):
            print(eval_matrix_pdf)
        print("==================================================\n")
        output_eval_path = f"{DATASET_OUTPUT_DIRECTORY}30d_backtest_model_eval_user_most_sessions.csv"
        eval_matrix_pdf.to_csv(output_eval_path, index=False)
    else:
        print(f"\n[WARNING] Insufficient history length ({len(daily_series_pdf_clean)} days) for evaluation split. Skipping validation layer.\n")

    # ==========================================================================
    # 5.3: PRODUCTION EXTRAPOLATION (Full History Training Window)
    # ==========================================================================
    # SARIMAX MODEL Production Forecast (3 months from last record)  
    model = SARIMAX(daily_series_pdf_clean['total_session_count'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    model_fit = model.fit(disp=False)

    # Forecast next 3 months (90d)
    forecast_steps = 90
    forecast = model_fit.get_forecast(steps=forecast_steps) # automatically constructs future indices starting exactly one day after last historical record
    forecast_values = forecast.predicted_mean
    
    # Extract confidence intervals (lower/upper limits)
    confidence_intervals = forecast.conf_int()

    # Build future date index for the projection
    last_date = daily_series_pdf_clean.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

    forecast_pdf = pd.DataFrame({
        "date": future_dates.strftime("%Y-%m-%d"),
        "forecasted_sessions": forecast_values.values,
        "lower_bound": confidence_intervals.iloc[:, 0].values,
        "upper_bound": confidence_intervals.iloc[:, 1].values
    })

    # Post-processing polish in Pandas DF
    forecast_pdf['forecasted_sessions'] = forecast_pdf['forecasted_sessions'].clip(lower=0)
    forecast_pdf['lower_bound'] = forecast_pdf['lower_bound'].clip(lower=0)
    forecast_pdf['upper_bound'] = forecast_pdf['upper_bound'].clip(lower=0)

    # Save the Results - no need to use SparkDF since small dataset
    output_prod_path = f"{DATASET_OUTPUT_DIRECTORY}forecast90d_user_most_sessions.csv"
    forecast_pdf.to_csv(output_prod_path, index=False)

    # Print the first 5 days and last 5 days of the 90-day forecast
    print("\n==================================================")
    print("           90-DAY USER SESSION FORECAST           ")
    print("==================================================")
    print(forecast_pdf.head(5))
    print("...")
    print(forecast_pdf.tail(5))
    print("==================================================\n")

    # Stop the Spark session
    spark_session.stop()