# scripts/train_forecast.py

import pandas as pd
import numpy as np
import prophet
from prophet import Prophet
from datetime import datetime,timedelta

###############################################################################
# 1. LOAD LATEST DATA
###############################################################################
def forecast_prod() -> bool:
    PRICE_LATEST = "precio_final_merged_latest.xlsx"
    VOLUME_LATEST = "volumen_final_merged_latest.xlsx"
    CHAKRA_FILE   = "chakra_final_merged_v3.xlsx"
    BASE_DATA_PATH = "data/data_raw"
    df_price = pd.read_excel(f"{BASE_DATA_PATH}/{PRICE_LATEST}")    # e.g., columns: CATEGORIA, Extraction Date, precio_prom, ...
    df_volume = pd.read_excel(f"{BASE_DATA_PATH}/{VOLUME_LATEST}")  # e.g., columns: CATEGORIA, Extraction Date, Volumen
    df_chakra = pd.read_excel(f"{BASE_DATA_PATH}/{CHAKRA_FILE}")    # e.g., columns: AÑO, MES, CATEGORIA, PRECIO_CHACRA, ...

    # Ensure date columns are datetime
    df_price['Extraction Date'] = pd.to_datetime(df_price['Extraction Date'])
    df_volume['Extraction Date'] = pd.to_datetime(df_volume['Extraction Date'])

    last_date_price = df_price['Extraction Date'].max()
    print(f"Last aggregated date found: {last_date_price.date()}")

    ###############################################################################
    # 2. DEFINE START & END DATES FOR SCRAPING OR CSV LOADING
    ###############################################################################
    # start_date is the day after 'last_date_price'
    start_date = last_date_price + timedelta(days=1)

    # end_date is "yesterday"
    end_date = datetime.now() - timedelta(days=1)

    print(f"Will look for new data from {start_date.date()} to {end_date.date()}")

    # If the start_date > end_date, it means there’s no new data to add.



    # Merge them so we get a single DataFrame with columns:
    #  CATEGORIA, Extraction Date, precio_prom, Volumen, etc.
    df_merged = pd.merge(df_price, df_volume, on=['CATEGORIA','Extraction Date'], how='outer')

    # You might fill missing volumes with 0, or do other cleaning:
    df_merged['Volumen'] = df_merged['Volumen'].fillna(0)
    df_merged['precio_prom'] = df_merged['precio_prom'].fillna(method='ffill')  # Or some other logic

    ###############################################################################
    # 2. FORECAST: PER-CATEGORY DAILY PRICE & VOLUME
    ###############################################################################
    # We'll store all forecasts in a single DataFrame "df_forecasts"
    all_forecasts = []

    # Decide how many days forward to predict
    FORECAST_DAYS = 30

    # Optionally group by category:
    categories = df_merged['CATEGORIA'].dropna().unique()

    for cat in categories:
        df_cat = df_merged[df_merged['CATEGORIA'] == cat].copy()
        # Make sure it’s sorted by date
        df_cat = df_cat.sort_values(by='Extraction Date')
        
        # We’ll skip categories if too few points
        if len(df_cat) < 10:
            continue
        
        # Prepare data for Prophet
        prophet_data = df_cat[['Extraction Date','precio_prom','Volumen']].rename(
            columns={'Extraction Date':'ds','precio_prom':'y'}
        )
        
        # Fill or remove outliers if needed
        prophet_data['y'] = prophet_data['y'].fillna(method='ffill')
        prophet_data['Volumen'] = prophet_data['Volumen'].fillna(0)
        
        # Initialize and train
        model = Prophet(yearly_seasonality = True,seasonality_mode='multiplicative',
        changepoint_prior_scale=0.5,  # Default is 0.05
        seasonality_prior_scale=10.0,  # Default is 10.0
        n_changepoints=50 )
        model.add_regressor('Volumen')
        model.fit(prophet_data)

        def remove_outliers(series):
            q1 = np.percentile(series, 25)
            q3 = np.percentile(series, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return series[(series >= lower_bound) & (series <= upper_bound)]
        prophet_data['y'] = remove_outliers(prophet_data['y'])
        prophet_data['y'] = prophet_data['y'].fillna(method='ffill')  # Fill missing prices

        # Make future dataframe
        future = model.make_future_dataframe(periods=FORECAST_DAYS)
        # Merge volume for future dates (fill with 0 if missing)
        future = pd.merge(future, 
                        prophet_data[['ds','Volumen']], 
                        on='ds', how='left')
        future['Volumen'] = future['Volumen'].fillna(0)
        
        forecast = model.predict(future)
        # Keep only relevant columns
        forecast = forecast[forecast['ds'] > prophet_data['ds'].max()]
        forecast = forecast[['ds','yhat','yhat_lower','yhat_upper']]
        forecast['CATEGORIA'] = cat
        forecast['FORECAST_TYPE'] = 'daily'

        
        all_forecasts.append(forecast)

    ###############################################################################
    # 3. FORECAST: PER-CATEGORY MONTHLY PRICE (CHAKRA)
    ###############################################################################
    # Example approach: For each category, group df_chakra by monthly data, run Prophet if you want:
    df_chakra['Date'] = pd.to_datetime(df_chakra['AÑO'].astype(str) + '-' + df_chakra['MES'].astype(str) + '-01')
    df_chakra_grp = df_chakra.groupby(['CATEGORIA','Date'])['PRECIO_CHACRA'].mean().reset_index()

    for cat in categories:
        # Filter
        df_cat = df_chakra_grp[df_chakra_grp['CATEGORIA'] == cat].copy()
        df_cat = df_cat.sort_values('Date')
        if len(df_cat) < 5:
            continue
        
        # Prepare data
        prophet_data = df_cat.rename(columns={'Date':'ds','PRECIO_CHACRA':'y'})
        prophet_data['y'] = prophet_data['y'].fillna(method='ffill')
        
        model = Prophet()
        model.fit(prophet_data)
        
        future_months = model.make_future_dataframe(periods=6, freq='M')
        forecast = model.predict(future_months)
        forecast = forecast[forecast['ds'] > prophet_data['ds'].max()]
        forecast = forecast[['ds','yhat','yhat_lower','yhat_upper']]
        forecast['CATEGORIA'] = cat
        forecast['FORECAST_TYPE'] = 'monthly_chakra'
        
        all_forecasts.append(forecast)

    ###############################################################################
    # 4. COMBINE & SAVE TO “forecast_latest.xlsx”
    ###############################################################################
    df_forecasts = pd.concat(all_forecasts, ignore_index=True)
    df_forecasts['ds'] = pd.to_datetime(df_forecasts['ds']).dt.date  # or keep as datetime
    df_forecasts.sort_values(by=['CATEGORIA','ds'], inplace=True)

    today_str = datetime.now().strftime('%Y_%m_%d')
    out_file = f"forecast_{today_str}.xlsx"
    df_forecasts.to_excel(out_file, index=False)
    print(f"[INFO] Forecast saved to {out_file}")

    # Also overwrite a “latest” version
    df_forecasts.to_excel(f"{BASE_DATA_PATH}/forecast_latest.xlsx", index=False)
    print("[INFO] Also updated forecast_latest.xlsx")
    return True

