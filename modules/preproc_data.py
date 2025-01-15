import pandas as pd
def update_chatbot_data():
    DATA_PATH_PRICE = "data/data_raw/precio_final_merged_latest.xlsx"
    DATA_PATH_VOLUME = "data/data_raw/volumen_final_merged_latest.xlsx"
    DATA_PATH_FORECAST = "data/data_raw/forecast_latest.xlsx"
    DATA_PATH_CHAKRA = "data/data_raw/chakra_final_merged_v3.xlsx"
    precio_chakra = pd.read_excel(DATA_PATH_CHAKRA)
    aggregated_precio = pd.read_excel(DATA_PATH_PRICE)
    filtered_volumen = pd.read_excel(DATA_PATH_VOLUME)
    df_forecast = pd.read_excel(DATA_PATH_FORECAST)
    price_volume_data = filtered_volumen.merge(
        aggregated_precio, on=['CATEGORIA', 'Extraction Date'], how='outer'
    )
    price_volume_data['precio_prom'] = price_volume_data['precio_prom'].fillna(method='ffill')
    price_volume_data['Volumen'] = price_volume_data['Volumen'].fillna(0)

    precio_chakra['Date'] = pd.to_datetime(
            precio_chakra['AÑO'].astype(str) + '-' + precio_chakra['MES'].astype(str) + '-01'
        )

    last_hist_date = price_volume_data['Extraction Date'].max()
    data_last_price = df_forecast[(df_forecast['ds']>last_hist_date) & (df_forecast['FORECAST_TYPE']=='daily')]
    data_last = data_last_price.rename(columns={'ds': 'Extraction Date', 'yhat': 'precio_prom'}
            )
    data_last = data_last.drop(['yhat_lower','yhat_upper'],axis = 1)
    data_final_daily_price_market  = price_volume_data.merge(
        data_last, on=['CATEGORIA', 'Extraction Date'], how='outer'
    )
    data_final_daily_price_market['precio_prom'] = data_final_daily_price_market['precio_prom_x'].fillna(data_final_daily_price_market['precio_prom_y'])
    data_final_daily_price_market = data_final_daily_price_market.drop(['precio_prom_x', 'precio_prom_y'], axis=1)


    precio_chakra2 = precio_chakra.rename(columns={'Date': 'Extraction Date', 'PRECIO_CHACRA': 'precio_prom'})
    last_date_chakra = precio_chakra2['Extraction Date'].max()
    data_last_price = df_forecast[
        (df_forecast['ds'] > last_date_chakra) & 
        (df_forecast['FORECAST_TYPE'] == 'monthly_chakra')
    ]
    data_last = data_last_price.rename(columns={'ds': 'Extraction Date', 'yhat': 'precio_prom'})
    data_last = data_last.drop(['yhat_lower', 'yhat_upper'], axis=1)
    data_final_monthly_price_farmers = precio_chakra2.merge(
        data_last, 
        on=['CATEGORIA', 'Extraction Date'], 
        how='outer'
    )
    data_final_monthly_price_farmers['precio_prom'] = data_final_monthly_price_farmers['precio_prom_x'].fillna(data_final_monthly_price_farmers['precio_prom_y'])
    data_final_monthly_price_farmers = data_final_monthly_price_farmers.drop(['precio_prom_x', 'precio_prom_y','Unnamed: 0','AÑO','MES'], axis=1)
    data_final_daily_price_market = data_final_daily_price_market.drop(['Unnamed: 0_x','Unnamed: 0_y'],axis = 1)

    data_final_daily_price_market.to_excel('data/data_chatbot/precio_volumen_mercado_diario.xlsx')
    data_final_monthly_price_farmers.to_excel('data/data_chatbot/precio_agricultores_mensual.xlsx')
    