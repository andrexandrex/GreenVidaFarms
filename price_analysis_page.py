import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

# Load datasets
_aggregated_precio = None
_filtered_volumen = None
_df_forecast = None
_precio_chakra = None
def set_data(aggregated_precio, filtered_volumen, df_forecast, precio_chakra):
    global _aggregated_precio, _filtered_volumen, _df_forecast, _precio_chakra
    _aggregated_precio = aggregated_precio
    _filtered_volumen  = filtered_volumen
    _df_forecast       = df_forecast
    _precio_chakra     = precio_chakra

# Merge data
def calculate_relative_change_with_flag(df, category):
    """
    Calculate relative changes (30-15, 15-1, 60-30) with a flag for downhill trends.
    """
    category_data = df[df['CATEGORIA'] == category].sort_values(by='Extraction Date')
    last_60_days = category_data.tail(60).set_index('Extraction Date').asfreq('D')
    last_60_days['precio_prom'] = last_60_days['precio_prom'].fillna(method='ffill')
    if last_60_days['precio_prom'].isna().sum() > 0 or len(last_60_days) < 60:
        return None

    mean_60_30 = last_60_days.iloc[:30]['precio_prom'].mean()
    mean_30_15 = last_60_days.iloc[30:45]['precio_prom'].mean()
    mean_15_1 = last_60_days.iloc[45:]['precio_prom'].mean()
    relative_change_30_15 = (mean_15_1 - mean_30_15) / mean_30_15 if mean_30_15 != 0 else None

    return {
        'CATEGORIA': category,
        'mean_60_30': mean_60_30,
        'mean_30_15': mean_30_15,
        'mean_15_1': mean_15_1,
        'relative_change_30_15': relative_change_30_15
    }

category_groups = {
    "Vegetables": [
        "ACELGA", "AJI", "AJO", "ALBAHACA", "ALCACHOFA", "APIO", "ARVEJA", 
        "BETARRAGA", "BROCOLI", "CAIGUA", "CAMOTE", "CEBOLLA", "CHOCLO", 
        "COL", "COLIFLOR", "CULANTRO", "ESPARRAGO", "ESPINACA", "HABA", 
        "LECHUGA", "NABO", "PIMIENTO", "PORO", "TOMATE", "VAINITA", 
        "ZANAHORIA", "ZAPALLO", "PEPINILLO"
    ],
    "Fruits": [
        "ACEITUNA", "AGUAYMANTO", "CARAMBOLA", "CHIRIMOYA", "CIRUELA", 
        "COCO", "COCONA", "DURAZNO", "FRESA", "GRANADA", "GRANADILLA", 
        "GUANABANA", "LIMA", "LIMON", "LUCUMA", "MANDARINA", "MANGO", 
        "MANZANA", "MARACUYA", "MELOCOTON", "MELON", "MEMBRILLO", "NARANJA", 
        "PAPAYA", "PERA", "PIÑA", "PLATANOS", "SANDIA", "TAMARINDO", 
        "TORONJA", "TUNA", "UVA"
    ],
    "Herbs and Spices": [
        "ACHIOTE", "AJONJOLI", "ANIS", "CULANTRO", "HIERBA", "HUACATAY", 
        "OREGANO", "PEREJIL", "ROMERO", "MANZANILLA"
    ],
    "Tubers and Roots": [
        "CAMOTE", "MASHUA", "OCA", "OLLUCO", "PAPA", "YACON", "YUCA"
    ],
    "Grains and Cereals": [
        "ARROZ", "MAIZ", "QUINUA", "TRIGO", "ALFALFA"
    ],
    "Other": [
        "PEPINO", "PALLAR", "PALTA", "ZAPALLO"
    ]
}

import pandas as pd
import plotly.graph_objects as go

def price_forecast_for_categories(
    categories, 
    price_volume_data, 
    df_forecast,        # <--- NEW param: precomputed daily forecasts
    display_days=30
):
    """
    Generate a multi-category forecast graph using precomputed Prophet results 
    and display the results in a single graph.

    Parameters:
        categories (list): List of categories to forecast.
        price_volume_data (DataFrame): The DataFrame containing historical price & volume data.
        df_forecast (DataFrame): DataFrame containing precomputed daily forecasts 
                                 (with columns ['ds','yhat','yhat_lower','yhat_upper','CATEGORIA','FORECAST_TYPE']).
        display_days (int): Number of days to display from the past.
        forecast_days (int): Number of days to forecast into the future (for labeling only).

    Returns:
        Plotly figure: A graph with historical and forecasted price trends for all categories.
    """
    fig = go.Figure()

    for categoria in categories:
        # ------------------------------------
        # 1) Historical data
        # ------------------------------------
        df_filtered = price_volume_data[price_volume_data['CATEGORIA'] == categoria].copy()
        df_filtered['Extraction Date'] = pd.to_datetime(df_filtered['Extraction Date'], errors='coerce')
        df_filtered = df_filtered.dropna(subset=['Extraction Date'])

        # Group daily for historical chart
        df_daily = df_filtered.groupby('Extraction Date').agg({
            'Volumen': 'mean',
            'precio_prom': 'mean'
        }).reset_index()

        # Filter out only the last 'display_days' for plotting historical
        df_daily = df_daily.sort_values('Extraction Date')
        df_daily = df_daily.tail(display_days)  # keep last N days

        # ------------------------------------
        # 2) Forecast data (daily)
        # ------------------------------------
        # Filter df_forecast for this category and FORECAST_TYPE='daily'
        # Also ensure we only keep future predictions (dates > last historical date) if desired
        df_fore = df_forecast[
            (df_forecast['CATEGORIA'] == categoria) &
            (df_forecast['FORECAST_TYPE'] == 'daily')
        ].copy()

        # 'ds' might be stored as date without time; ensure it's datetime for plotting:
        df_fore['ds'] = pd.to_datetime(df_fore['ds'])

        # We can also filter to keep only up to 'forecast_days' out if needed
        if len(df_daily) > 0:
            last_hist_date = df_daily['Extraction Date'].max()
            df_fore = df_fore[df_fore['ds'] > last_hist_date]

        # Sort
        df_fore = df_fore.sort_values('ds')

        # ------------------------------------
        # 3) Combine for continuous line
        # ------------------------------------
        # We’ll rename columns for forecast to be consistent with historical
        df_fore_rename = df_fore.rename(
            columns={'ds': 'Date', 'yhat': 'precio_prom'}
        )
        df_fore_rename['type'] = 'forecast'

        # For confidence intervals, we keep 'yhat_upper' & 'yhat_lower'
        # We won't combine them directly with historical, but we’ll plot them separately.

        # Historical
        df_daily_rename = df_daily.rename(columns={'Extraction Date': 'Date'})
        df_daily_rename['type'] = 'historical'

        # Concatenate them
        combined_data = pd.concat([df_daily_rename[['Date','precio_prom','type']], 
                                   df_fore_rename[['Date','precio_prom','type']]], 
                                  ignore_index=True)
        combined_data = combined_data.sort_values('Date')

        # ------------------------------------
        # 4) Plot lines
        # ------------------------------------
        fig.add_trace(go.Scatter(
            x=combined_data['Date'],
            y=combined_data['precio_prom'],
            mode='lines+markers',
            name=f'{categoria} - Historical & Forecast'
        ))

        # Add confidence intervals for forecast portion
        # (We only plot them where type='forecast')
        fig.add_trace(go.Scatter(
            x=df_fore['ds'],
            y=df_fore['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df_fore['ds'],
            y=df_fore['yhat_lower'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(width=0),
            showlegend=False
        ))

    # Update layout
    fig.update_layout(
        title=f"Price Forecast for Selected Categories ({', '.join(categories)})",
        xaxis_title='Date',
        yaxis_title='Price (S/.)',
        template='plotly_white'
    )

    return fig

# Generate the final_results_sorted
# Generate top 5 categories based on relative change

# Create multi-category forecast graph
def plot_price_vs_volume_with_monthly_line(
    price_volume_data, 
    precio_chakra, 
    df_forecast,        # <--- NEW param: precomputed daily + monthly forecasts
    category, 
    num_days=30
):
    """
    Plot the relationship between price and volume for a given category over the last 'num_days',
    including a monthly average price line (historical from precio_chakra + monthly forecast) 
    and daily forecast (from df_forecast).

    Parameters:
        price_volume_data (DataFrame): The merged DataFrame containing daily price and volume data (historical).
        precio_chakra (DataFrame): The DataFrame containing monthly historical price data.
        df_forecast (DataFrame): Precomputed forecast data, with at least:
            - daily forecast rows (FORECAST_TYPE='daily')
            - monthly forecast rows (FORECAST_TYPE='monthly_chakra')
            columns: ['ds','yhat','yhat_lower','yhat_upper','CATEGORIA','FORECAST_TYPE']
        category (str): The category to visualize.
        num_days (int): The number of days to display for historical daily data (default=30).

    Returns:
        Plotly Figure object.
    """

    # ----------------------------------------------------
    # 1) Historical DAILY data (last num_days)
    # ----------------------------------------------------
    daily_data = price_volume_data[price_volume_data['CATEGORIA'] == category].copy()
    daily_data = daily_data.sort_values(by='Extraction Date')
    daily_data.set_index('Extraction Date', inplace=True)
    # Ensure continuity
    daily_data = daily_data.asfreq('D')
    daily_data['precio_prom'] = daily_data['precio_prom'].fillna(method='ffill')
    daily_data['Volumen'] = daily_data['Volumen'].fillna(0)

    # The last day in daily historical
    last_day_daily = daily_data.index[-1]  
    start_date_daily = last_day_daily - pd.Timedelta(days=num_days)

    # Slice the daily historical data
    filtered_daily_data = daily_data.loc[start_date_daily:last_day_daily]

    # ----------------------------------------------------
    # 2) DAILY Forecast from df_forecast
    # ----------------------------------------------------
    df_fore_daily = df_forecast[
        (df_forecast['CATEGORIA'] == category) & 
        (df_forecast['FORECAST_TYPE'] == 'daily')
    ].copy()

    df_fore_daily['ds'] = pd.to_datetime(df_fore_daily['ds'])
    # Keep only forecast dates strictly after your last historical daily date:
    df_fore_daily = df_fore_daily[df_fore_daily['ds'] > last_day_daily]
    df_fore_daily.sort_values(by='ds', inplace=True)

    # Merge daily historical + daily forecast into one table (for a continuous line)
    hist_daily_df = filtered_daily_data.reset_index()[['Extraction Date','precio_prom','Volumen']]
    hist_daily_df.rename(columns={'Extraction Date':'ds'}, inplace=True)
    hist_daily_df['type'] = 'historical'

    fore_daily_df = df_fore_daily[['ds','yhat','yhat_lower','yhat_upper']].copy()
    fore_daily_df.rename(columns={'yhat':'precio_prom'}, inplace=True)
    fore_daily_df['type'] = 'forecast'

    combined_daily = pd.concat(
        [hist_daily_df[['ds','precio_prom','Volumen','type']],
         fore_daily_df[['ds','precio_prom','type']]], 
        ignore_index=True
    )
    combined_daily.sort_values(by='ds', inplace=True)

    # ----------------------------------------------------
    # 3) Historical MONTHLY data from precio_chakra
    # ----------------------------------------------------
    # We'll build monthly_data (historical)
    monthly_data = precio_chakra[precio_chakra['CATEGORIA'] == category].copy()
    # Create a datetime column from year & month
    monthly_data['Date'] = pd.to_datetime(
        monthly_data['AÑO'].astype(str) + '-' + monthly_data['MES'].astype(str) + '-01'
    )
    # Group by month
    monthly_data = monthly_data.groupby('Date')['PRECIO_CHACRA'].mean().reset_index()
    monthly_data = monthly_data.sort_values('Date')

    # Convert to an index
    monthly_data.set_index('Date', inplace=True)
    monthly_data.rename(columns={'PRECIO_CHACRA':'y'}, inplace=True)

    # ----------------------------------------------------
    # 4) MONTHLY Forecast from df_forecast
    # ----------------------------------------------------
    df_fore_monthly = df_forecast[
        (df_forecast['CATEGORIA'] == category) & 
        (df_forecast['FORECAST_TYPE'] == 'monthly_chakra')
    ].copy()

    df_fore_monthly['ds'] = pd.to_datetime(df_fore_monthly['ds'])
    df_fore_monthly.sort_values(by='ds', inplace=True)
    # rename 'yhat' -> 'y' for easy merging
    df_fore_monthly.rename(columns={'yhat':'y'}, inplace=True)
    df_fore_monthly.set_index('ds', inplace=True)

    # We’ll mimic your original alignment logic:
    end_date_monthly = last_day_daily + pd.Timedelta(days=30) + pd.DateOffset(months=1)
    start_date_monthly = end_date_monthly - pd.Timedelta(days=num_days) - pd.DateOffset(months=2)

    # Slice the monthly historical
    # (In your original code, you took monthly_data up to last_day_daily, plus some offset)
    # We'll do similar, but not identical if you prefer to keep it minimal:
    # monthly_data covers up to the real last monthly date in historical.
    # We'll combine them with the forecast (which is presumably after that).
    combined_monthly_data = pd.concat(
        [
            monthly_data.loc[monthly_data.index <= last_day_daily],  # historical monthly up to last_day_daily
            df_fore_monthly  # all forecast rows
        ],
        ignore_index=False
    ).sort_index()

    # Then slice to [start_date_monthly, end_date_monthly]
    combined_monthly_data = combined_monthly_data.loc[
        (combined_monthly_data.index >= start_date_monthly) & 
        (combined_monthly_data.index <= end_date_monthly)
    ]

    # For confidence intervals
    forecast_monthly_filtered = df_fore_monthly.loc[
        (df_fore_monthly.index >= start_date_monthly) & 
        (df_fore_monthly.index <= end_date_monthly)
    ]

    # ----------------------------------------------------
    # 5) Build the Plotly Figure
    # ----------------------------------------------------
    fig = go.Figure()

    # A) Daily historical line
    df_hist_only = combined_daily[combined_daily['type'] == 'historical']
    fig.add_trace(go.Scatter(
        x=df_hist_only['ds'],
        y=df_hist_only['precio_prom'],
        mode='lines+markers',
        name='Daily Price (Historical)',
        line=dict(color='blue'), 
        marker=dict(color='blue')
    ))

    # B) Daily forecast line
    df_fore_only = combined_daily[combined_daily['type'] == 'forecast']
    fig.add_trace(go.Scatter(
        x=df_fore_only['ds'],
        y=df_fore_only['precio_prom'],
        mode='lines+markers',
        name='Daily Price (Forecast)',
        line=dict(dash='dot', color='blue')
    ))

    # Confidence intervals for daily
    fig.add_trace(go.Scatter(
        x=df_fore_daily['ds'],
        y=df_fore_daily['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df_fore_daily['ds'],
        y=df_fore_daily['yhat_lower'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(width=0),
        showlegend=False
    ))

    # C) Daily volume (bar), only for the historical portion
    fig.add_trace(go.Bar(
        x=df_hist_only['ds'],
        y=df_hist_only['Volumen'],
        name='Daily Volume',
        marker_color='orange',
        opacity=0.6,
        yaxis='y2'
    ))

    # D) Monthly historical + forecast line
    # combined_monthly_data has both historical monthly (index <= real last month)
    # and forecast monthly (index beyond real last month).
    fig.add_trace(go.Scatter(
        x=combined_monthly_data.index,
        y=combined_monthly_data['y'],
        mode='lines+markers',
        name='Monthly Avg Price (Historical + Forecast)',
        line=dict(color='green'), 
        marker=dict(color='green')
    ))

    # Confidence intervals for monthly (only where forecast exists)
    fig.add_trace(go.Scatter(
        x=forecast_monthly_filtered.index,
        y=forecast_monthly_filtered['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast_monthly_filtered.index,
        y=forecast_monthly_filtered['yhat_lower'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0,200,100,0.2)',
        line=dict(width=0),
        showlegend=False
    ))

    # ----------------------------------------------------
    # 6) Layout / Axis config
    # ----------------------------------------------------
    fig.update_layout(
        title=f"Price vs. Volume for {category} (Last {num_days} Days) + Forecast",
        xaxis_title='Date',
        yaxis=dict(
            title='Price (S/.)',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title='Volume',
            titlefont=dict(color='orange'),
            tickfont=dict(color='orange'),
            anchor='x',
            overlaying='y',
            side='right'
        ),
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def plot_category_group_with_dual_axes(data, category_groups, group_name, days=30):
    """
    Create a grouped bar chart for the selected category group with dual axes.

    Parameters:
    - data (DataFrame): Contains the category, price, volume, and date data.
    - category_groups (dict): A dictionary where keys are group names and values are category lists.
    - group_name (str): The selected group name to visualize.
    - days (int): Number of days to calculate the average price and total volume.
    """
    # Get the categories for the selected group
    if group_name not in category_groups:
        raise ValueError(f"Group '{group_name}' not found in category_groups.")
    category_group = category_groups[group_name]

    # Filter data for the last `days`
    data['Date'] = pd.to_datetime(data['Extraction Date'])
    last_date = data['Date'].max()
    start_date = last_date - pd.Timedelta(days=days)
    filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= last_date)]

    # Filter data for the selected category group
    group_data = filtered_data[filtered_data['CATEGORIA'].isin(category_group)]

    # Calculate average price and total volume for each category
    category_stats = group_data.groupby('CATEGORIA').agg(
        avg_price=('precio_prom', 'mean'),
        total_volume=('Volumen', 'sum')
    ).reset_index()

    # Create the figure
    fig = go.Figure()

    # Add average price bars (left y-axis)
    fig.add_trace(go.Bar(
        x=category_stats['CATEGORIA'],
        y=category_stats['avg_price'],
        name="Average Price (S/.)",
        marker_color='blue',
        yaxis='y1',  # Attach to left y-axis
        offsetgroup=0  # Group for alignment
    ))

    # Add total volume bars (right y-axis)
    fig.add_trace(go.Bar(
        x=category_stats['CATEGORIA'],
        y=category_stats['total_volume'],
        name="Total Volume",
        marker_color='orange',
        yaxis='y2',  # Attach to right y-axis
        offsetgroup=1  # Group for alignment
    ))

    # Update layout for dual y-axes
    fig.update_layout(
        barmode='group',  # Side-by-side bars
        title=f"Average Price and Total Volume for {group_name.upper()} (Last {days} Days)",
        xaxis_title="Category",
        xaxis=dict(
            tickangle=45  # Rotate category labels
        ),
        yaxis=dict(
            title="Average Price (S/.)",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
        ),
        yaxis2=dict(
            title="Total Volume",
            titlefont=dict(color="orange"),
            tickfont=dict(color="orange"),
            overlaying="y",  # Overlay on the same plot
            side="right"
        ),
        legend_title="Metric",
        template="plotly_white"
    )

    return fig


