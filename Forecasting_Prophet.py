import numpy as np
import pandas as pd
import random
from itertools import product
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import time

plt.style.use('fivethirtyeight')

# Load and preprocess data
df = pd.read_csv('aggregated_remitly_ncas.csv')
df.columns = ['ds', 'country_code', 'y']
df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d')
df['y'] = pd.to_numeric(df['y'], errors='coerce').fillna(0)
all_dates = pd.DataFrame({'ds': pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='D')})
countries_list = df['country_code'].unique()
# Print the number of unique countries
print(f"Found {len(countries_list)} unique countries: {countries_list}")


all_forecasts = []

# Hyperparameter optimization
def optimize_prophet_parameters(df_train, param_sample=50):
    param_grid = {
        'changepoint_prior_scale': [0.5, 1.0, 5.0, 10.0, 20.0],  # Added higher values
        'seasonality_prior_scale': [1.0, 10.0, 100.0, 1000.0],      # Added higher values
        'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'seasonality_mode': ['multiplicative'],             # Added additive mode
        'changepoint_range': [0.9, 0.95, 0.98],
        'n_changepoints': [50, 75, 100]                             # Added n_changepoints
    }
    all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    sampled_params = random.sample(all_params, min(len(all_params), param_sample))
    
    best_params, best_mape = None, float('inf')
    for params in sampled_params:
        try:
            m = Prophet(growth='linear', **params)
            m.fit(df_train)
            df_cv = cross_validation(m, initial='366 days', period='30 days', horizon='90 days')
            df_p = performance_metrics(df_cv, rolling_window=1)
            mape = df_p['mape'].mean()
            if mape < best_mape:
                best_params, best_mape = params, mape
        except Exception as e:
            print(f"Error with parameters {params}: {e}")
    print(f"Best params: {best_params} with MAPE: {best_mape:.2f}%")
    return best_params

# Forecasting for each country
for country in countries_list:
    try:
        print(f"\nStarting forecast for {country}")
        start_time = time.time()
        
        # Prepare country data
        country_df = df[df['country_code'] == country].copy()
        country_df = all_dates.merge(country_df, on='ds', how='left').fillna(0)

        # Find optimal parameters
        best_params = optimize_prophet_parameters(country_df[['ds', 'y']])

        # Fit the model
        m = Prophet(
            growth='linear',
            daily_seasonality=True,      # Enable daily seasonality
            weekly_seasonality=True,     # Enable weekly seasonality
            yearly_seasonality=True,     # Enable yearly seasonality
            **best_params
        )
        m.add_country_holidays(country_name=country)
        m.fit(country_df[['ds', 'y']])

        # Predict future
        future = m.make_future_dataframe(periods=90)
        forecast = m.predict(future)
        forecast['country_code'] = country
        
        # Calculate MAPE for the historical period only
        historical_forecast = forecast[['ds', 'yhat']].merge(
            country_df[['ds', 'y']], 
            on='ds', 
            how='inner'
        )
        
        # Add date range information
        print(f"\nMAPE calculation period:")
        print(f"Start date: {historical_forecast['ds'].min()}")
        print(f"End date: {historical_forecast['ds'].max()}")
        print(f"Number of days: {len(historical_forecast)}")
        
        # Calculate MAPE
        mape = np.mean(np.abs((historical_forecast['y'] - historical_forecast['yhat']) / (historical_forecast['y'] + 1e-10))) * 100
        print(f'MAPE for {country}: {mape:.2f}%')
        
        # Optional: Calculate MAPE for different time periods
        # Last 30 days
        last_30_days = historical_forecast[historical_forecast['ds'] >= historical_forecast['ds'].max() - pd.Timedelta(days=30)]
        mape_30d = np.mean(np.abs((last_30_days['y'] - last_30_days['yhat']) / (last_30_days['y'] + 1e-10))) * 100
        print(f'MAPE (last 30 days) for {country}: {mape_30d:.2f}%')
        
        # Last 90 days
        last_90_days = historical_forecast[historical_forecast['ds'] >= historical_forecast['ds'].max() - pd.Timedelta(days=90)]
        mape_90d = np.mean(np.abs((last_90_days['y'] - last_90_days['yhat']) / (last_90_days['y'] + 1e-10))) * 100
        print(f'MAPE (last 90 days) for {country}: {mape_90d:.2f}%')
        
        all_forecasts.append(forecast)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(country_df['ds'], country_df['y'], label='Historical')
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)
        plt.title(f'Forecast for {country}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(f'forecast_{country}.png')
        plt.close()
        
        print(f"Finished {country} in {(time.time() - start_time):.2f} seconds")
        print(f"Successfully added forecast for {country}")
    
    except Exception as e:
        print(f"Error processing {country}: {str(e)}")
        continue

# Check if we have any forecasts before concatenating
if all_forecasts:
    combined_forecast = pd.concat(all_forecasts, ignore_index=True)
    print("Successfully combined all forecasts")
    combined_forecast.to_csv('Forecast_CT_aggregated_First_Time_Downloads.csv', index=False)
else:
    print("No forecasts were generated successfully")
