
# Sales Pulse

Sales Pulse is a multi-dataset sales analytics and forecasting application.

## Features

- Upload multiple CSV and Excel datasets.
- Automatically suggest column mappings.
- Clean and validate uploaded data.
- Analyze different sales datasets independently.
- Filter historical sales by:
  - Last 30 days
  - Last 3 months
  - Last 4 months
  - Last 6 months
  - Last year
  - All available history
  - Custom date range
- View daily, weekly, monthly and quarterly trends.
- Generate quarterly sales and growth reports.
- Analyze product, category, store and regional performance.
- Calculate units sold and estimated gross profit when possible.
- Forecast revenue or units sold.
- Compare baseline, ARIMA and Prophet models.
- Evaluate predictions using MAE, RMSE and sMAPE.
- Download reports and predictions as CSV files.

## Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Statsmodels
- Prophet

## Requirements

Python 3.10 or newer is recommended.

## Installation

Create a virtual environment:

    py -m venv .venv

Activate it on Windows:

    .venv\Scripts\activate

Install the dependencies:

    pip install -r requirements.txt

Run the application:

    streamlit run app.py

## Dataset requirements

For sales analysis, provide:
- A date column
- A revenue/sales column

Alternatively, provide:
- A date column
- A quantity column
- A unit-price column

Optional columns enable additional reports and visualizations.

Always review the automatic column mapping before analyzing a dataset.

## Forecasting notes

Forecasting uses the full available history for the selected dataset, independently of the historical dashboard date filter.

Predictions are estimates. Missing dates, promotions, returns,
stockouts and seasonal patterns can affect accuracy.

The quarterly report groups records into calendar quarters.
When a selected date range covers only part of a quarter,
the displayed total represents only the selected portion.

# cd Sales_pulse Project
# dir
# py -m venv .venv
# .venv\Scripts\activate.bat
# python -m pip install --upgrade pip
# python -m pip install -r requirements.txt
# python -m py_compile app.py preprocessing.py forecasting.py evaluation.py
# python -m streamlit run app.py
