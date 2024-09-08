from fasthtml import FastHTML
from fasthtml.common import *
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta
import io
import base64
import json

app = FastHTML()

# Load ETF info from JSON
with open('etf_info_favourites.json', 'r') as f:
    ETF_INFO = {etf['ticker']: etf for etf in json.load(f)['etfs']}

TICKERS = list(ETF_INFO.keys())

def get_past_date(days_ago):
    return (date.today() - timedelta(days=days_ago)).isoformat()

def fit_model(data, start_date):
    filtered_data = data[data['Date'] >= pd.to_datetime(start_date)]
    X = filtered_data[['DateNumeric']]
    y = filtered_data['Close']
    model = LinearRegression()
    model.fit(X, y)
    return model, filtered_data

def calculate_cagr(start_price, end_price, years):
    # Compound Annual Growth Rate (CAGR)
    # CAGR = (Ending Value / Beginning Value) ^ (1 / Number of Years) - 1
    return (end_price / start_price) ** (1 / years) - 1

def create_plot(etf_data, models):
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    latest_date = etf_data['DateNumeric'].iloc[-1]
    latest_price = etf_data['Close'].iloc[-1]

    for i, (timespan, (model, data)) in enumerate(models.items()):
        predicted_price = model.predict(pd.DataFrame({'DateNumeric': [latest_date]}))[0]
        delta_percent = ((latest_price - predicted_price) / predicted_price) * 100
        direction = "above" if delta_percent > 0 else "below"
        color = "green" if delta_percent < 0 else "red"

        axs[i].plot(data['Date'], data['Close'], label='Actual Price', color='black', linewidth=1)
        axs[i].plot(data['Date'], model.predict(data[['DateNumeric']]), label=f'{timespan} Prediction', linewidth=1)
        axs[i].set_title(f"Based on {timespan} of data", fontsize=14)
        axs[i].set_ylabel('Price (AUD)', fontsize=10)
        axs[i].tick_params(axis='both', which='major', labelsize=8)
        axs[i].annotate(f'Price is {abs(delta_percent):.2f}% {direction} prediction',
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=18, ha='left', va='top', color=color)
        # axs[i].legend(fontsize=8)

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150)
    buffer.seek(0)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()

def process_ticker(ticker):
    etf_data = yf.download(f"{ticker}.AX", start='2000-01-01')
    if etf_data.empty:
        return None

    etf_data['Date'] = etf_data.index
    etf_data['DateNumeric'] = etf_data['Date'].apply(lambda date: date.toordinal())

    models = {
        '20 Years': fit_model(etf_data, get_past_date(20 * 365)),
        '3 Years': fit_model(etf_data, get_past_date(3 * 365)),
        '1 Year': fit_model(etf_data, get_past_date(365)),
    }

    model_20y, data_20y = models['20 Years']
    start_price = data_20y['Close'].iloc[0]
    end_price = data_20y['Close'].iloc[-1]
    years = (data_20y['Date'].iloc[-1] - data_20y['Date'].iloc[0]).days / 365.25

    cagr = calculate_cagr(start_price, end_price, years)
    pct_yearly_change = round(cagr * 100, 2)
    length_of_time = round(years, 2)

    image_base64 = create_plot(etf_data, models)

    return {
        'ticker': ticker,
        'title': ETF_INFO[ticker]['title'],
        'pct_yearly_change': pct_yearly_change,
        'length_of_time': length_of_time,
        'image_base64': image_base64
    }

@app.get("/")
def home():
    ticker_data = [process_ticker(ticker) for ticker in TICKERS]
    ticker_data = [data for data in ticker_data if data is not None]
    ticker_data.sort(key=lambda x: x['title'])

    style = Style("""
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f0f4f8; }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; text-align: center; }
        h2 { color: #2980b9; margin-top: 0; }
        .etf-card { background-color: #ffffff; border-radius: 12px; padding: 25px; margin-bottom: 30px; box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1); }
        .etf-info { margin-bottom: 20px; }
        .etf-stats { display: flex; justify-content: flex-start; margin-bottom: 20px; }
        .stat { padding: 10px 20px 10px 0; }
        .stat-label { font-size: 0.9em; color: #7f8c8d; margin-bottom: 2px; }
        .stat-value { font-size: 1.1em; font-weight: bold; color: #2c3e50; }
        img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); }
        .disclaimer { color: #7f8c8d; font-style: italic; margin-bottom: 20px; }
    """)

    results = [
        style,
        Title("ETF Analysis"),
        H1("ETF Performance Analysis"),
        P("This is a simple analysis of the performance of some ETFs in the Australian market. This is not financial advice.", cls="disclaimer"),
    ]

    for data in ticker_data:
        ticker = data['ticker']
        etf_info = ETF_INFO[ticker]
        results.append(Div(
            H2(f"{etf_info['title']} ({ticker})"),
            Div(
                P(f"{etf_info['name']} ({etf_info['etfType']})"),
                P(f"{etf_info['description']}"),
                cls="etf-info"
            ),
            Div(
                Div(
                    Div("Compound Annual Growth Rate", cls="stat-label"),
                    Div(f"{data['pct_yearly_change']:.2f}%", cls="stat-value"),
                    cls="stat"
                ),
                Div(
                    Div("Data Period", cls="stat-label"),
                    Div(f"{data['length_of_time']:.2f} years", cls="stat-value"),
                    cls="stat"
                ),
                cls="etf-stats"
            ),
            Img(src=f"data:image/png;base64,{data['image_base64']}", alt=f"{ticker} Analysis"),
            cls="etf-card"
        ))

    return Main(*results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
