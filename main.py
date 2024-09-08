from fasthtml import FastHTML, Title, Main, H1, Img
from fasthtml.common import *
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta
import io
import base64
import json

app = FastHTML()

# Load ETF info from JSON
with open('etf_info_favourites.json', 'r') as f:
    etf_data = json.load(f)
    ETF_INFO = {etf['ticker']: etf for etf in etf_data['etfs']}

TICKERS = list(ETF_INFO.keys())

def get_past_date(days_ago):
    today = date.today()
    return (today - timedelta(days=days_ago)).isoformat()

def fit_model(data, start_date):
    filtered_data = data[data['Date'] >= pd.to_datetime(start_date)]
    X = filtered_data[['DateNumeric']]
    y = filtered_data['Close']
    model = LinearRegression()
    model.fit(X, y)
    return model, filtered_data

@app.get("/")
def home():
    ticker_data = []
    for ticker in TICKERS:
        etf_data = yf.download(f"{ticker}.AX", start='1973-01-01')  # Fetch 50 years of data
        if etf_data.empty:
            continue  # Skip this ticker if no data is available
        etf_data['Date'] = etf_data.index
        etf_data['DateNumeric'] = etf_data['Date'].apply(lambda date: date.toordinal())

        models = {
            '50Y': fit_model(etf_data, get_past_date(50 * 365)),
            '20Y': fit_model(etf_data, get_past_date(20 * 365)),
            '3Y': fit_model(etf_data, get_past_date(3 * 365)),
            '1Y': fit_model(etf_data, get_past_date(365)),
        }

        latest_date = etf_data['DateNumeric'].iloc[-1]
        latest_price = etf_data['Close'].iloc[-1]

        model_50y, _ = models['50Y']
        predicted_price_50y = model_50y.predict([[latest_date]])[0]
        delta_50y = predicted_price_50y - latest_price

        slope_50y = model_50y.coef_[0]
        yearly_change = round(slope_50y * 365, 2)
        pct_yearly_change = round((yearly_change / latest_price) * 100, 2)
        length_of_time = round((etf_data['Date'].iloc[-1] - etf_data['Date'].iloc[0]).days / 365, 2)

        ticker_data.append((ticker, etf_data, models, delta_50y, pct_yearly_change, length_of_time))

    # Sort tickers based on 50Y delta (not absolute value)
    ticker_data.sort(key=lambda x: x[4], reverse=True)

    results = [Title("ETF Analysis"), H1("ETF Analysis")]
    for ticker, etf_data, models, _, yearly_change, length_of_time in ticker_data:
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'{ticker} - {ETF_INFO[ticker]["name"]} ({ETF_INFO[ticker]["etfType"]})\n{ETF_INFO[ticker]["description"]}\nyearly change: {yearly_change}%\nyears of data: {length_of_time}', fontsize=14, wrap=True)

        latest_date = etf_data['DateNumeric'].iloc[-1]
        latest_price = etf_data['Close'].iloc[-1]

        for i, (timespan, (model, data)) in enumerate(models.items()):
            input_data = pd.DataFrame([[latest_date]], columns=['DateNumeric'])
            predicted_price = model.predict(input_data)[0]

            slope_50y = model_50y.coef_[0]
            yearly_change = round(slope_50y * 365, 2)
            length_of_time = round((etf_data['Date'].iloc[-1] - etf_data['Date'].iloc[0]).days / 365, 2)

            axs[i].plot(data['Date'], data['Close'], label='Actual Price', color='black', linewidth=1)
            axs[i].plot(data['Date'], model.predict(data[['DateNumeric']]), label=f'{timespan} Prediction', linewidth=1)
            axs[i].set_title(timespan, fontsize=12)
            axs[i].set_ylabel('Price (AUD)', fontsize=8)
            axs[i].legend(fontsize=6, loc='upper left')
            axs[i].tick_params(axis='both', which='major', labelsize=6)

            delta = predicted_price - latest_price
            buy_sell = "Buy" if delta > 0 else "Sell"
            delta_percent = (delta / latest_price) * 100

            color = "green" if delta > 0 else "red"
            axs[i].annotate(f'Prediction: ${predicted_price:.2f}\nDelta: ${delta:.2f} ({delta_percent:.2f}%)\n{buy_sell}',
                            xy=(0.05, 0.95), xycoords='axes fraction',
                            fontsize=8, ha='left', va='top',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            color=color)

        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        results.append(Img(src=f"data:image/png;base64,{image_base64}", alt=f"{ticker} Analysis", style="width: 100%; margin-top: 10px;"))
        plt.close(fig)

    return Main(*results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
