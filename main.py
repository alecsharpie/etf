from fasthtml import FastHTML
from fasthtml.common import *
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta
import io
import base64
import json

app = FastHTML()

with open('etf_info_favourites.json', 'r') as f:
    ETF_INFO = {etf['ticker']: etf for etf in json.load(f)['etfs']}

TICKERS = list(ETF_INFO.keys())

def get_past_date(days_ago):
    return (date.today() - timedelta(days=days_ago)).isoformat()

def fit_model(data, start_date):
    filtered_data = data[data['Date'] >= pd.to_datetime(start_date)]
    if filtered_data.empty:
        return None, None
    X = filtered_data[['DateNumeric']]
    y = filtered_data['Close']
    model = LinearRegression()
    model.fit(X, y)
    return model, filtered_data

def calculate_cagr(start_price, end_price, years):
    return (end_price / start_price) ** (1 / years) - 1

def create_plot(etf_data, model, data):
    fig, ax = plt.subplots(figsize=(12, 6))  # Increased figure size

    if model is None or data is None:
        ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
        prediction_info = None
    else:
        ax.plot(data['Date'], data['Close'], label='Actual Price', color='black', linewidth=1)
        ax.plot(data['Date'], model.predict(data[['DateNumeric']]), label='Prediction', linewidth=1)
        ax.set_ylabel('Price (AUD)', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        # remove legend
        ax.legend().set_visible(False)
        # hide top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # add light grid to y axis
        ax.yaxis.grid(color='gray', linestyle='--', alpha=0.5)
        # axis labels
        plt.ylabel('Price (AUD)', fontsize=16)
        # axis ticks
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)


        # Format y-axis labels as currency
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'${x:,.2f}'))

        latest_date = etf_data['DateNumeric'].iloc[-1]
        latest_price = etf_data['Close'].iloc[-1]
        predicted_price = model.predict(pd.DataFrame({'DateNumeric': [latest_date]}))[0]
        delta_percent = ((latest_price - predicted_price) / predicted_price) * 100
        prediction_info = {
            'delta_percent': delta_percent,
            'direction': 'above' if delta_percent > 0 else 'below'
        }

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')  # Added bbox_inches='tight' to reduce whitespace
    buffer.seek(0)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode(), prediction_info

def process_ticker(ticker):
    etf_data = yf.download(f"{ticker}.AX", start='2000-01-01')
    if etf_data.empty:
        return None

    etf_data['Date'] = etf_data.index
    etf_data['DateNumeric'] = etf_data['Date'].apply(lambda date: date.toordinal())

    total_years = (etf_data.index[-1] - etf_data.index[0]).days / 365.25
    max_years = min(20, total_years)

    models = {
        f'Max ({max_years:.1f} Years)': fit_model(etf_data, etf_data.index[0]),
        '3 Years': fit_model(etf_data, get_past_date(3 * 365)),
        '1 Year': fit_model(etf_data, get_past_date(365)),
    }

    plots_and_cagr = {}
    for period, (model, data) in models.items():
        plot, prediction_info = create_plot(etf_data, model, data)

        if model is not None and data is not None:
            start_price = data['Close'].iloc[0]
            end_price = data['Close'].iloc[-1]
            years = (data['Date'].iloc[-1] - data['Date'].iloc[0]).days / 365.25
            cagr = calculate_cagr(start_price, end_price, years)
            cagr_data = {
                'cagr': round(cagr * 100, 2),
                'years': round(years, 2)
            }
        else:
            cagr_data = None

        plots_and_cagr[period] = {
            'plot': plot,
            'cagr_data': cagr_data,
            'prediction_info': prediction_info
        }

    return {
        'ticker': ticker,
        'title': ETF_INFO[ticker]['title'],
        'plots_and_cagr': plots_and_cagr
    }

@app.get("/")
def home():
    ticker_data = [process_ticker(ticker) for ticker in TICKERS]
    ticker_data = [data for data in ticker_data if data is not None]
    ticker_data.sort(key=lambda x: x['title'])

    style = Style("""
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f0f4f8;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            text-align: center;
        }
        h2 {
            color: #2980b9;
            margin-top: 0;
        }
        .etf-card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        }
        .etf-info {
            margin-bottom: 20px;
        }
        .plot-cards {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }
        .plot-card {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            display: flex;
            flex-direction: column;
        }
        .plot-title {
            font-size: 1em;
            font-weight: bold;
            margin-bottom: 5px;
            color: #2c3e50;
        }
        .plot-info {
            margin-top: 5px;
            font-size: 0.8em;
            color: #34495e;
        }
        .prediction-info {
            margin-top: 5px;
            font-weight: bold;
            font-size: 0.9em;
        }
        .above { color: #e74c3c; }
        .below { color: #2ecc71; }
        .cagr-info {
            margin-top: 5px;
        }
        .cagr-label {
            font-size: 0.8em;
            color: #7f8c8d;
        }
        .cagr-value {
            font-size: 0.9em;
            font-weight: bold;
            color: #2c3e50;
        }
        img {
            width: 100%;
            height: auto;
            border-radius: 8px;
        }
        .disclaimer {
            color: #7f8c8d;
            font-style: italic;
            margin-bottom: 20px;
        }
        @media (max-width: 1200px) {
            .plot-cards {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        @media (max-width: 800px) {
            .plot-cards {
                grid-template-columns: 1fr;
            }
        }
    """)

    results = [
        style,
        Title("ETF Analysis"),
        H1("ETF Analysis"),
        P("This is not financial advice, this is a simple analysis of some ETFs in the Australian market. It's based on the mean reversion principle & linear modelling. I hope the delta between actual and predicted can help guide the proportions of my buys.", cls="disclaimer"),
        Span("Created by ", A("Alec Sharp", href="https://www.alecsharpie.me/"), cls="disclaimer"),
    ]

    for data in ticker_data:
        ticker = data['ticker']
        etf_info = ETF_INFO[ticker]
        plot_cards = []

        for period, info in data['plots_and_cagr'].items():
            cagr_info = ""
            if info['cagr_data']:
                cagr_info = Div(
                    Div("CAGR:", cls="cagr-label"),
                    Div(f"{info['cagr_data']['cagr']}%", cls="cagr-value"),
                    cls="cagr-info"
                )

            prediction_info = ""
            if info['prediction_info']:
                direction = info['prediction_info']['direction']
                delta = abs(info['prediction_info']['delta_percent'])
                prediction_info = Div(
                    f"Price is {delta:.2f}% {direction} prediction",
                    cls=f"prediction-info {direction}"
                )

            plot_cards.append(
                Div(
                    Div(period, cls="plot-title"),
                    Img(src=f"data:image/png;base64,{info['plot']}", alt=f"{ticker} {period} Analysis"),
                    prediction_info,
                    cagr_info,
                    cls="plot-card"
                )
            )

        results.append(Div(
            H2(f"{etf_info['title']} ({ticker})"),
            Div(
                P(f"{etf_info['name']} ({etf_info['etfType']})"),
                P(f"{etf_info['description']}"),
                cls="etf-info"
            ),
            Div(*plot_cards, cls="plot-cards"),
            cls="etf-card"
        ))

    return Main(*results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
