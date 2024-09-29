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
    fig, ax = plt.subplots(figsize=(12, 6))

    if model is None or data is None:
        ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
        prediction_info = None
    else:
        ax.plot(data['Date'], data['Close'], label='Actual Price', color='#2c3e50', linewidth=2)
        ax.plot(data['Date'], model.predict(data[['DateNumeric']]), label='Prediction', color='#34495e', linewidth=2, linestyle='--')
        ax.set_ylabel('Price (AUD)', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(color='#ecf0f1', linestyle='--', alpha=0.7)
        plt.ylabel('Price (AUD)', fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

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
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
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
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        :root {
            --primary-color: #3498db;
            --secondary-color: #2c3e50;
            --accent-color-1: #e74c3c;
            --accent-color-2: #2ecc71;
            --background-color: #f5f7fa;
            --card-background: #ffffff;
            --text-color: #2c3e50;
            --light-text-color: #7f8c8d;
            --border-color: #e0e0e0;
        }

        body {
            font-family: 'Poppins', sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px;
            background-color: var(--background-color);
        }

        .header-section {
            background-color: white;
            padding: 60px 40px;
            border-radius: 12px;
            margin-bottom: 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            border: 1px solid var(--border-color);
        }

        .header-section::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100%25' height='100%25' viewBox='0 0 800 400'%3E%3Cpath d='M0 200 C100 160, 100 240, 200 200 S300 160, 400 200 S500 240, 600 200 S700 160, 800 200' stroke='rgba(231, 76, 60, 0.3)' fill='none' stroke-width='2'/%3E%3Cpath d='M0 200 C60 140, 140 260, 200 200 S260 140, 400 200 S540 260, 600 200 S660 140, 800 200' stroke='rgba(46, 204, 113, 0.3)' fill='none' stroke-width='2'/%3E%3Cpath d='M0 200 C160 100, 240 300, 400 200 S560 100, 800 200' stroke='rgba(231, 76, 60, 0.2)' fill='none' stroke-width='1'/%3E%3Cpath d='M0 200 C200 140, 400 260, 600 200 S800 140, 1000 200' stroke='rgba(46, 204, 113, 0.2)' fill='none' stroke-width='1'/%3E%3Cpath d='M-100 200 Q0 100, 100 200 T300 200 T500 200 T700 200 T900 200' stroke='rgba(231, 76, 60, 0.1)' fill='none' stroke-width='1'/%3E%3Cpath d='M-100 200 Q0 300, 100 200 T300 200 T500 200 T700 200 T900 200' stroke='rgba(46, 204, 113, 0.1)' fill='none' stroke-width='1'/%3E%3C/svg%3E");
            background-size: cover;
            background-position: center;
            background-repeat: repeat-x;
            opacity: 0.7;
            z-index: 1;
        }

        .header-content {
            position: relative;
            z-index: 2;
        }

        .header-section h1 {
            color: var(--secondary-color);
            font-size: 2.5em;
            margin-bottom: 20px;
            font-weight: 600;
        }

        .header-section p {
            color: var(--text-color);
            font-size: 1em;
            max-width: 800px;
            margin: 0 auto 15px;
        }

        .creator-info {
            font-size: 0.9em;
            margin-top: 20px;
            color: var(--light-text-color);
        }

        .creator-info a {
            color: var(--primary-color);
            text-decoration: none;
            font-weight: 600;
        }

        h2 {
            color: var(--secondary-color);
            margin-top: 0;
            font-size: 1.8em;
            font-weight: 600;
        }

        .etf-card {
            background-color: var(--card-background);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            border: 1px solid var(--border-color);
        }

        .etf-info {
            margin-bottom: 25px;
            font-size: 1em;
            color: var(--light-text-color);
        }

        .plot-cards {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
        }

        .plot-card {
            background-color: var(--card-background);
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            display: flex;
            flex-direction: column;
            border: 1px solid var(--border-color);
        }

        .plot-title {
            font-size: 1.1em;
            font-weight: 600;
            margin-bottom: 15px;
            color: var(--secondary-color);
        }

        .plot-info {
            margin-top: 15px;
            font-size: 0.9em;
            color: var(--light-text-color);
        }

        .prediction-info {
            margin-top: 15px;
            font-weight: 600;
            font-size: 0.9em;
            padding: 6px 12px;
            border-radius: 4px;
            display: inline-block;
        }

        .above { background-color: #e74c3c; color: white; }
        .below { background-color: #2ecc71; color: white; }

        .cagr-info {
            margin-top: 15px;
            display: flex;
            align-items: center;
        }

        .cagr-label {
            font-size: 0.9em;
            color: var(--light-text-color);
            margin-right: 8px;
        }

        .cagr-value {
            font-size: 1em;
            font-weight: 600;
            color: var(--secondary-color);
        }

        img {
            width: 100%;
            height: auto;
            border-radius: 6px;
            margin-bottom: 15px;
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
            body {
                padding: 20px;
            }
            .header-section {
                padding: 40px 20px;
            }
        }
    """)

    results = [
        style,
        Title("ETF Analysis: Your Financial Compass"),
        Div(
            Div(
                H1("ETF Analysis: Your Financial Compass"),
                P("Explore the Australian ETF market with our unique perspective, combining mean reversion principles and linear modeling."),
                P("Remember, this tool provides insights, not financial advice. Use it wisely in your investment journey."),
                Div(
                    Span("Developed by "),
                    A("Alec Sharp", href="https://www.alecsharpie.me/", target="_blank"),
                    cls="creator-info"
                ),
                cls="header-content"
            ),
            cls="header-section"
        ),
    ]

    for data in ticker_data:
        ticker = data['ticker']
        etf_info = ETF_INFO[ticker]
        plot_cards = []

        for period, info in data['plots_and_cagr'].items():
            cagr_info = ""
            if info['cagr_data']:
                cagr_info = Div(
                    Span("CAGR:", cls="cagr-label"),
                    Span(f"{info['cagr_data']['cagr']}%", cls="cagr-value"),
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
