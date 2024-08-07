from fasthtml import FastHTML, Title, Main, Div, H1, P, Button, Img, Select, Option, Table, Tr, Th, Td
from fasthtml.common import *
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta
import io
import base64

app = FastHTML()

# ETF information dictionary
ETF_INFO = {
    'VAS.AX': {
        'name': 'Vanguard Australian Shares Index ETF',
        'description': "Provides exposure to Australia's 300 largest companies, aiming to track the performance of the S&P/ASX 300 index performance."
    },
    'VHY.AX': {
        'name': 'Vanguard Australian Shares High Yield ETF',
        'description': "Provides exposure to companies listed on the ASX that have high forecast dividends, aiming to track the performance of the FTSE ASFA Australia High Dividend Yield Index."
    },
    'VTS.AX': {
        'name': 'Vanguard US Total Market Shares Index',
        'description': "Provides exposure to 99.5% of the US stock market, aiming to track the performance of the CRSP US Total Market Index."
    },
    'VGS.AX': {
        'name': 'Vanguard MSCI Index International Shares ETF',
        'description': "Provides exposure to global developed markets, the largest country exposures are the US, Japan, UK, France and Canada, aiming to track the performance of the MSCI World ex-Australia Index (with net dividends reinvested)."
    },
    'VGE.AX': {
        'name': 'Vanguard FTSE Emerging Markets Shares ETF',
        'description': "Provides exposure to companies listed in emerging markets, aiming to track the performance of the FTSE Emerging Markets All Cap China A Inclusion Index, hedged into Australian dollars."
    },
    'VAF.AX': {
        'name': 'Vanguard Australian Fixed Interest Index',
        'description': "Provides exposure to investment-grade bonds issued in the Australian bond market, aiming to track the performance of the Bloomberg AusBond Composite 0+ Yr Index."
    }
}

TICKERS = list(ETF_INFO.keys())

# ... (keep the existing helper functions)

@app.get("/")
def home():
    return Title("ETF Analysis"), Main(
        H1("ETF Analysis"),
        P("Select ETFs to analyze and click the button to generate the analysis."),
        Select(*[Option(f"{ticker} - {ETF_INFO[ticker]['name']}", value=ticker) for ticker in TICKERS], name="tickers", multiple=True),
        Button("Generate Analysis", hx_post="/generate", hx_target="#results", hx_include="[name='tickers']"),
        Div(id="results")
    )

@app.post("/generate")
def generate(tickers: list):
    ticker_data = []
    for ticker in tickers:
        etf_data = yf.download(ticker, start='2003-01-01')
        etf_data['Date'] = etf_data.index
        etf_data['DateNumeric'] = etf_data['Date'].apply(lambda date: date.toordinal())

        models = {
            '20Y': fit_model(etf_data, get_past_date(20 * 365)),
            '3Y': fit_model(etf_data, get_past_date(3 * 365)),
            '1Y': fit_model(etf_data, get_past_date(365)),
        }

        latest_date = etf_data['DateNumeric'].iloc[-1]
        latest_price = etf_data['Close'].iloc[-1]

        model_20y, _ = models['20Y']
        predicted_price_20y = model_20y.predict([[latest_date]])[0]
        delta_20y = predicted_price_20y - latest_price

        ticker_data.append((ticker, etf_data, models, abs(delta_20y)))

    # Sort tickers based on absolute 20Y delta
    ticker_data.sort(key=lambda x: x[3], reverse=True)

    results = []
    for ticker, etf_data, models, _ in ticker_data:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{ticker} - {ETF_INFO[ticker]["name"]} Analysis', fontsize=16)

        latest_date = etf_data['DateNumeric'].iloc[-1]
        latest_price = etf_data['Close'].iloc[-1]

        ticker_results = [
            H2(f"{ticker} - {ETF_INFO[ticker]['name']}"),
            P(ETF_INFO[ticker]['description']),
            Table(
                Tr(Th("Timespan"), Th("Predicted Price"), Th("Actual Price"), Th("Recommendation"), Th("Delta"))
            )
        ]

        for i, (timespan, (model, data)) in enumerate(models.items()):
            input_data = pd.DataFrame([[latest_date]], columns=['DateNumeric'])
            predicted_price = model.predict(input_data)[0]

            axs[i].plot(data['Date'], data['Close'], label='Actual Price', color='black', linewidth=2)
            axs[i].plot(data['Date'], model.predict(data[['DateNumeric']]), label=f'{timespan} Prediction', linewidth=2)
            axs[i].set_title(timespan, fontsize=15)
            axs[i].set_ylabel('Price (AUD)', fontsize=10)
            axs[i].legend(fontsize=8, loc='upper left')

            delta = predicted_price - latest_price
            buy_sell = "Buy" if delta > 0 else "Sell"
            ticker_results.append(
                Tr(
                    Td(timespan),
                    Td(f"${predicted_price:.2f}"),
                    Td(f"${latest_price:.2f}"),
                    Td(buy_sell),
                    Td(f"${delta:.2f}")
                )
            )

        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        ticker_results.append(Img(src=f"data:image/png;base64,{image_base64}", alt=f"{ticker} Analysis"))
        results.extend(ticker_results)
        plt.close(fig)

    return Div(*results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
