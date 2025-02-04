
import logging
import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from flask import Flask, render_template, request
from flask_mail import Mail, Message

# Set up logging and Flask app
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'kcpalmisano@gmail.com'
app.config['MAIL_PASSWORD'] = 'bbox yvcf tgtp oauu'  # Consider using environment variables instead for security
mail = Mail(app)

# Home route
@app.route('/')
def home():
    app.logger.debug("Rendering Home Page")
    return render_template('index.html')

# About route
@app.route('/about')
def about():
    app.logger.debug("Rendering About Page")
    return render_template('about.html')

# Contact route
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        # Send email
        msg = Message(f"New Message from {name}", sender=email, recipients=['your-email@gmail.com'])
        msg.body = f"Name: {name}\nEmail: {email}\nMessage: {message}"
        mail.send(msg)

        return f"""
        <html>
        <head>
            <title>Thank You</title>
            <link rel="stylesheet" href="/static/style.css">
        </head>
        <body>
            <h1>Thank You, {name}!</h1>
            <p>Your message has been sent successfully.</p>
            <a href="/" class="button">Return Home</a>
        </body>
        </html>
        """
    return render_template('contact.html')

# Portfolio route
@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')

# Stock Predictor functions
def get_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data['Close']

def forecast_stock(data, steps=90):
    model = auto_arima(data, seasonal=True, m=5, trace=True, error_action='ignore', suppress_warnings=True)
    forecast, conf_int = model.predict(n_periods=steps, return_conf_int=True)
    return forecast, conf_int

def plot_forecast(data, forecast, conf_int, steps, ticker):
    plt.figure(figsize=(14, 7))

    # Plot historical data
    plt.plot(data.index, data, label="Historical Data", color="blue")

    # Plot forecasted data
    forecast_index = pd.date_range(data.index[-1], periods=steps + 1, freq='B')[1:]
    plt.plot(forecast_index, forecast, label="Forecast", color="green")

    # Plot confidence intervals
    plt.fill_between(forecast_index, conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.3, label="Confidence Interval")

    # Add titles and labels
    plt.title(f"{ticker} Stock Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="upper left")
    plt.grid(True)

    # Save the plot as an image
    image_path = os.path.join('static', 'forecast_plot.png')
    plt.savefig(image_path)
    plt.close()
    return image_path

# Stock Predictor route
@app.route('/stock-predictor', methods=['GET', 'POST'])
def stock_predictor():
    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        steps = int(request.form['steps'])

        # Get stock data and generate forecast
        historical_data = get_stock_data(ticker)
        forecasted_data, confidence_intervals = forecast_stock(historical_data, steps=steps)
        image_path = plot_forecast(historical_data, forecasted_data, confidence_intervals, steps, ticker)

        return render_template('stock_result.html', image_path=image_path, ticker=ticker)

    return render_template('stock_form.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
