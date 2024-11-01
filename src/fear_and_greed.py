import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from io import BytesIO
import matplotlib.dates as mdates

# Telegram configuration
TELEGRAM_BOT_TOKEN = "7898328289:AAGJF0EUAxizLb9I19QFOmXK8c0TM2rlnqI"
TELEGRAM_CHAT_ID = "-4520793526"
def fetch_fear_and_greed():
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/2022-01-01"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code}")

    data = response.json()
    if 'fear_and_greed_historical' in data and 'data' in data['fear_and_greed_historical']:
        historical_data = data['fear_and_greed_historical']['data']
        df = pd.DataFrame(historical_data)

        # Convert the timestamp and value columns appropriately
        df['timestamp'] = pd.to_datetime(df['x'], unit='ms')  # Ensure 'x' is the correct timestamp column name
        df['value'] = df['y']  # Make sure to adjust if 'y' is the correct column name for the index value
        
        return df
    else:
        raise Exception("No historical data found in the response")

# Function to send message via Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, json=payload)

# Function to send image via Telegram
def send_telegram_image(image_bytes):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {'photo': ('chart.png', image_bytes, 'image/png')}
    data = {'chat_id': TELEGRAM_CHAT_ID}
    requests.post(url, files=files, data=data)

# Function to calculate EMA
def calculate_ema(df, span):
    return df['y'].ewm(span=span, adjust=False).mean()

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(df, window=20, num_std=2):
    df['sma'] = df['y'].rolling(window=window).mean()
    df['std'] = df['y'].rolling(window=window).std()
    df['upper_band'] = df['sma'] + (num_std * df['std'])
    df['lower_band'] = df['sma'] - (num_std * df['std'])
    return df

# Function to calculate RSI
def calculate_rsi(df, period=14):
    delta = df['y'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

# Function to calculate MACD and Histogram
def calculate_macd(df, span_short=12, span_long=26, span_signal=9):
    df['ema_short'] = df['y'].ewm(span=span_short, adjust=False).mean()
    df['ema_long'] = df['y'].ewm(span=span_long, adjust=False).mean()
    df['macd'] = df['ema_short'] - df['ema_long']
    df['macd_signal'] = df['macd'].ewm(span=span_signal, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    return df

# Function to calculate divergence confidence
def calculate_divergence_confidence(df, i, rsi_bull_threshold=60, rsi_bear_threshold=40):
    price_diff = abs(df['y'][i] - df['y'][i-1])
    rsi_diff = abs(df['rsi'][i] - df['rsi'][i-1])
    magnitude_confidence = price_diff / (rsi_diff if rsi_diff != 0 else 1)  # Avoid division by zero

    # RSI Adjustment (proximity to thresholds)
    if df['rsi'][i] < rsi_bear_threshold:
        rsi_adjustment = rsi_bear_threshold - df['rsi'][i]
    elif df['rsi'][i] > rsi_bull_threshold:
        rsi_adjustment = df['rsi'][i] - rsi_bull_threshold
    else:
        rsi_adjustment = 0

    duration_confidence = 1  # Assume duration is 1 for now

    # Confidence score (adjust weights for factors)
    confidence_score = (0.5 * magnitude_confidence) + (0.3 * duration_confidence) + (0.2 * rsi_adjustment)
    
    # Convert the confidence score into a percentage (0 to 100%)
    confidence_percentage = min(max((confidence_score / 5) * 100, 0), 100)  # Scale score, cap at 100%
    
    return confidence_percentage

# Detect divergences based on RSI
def detect_divergences(df):
    divergences = []
    for i in range(1, len(df)):
        if df['rsi'][i] < 40 and df['y'][i] > df['y'][i-1] and df['rsi'][i] > df['rsi'][i-1]:
            confidence = calculate_divergence_confidence(df, i)
            divergences.append((df['timestamp'][i], 'Bullish Divergence', df['y'][i], confidence))
        elif df['rsi'][i] > 60 and df['y'][i] < df['y'][i-1] and df['rsi'][i] < df['rsi'][i-1]:
            confidence = calculate_divergence_confidence(df, i)
            divergences.append((df['timestamp'][i], 'Bearish Divergence', df['y'][i], confidence))
    return divergences

# Function to check if divergence is within the last 5 days
def divergence_within_5_days(divergences):
    today = dt.datetime.now()
    return any((today - pd.to_datetime(divergence[0])).days <= 5 for divergence in divergences)

# Detect divergences based on RSI with Fear and Greed thresholds
def detect_divergences(df):
    divergences = []
    for i in range(1, len(df)):
        # Bullish Divergence when Fear and Greed value is below 40
        if df['rsi'][i] < 40 and df['y'][i] > df['y'][i-1] and df['rsi'][i] > df['rsi'][i-1] and df['y'][i] < 40:
            confidence = calculate_divergence_confidence(df, i)
            divergences.append((df['timestamp'][i], 'Bullish Divergence', df['y'][i], confidence))
        # Bearish Divergence when Fear and Greed value is above 60
        elif df['rsi'][i] > 60 and df['y'][i] < df['y'][i-1] and df['rsi'][i] < df['rsi'][i-1] and df['y'][i] > 60:
            confidence = calculate_divergence_confidence(df, i)
            divergences.append((df['timestamp'][i], 'Bearish Divergence', df['y'][i], confidence))
    return divergences

# Updated function to plot the chart with blue-colored, whole-number confidence level annotations
def plot_chart(df, divergences):
    plt.figure(figsize=(12, 8))

    # Determine line color based on recent divergences
    recent_bullish = any(d[1] == 'Bullish Divergence' and (dt.datetime.now() - pd.to_datetime(d[0])).days <= 5 for d in divergences)
    recent_bearish = any(d[1] == 'Bearish Divergence' and (dt.datetime.now() - pd.to_datetime(d[0])).days <= 5 for d in divergences)

    # Set line color based on the latest divergence within 5 days
    line_color = 'green' if recent_bullish else 'red' if recent_bearish else 'gray'

    # Plot Fear and Greed Index line
    plt.plot(df['timestamp'], df['y'], color=line_color)

    # Plot Bollinger Bands
    plt.fill_between(df['timestamp'], df['upper_band'], df['lower_band'], color='gray', alpha=0.2)

    # Plot MACD and MACD Signal Line
    plt.plot(df['timestamp'], df['macd'], color='green')
    plt.plot(df['timestamp'], df['macd_signal'], color='red')

    # Add MACD Zero Line
    plt.axhline(0, color='black', linestyle='--')

    # Mark divergences with blue, whole-number confidence level annotations
    for divergence in divergences:
        plt.scatter(divergence[0], divergence[2], 
                    color='green' if divergence[1].startswith('Bullish') else 'red', s=50)
        
        # Annotate only if the divergence is within the last 5 days
        days_since_divergence = (dt.datetime.now() - pd.to_datetime(divergence[0])).days
        if days_since_divergence <= 5:
            # Format confidence level as a whole number in blue color
            confidence_text = f"{divergence[1]} ({int(divergence[3])}%)"
            plt.annotate(confidence_text,
                         xy=(divergence[0], divergence[2] * 1.05),  # Position above the point
                         ha='center', fontsize=8, color='blue', weight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='black'))

    # Get the current value of the Fear and Greed Index as a whole number
    current_value = int(df['y'].iloc[-1])  # Round to whole number

    # Formatting chart with current date, time, and value in red
    current_date = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    plt.title(f'Fear and Greed Index: Current Value {current_value}\n{current_date}', color='red')
    plt.xlabel('Date')
    plt.ylabel('Fear and Greed Index')

    # Rotate date labels vertically
    plt.xticks(rotation=90)

    # Save chart to BytesIO to send via Telegram
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    #plt.show()
    buf.seek(0)
    return buf

# Main function to execute the process
def main():
    # Fetch Fear and Greed Index data
    data = fetch_fear_and_greed()

    # Convert data to DataFrame
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['x'], unit='ms')

    # Calculate technical indicators
    df['ema_9'] = calculate_ema(df, span=9)
    df['ema_26'] = calculate_ema(df, span=26)
    df = calculate_bollinger_bands(df)
    df = calculate_rsi(df, period=14)
    df = calculate_macd(df)

    # Detect divergences
    divergences = detect_divergences(df)

    # Send detection message if divergences are within the last 5 days
    if divergence_within_5_days(divergences):
        message = "Divergence detected in the last 5 days:\n"
        for divergence in divergences:
            message += f"{divergence[1]} on {divergence[0].strftime('%Y-%m-%d')} with Confidence: {divergence[3]:.2f}%\n"
        send_telegram_message(message)
    
    # Plot the chart and send it via Telegram
    chart_image = plot_chart(df, divergences)
    send_telegram_image(chart_image)

if __name__ == "__main__":
    main()