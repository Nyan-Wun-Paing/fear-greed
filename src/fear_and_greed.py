import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from io import BytesIO
import matplotlib.dates as mdates
import os
from dotenv import load_dotenv

"""
Loads Telegram Bot Token and Chat ID from environment variables or .env file.
Raises KeyError if credentials are not found.
"""
# Load .env file (if it exists)
load_dotenv()

try:
    # First, check for the key in environment variables (e.g., from GitHub Actions)
    TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
    TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
    if TELEGRAM_BOT_TOKEN == "7898328289:AAGJF0EUAxizLb9I19QFOmXK8c0TM2rlnqI":
        print("True")
    if TELEGRAM_CHAT_ID == "-4520793526":
        print("2True")
    
except KeyError:
    # If not found, try to load it from the .env file
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', None)
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', None)

    if TELEGRAM_CHAT_ID and TELEGRAM_BOT_TOKEN:
        print("Telegram Bot Key found in .env file.")
    else:
        # Raise an error if the key is not found in both places
        raise KeyError("Error: Telegram Bot Key not found in environment variables or .env file.")

"""
    Fetches the historical Fear and Greed Index data from CNN.
    Returns:
        pd.DataFrame: DataFrame containing the timestamp and value columns for Fear and Greed Index.
    Raises:
        Exception: If data retrieval fails or no historical data is found.
"""

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

        df['timestamp'] = pd.to_datetime(df['x'], unit='ms')
        df['value'] = df['y']
        
        return df
    else:
        raise Exception("No historical data found in the response")

"""
    Sends a text message via Telegram.
    Args:
        bot_token (str): Telegram Bot Token.
        chat_id (str): Telegram Chat ID.
        message (str): Text message to send.
"""
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, json=payload)

"""
    Sends an image (plot) via Telegram.
    Args:
        bot_token (str): Telegram Bot Token.
        chat_id (str): Telegram Chat ID.
        image_bytes (BytesIO): Image data in byte format.
"""

def send_telegram_image(image_bytes):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {'photo': ('chart.png', image_bytes, 'image/png')}
    data = {'chat_id': TELEGRAM_CHAT_ID}
    requests.post(url, files=files, data=data)

"""
    Calculates Exponential Moving Average (EMA) for the given DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing the 'value' column.
        span (int): Span for EMA calculation.
    Returns:
        pd.Series: Series containing EMA values.
"""
def calculate_ema(df, span):
    return df['value'].ewm(span=span, adjust=False).mean()

"""
    Calculates Bollinger Bands for the given DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing the 'value' column.
        window (int): Window size for SMA and standard deviation.
        num_std (int): Number of standard deviations for upper/lower bands.
    Returns:
        pd.DataFrame: DataFrame with additional columns for SMA, upper_band, and lower_band.
"""
def calculate_bollinger_bands(df, window=20, num_std=2):
    df['sma'] = df['value'].rolling(window=window).mean()
    df['std'] = df['value'].rolling(window=window).std()
    df['upper_band'] = df['sma'] + (num_std * df['std'])
    df['lower_band'] = df['sma'] - (num_std * df['std'])
    return df

"""
    Calculates Relative Strength Index (RSI) for the given DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing the 'value' column.
        period (int): Number of periods for RSI calculation.
    Returns:
        pd.DataFrame: DataFrame with an additional column for RSI.
"""
def calculate_rsi(df, period=14):
    delta = df['value'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

"""
    Calculates MACD and MACD Signal Line for the given DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing the 'value' column.
        span_short (int): Short-term EMA span for MACD calculation.
        span_long (int): Long-term EMA span for MACD calculation.
        span_signal (int): Signal line EMA span.
    Returns:
        pd.DataFrame: DataFrame with additional columns for MACD and MACD Signal Line.
"""
def calculate_macd(df, span_short=12, span_long=26, span_signal=9):
    df['ema_short'] = df['value'].ewm(span=span_short, adjust=False).mean()
    df['ema_long'] = df['value'].ewm(span=span_long, adjust=False).mean()
    df['macd'] = df['ema_short'] - df['ema_long']
    df['macd_signal'] = df['macd'].ewm(span=span_signal, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    return df


def calculate_dynamic_thresholds(df, base_bull_threshold=60, base_bear_threshold=40):
    market_volatility = df['std'].rolling(window=20).mean()
    df['rsi_bull_threshold'] = base_bull_threshold - (market_volatility * 0.5)
    df['rsi_bear_threshold'] = base_bear_threshold + (market_volatility * 0.5)
    return df

"""
    Checks for higher low and lower high patterns.
    Returns (higher_low, lower_high) as a tuple.
"""
def detect_higher_lows_lower_highs(df, i):
    higher_low = lower_high = False
    
    # Ensure there's enough data to check patterns
    if i >= 2:
        # Check for higher low pattern (for bullish context)
        if df['value'][i] > df['value'][i-1] and df['value'][i-1] > df['value'][i-2]:
            higher_low = True
        # Check for lower high pattern (for bearish context)
        if df['value'][i] < df['value'][i-1] and df['value'][i-1] < df['value'][i-2]:
            lower_high = True
    
    return (higher_low, lower_high)  # Return as tuple

"""
    Detect bullish and bearish divergences in the RSI.

    Parameters:
    df (DataFrame): The dataframe containing the 'value' and 'rsi' columns.
    rsi_bull_threshold (int): RSI value above which bullish divergence may occur.
    rsi_bear_threshold (int): RSI value below which bearish divergence may occur.

    Returns:
    list: List of divergences in the format (timestamp, type, value, confidence).
"""
def detect_divergences(df, rsi_bull_threshold=60, rsi_bear_threshold=40, window=10):
    divergences = []

    for i in range(1, len(df)):
        # Calculate confidence level for the current index
        confidence = calculate_divergence_confidence(df, i, rsi_bull_threshold, rsi_bear_threshold, window)
        
        # Detect bullish divergence
        if (
            df['rsi'][i] < rsi_bear_threshold and               # RSI below bear threshold
            df['rsi'][i] > df['rsi'][i-1]                       # RSI going up
        ):
            # Append with confidence percentage only if it meets a minimum threshold (e.g., 50%)
            if confidence >= 66:  # Considered only if confidence is above 50%
                divergences.append((df['timestamp'][i], 'Bullish Divergence', df['value'][i], confidence))

        # Detect bearish divergence
        elif (
            df['rsi'][i] > rsi_bull_threshold and               # RSI above bull threshold
            df['rsi'][i] < df['rsi'][i-1]                       # RSI going down
        ):
            # Append with confidence percentage only if it meets a minimum threshold (e.g., 50%)
            if confidence >= 53:  # Considered only if confidence is above 50%
                divergences.append((df['timestamp'][i], 'Bearish Divergence', df['value'][i], confidence))
    
    return divergences

"""
    Calculates the confidence score for a divergence pattern between price and RSI.
    The confidence score is based on three factors: price movement, RSI thresholds, 
    and historical price significance (via Z-score).
    
    Arguments:
    df -- The dataframe containing the price ('value') and RSI ('rsi') data.
    i -- The current index in the dataframe for which confidence is calculated.
    rsi_bull_threshold -- The RSI threshold above which the market is considered bullish.
    rsi_bear_threshold -- The RSI threshold below which the market is considered bearish.
    window -- The window size used for calculating the historical price confidence (Z-score).

    Returns:
    confidence_percentage -- The final confidence score as a percentage (0-100) based on the 
                              magnitude of price movement, RSI deviation, and historical price 
                              confidence.
"""
def calculate_divergence_confidence(df, i, rsi_bull_threshold, rsi_bear_threshold, window):
    # Price movement and RSI differences
    price_diff = abs(df['value'][i] - df['value'][i-1])
    rsi_diff = abs(df['rsi'][i] - df['rsi'][i-1])
    magnitude_confidence = price_diff / (rsi_diff if rsi_diff != 0 else 1)
    
    # Adjust confidence based on RSI thresholds
    if df['rsi'][i] < rsi_bear_threshold:
        rsi_adjustment = rsi_bear_threshold - df['rsi'][i]
    else:
        rsi_adjustment = df['rsi'][i] - rsi_bull_threshold

    # Calculate price-based confidence (Z-score approach)
    price_confidence = calculate_price_confidence(df, i, window)

    # Combine all factors to calculate final confidence
    total_confidence = (
        1/3 * magnitude_confidence +  # Prioritize price magnitude
        1/3 * rsi_adjustment +       # RSI adjustment
        1/3 * price_confidence       # Price confidence
    )

    # Final confidence score as a percentage
    confidence_percentage = min(max((total_confidence / 10) * 100, 0), 100)
    
    return confidence_percentage

"""
    Calculate the price confidence based on a Z-score over a window.
    This method evaluates how extreme the current price is relative to the historical values.
"""
def calculate_price_confidence(df, i, window):
    # Ensure there's enough data to calculate a window-based Z-score
    if i >= window:
        # Historical price values for the last `window` days
        recent_prices = df['value'][i-window:i]

        # Calculate mean and standard deviation of prices in the window
        mean_price = recent_prices.mean()
        std_dev = recent_prices.std()

        # Avoid division by zero if std_dev is very small
        if std_dev == 0:
            return 1  # High confidence if there's no variability (flat price)

        # Calculate Z-score for the current price
        z_score = (df['value'][i] - mean_price) / std_dev

        # Return the absolute Z-score as the confidence (higher Z-score = more confidence)
        return abs(z_score)
    else:
        return 0  # Return 0 if there's insufficient data to calculate price confidence

"""
    Filters and returns divergences that occurred within the last 5 days.
"""
def divergence_within_5_days(divergences):
    today = dt.datetime.now()
    recent_divergences = [(d[0], d[1], d[3]) for d in divergences if (today - pd.to_datetime(d[0])).days <= 5]
    return recent_divergences


"""
    Plot the Fear and Greed Index with Bollinger Bands, MACD, and RSI.

    Parameters:
    df (DataFrame): The dataframe containing the historical data.
    divergences (list): List of detected divergences to annotate on the chart.

    Returns:
    BytesIO: The chart image in memory for sending via Telegram.
"""
def plot_chart(df, divergences):
    plt.figure(figsize=(12, 8))
    
    recent_bullish = any(d[1] == 'Bullish Divergence' and (dt.datetime.now() - pd.to_datetime(d[0])).days <= 5 for d in divergences)
    recent_bearish = any(d[1] == 'Bearish Divergence' and (dt.datetime.now() - pd.to_datetime(d[0])).days <= 5 for d in divergences)
    line_color = 'darkseagreen' if recent_bullish else 'salmon' if recent_bearish else 'black'    # Plot Fear and Greed Index line
    plt.plot(df['timestamp'], df['y'], color=line_color)

    # Plot Bollinger Bands
    plt.fill_between(df['timestamp'], df['upper_band'], df['lower_band'], color='gray', alpha=0.2)

    for timestamp, label, value, confidence in divergences:
        color = 'forestgreen' if 'Bullish' in label else 'firebrick'
        plt.scatter(timestamp, value, color=color, s=50, label=label)
        plt.text(timestamp, value, f"{label}\nConf: {int(confidence)}%", color=color, fontsize=8)

    # Plot MACD and MACD Signal Line
    plt.plot(df['timestamp'], df['macd'], color='green')
    plt.plot(df['timestamp'], df['macd_signal'], color='red')
    plt.axhline(0, color='black', linestyle='--')

    # Get the current value of the Fear and Greed Index as a whole number
    current_value = int(df['y'].iloc[-1])  # Round to whole number

    # Formatting chart with current date, time, and value in red
    current_date = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    plt.title(f'Fear and Greed Index: Current Value {current_value}\n{current_date}', color='red')
    plt.xlabel('Date')
    plt.ylabel('Fear and Greed Index')

    # Rotate date labels vertically
    plt.xticks(rotation=90)
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png', dpi=300)
    img_bytes.seek(0)
    return img_bytes

def main():
    df = fetch_fear_and_greed()

    df = calculate_bollinger_bands(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_dynamic_thresholds(df)

    divergences = detect_divergences(df)

    recent_divergences = divergence_within_5_days(divergences)
    if recent_divergences:
        divergence_msg = "Divergences within last 5 days:\n" + \
                         "\n".join([f"{d[1]} on {d[0].date()} - Confidence: {int(d[2])}%" for d in recent_divergences])
        send_telegram_message(divergence_msg)

    image_bytes = plot_chart(df, divergences)
    send_telegram_image(image_bytes)

if __name__ == "__main__":
    main()
