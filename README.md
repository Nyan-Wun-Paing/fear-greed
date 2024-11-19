# Fear and Greed Index Tracker

This project tracks CNN's Fear and Greed Index, applies technical indicators such as RSI, MACD, and Bollinger Bands, and detects divergences. The data is visualized and sent as a Telegram notification, complete with charts.

## Features
- Fetches Fear and Greed Index data from CNN.
- Calculates RSI, MACD, and Bollinger Bands.
- Detects bullish and bearish divergences.
- Sends Telegram messages with chart images.

## Setup
1. Clone the repository:
    ```
    git clone https://github.com/your-username/fear-greed-index-tracker.git
    cd fear-greed-index-tracker
    ```

2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Create a `.env` file for your Telegram bot credentials:
    ```
    TELEGRAM_BOT_TOKEN=your_bot_token
    TELEGRAM_CHAT_ID=your_chat_id
    ```

4. Run the application:
    ```
    python src\fear_and_greed.py
    ```


## To-Do
- [ ] Improve divergence confidence levels.
- [ ] Add additional technical indicators.
- [ ] Add backtesting with DCA
- [ ] Logging for better error catching.
