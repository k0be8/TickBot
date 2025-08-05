# 📊 My First AI Trading Simulation - "TickBot" 🚀

## What I Built
I just completed my first end-to-end AI trading simulation project! This was my very first attempt at building a predictive trading system using PyTorch and real stock data.

## The Journey
As someone new to both machine learning and trading, I wanted to create something that would help me understand how AI might be used in financial markets. This project was my baby step into the world of algorithmic trading.

## What It Does
I built **TickBot** – a simple but functional trading simulation that:

- Downloads real 1-minute NVDA stock data from Yahoo Finance
- Focuses on a specific trading window (9:30am–10:30am Eastern time)
- Uses a neural network to predict whether the next minute will go up (Buy) or down (Sell)
- Makes predictions every minute using a rolling 4-minute historical window
- **Retrains the model from scratch each run using only that day’s data**  <!-- 🔧 NEW -->

## The Tech Stack

- **PyTorch** – for building and training the neural network
- **Yahoo Finance API** – to pull real stock data
- **Matplotlib** – to visualize predictions on price charts
- **Python** – my first language for machine learning projects

## How It Works (in Simple Terms)

1. I collect 4 consecutive minutes of stock price + volume data
2. The neural network learns whether the next minute’s price will go up or down
3. If it predicts “go up,” I show a green triangle (Buy)
4. If it predicts “go down,” I show a red triangle (Sell)
5. **Each run is non-deterministic, meaning the plot will change every time unless I add a fixed random seed** <!-- 🔧 NEW -->

## The Result

The model scanned NVDA's 1-minute price chart for a 1-hour period and produced buy/sell predictions.  
📈 In one test, **TickBot made a simulated $2.20 gain per share** by following its signals.

> ⚠️ **Note:** This result varies on each run because the model retrains from scratch without saving state or seeding randomness. <!-- 🔧 NEW -->

## What I Learned

- How to structure ML projects from data collection to visualization
- Time zone handling and filtering real-world market data
- How neural networks can be trained to make predictions on time-series data
- The challenge of working with real, noisy financial data
- **How randomness in model initialization affects prediction reproducibility** <!-- 🔧 NEW -->

## What’s Next?

- Add more input features like technical indicators and moving averages
- Expand testing to more stocks beyond NVDA
- Implement real trading logic with position sizing and stop-loss
- Add backtesting and performance evaluation
- **Add fixed random seed and model checkpointing for consistent results** <!-- 🔧 NEW -->
- **Filter to just the first 5–15 minutes of open market trading for sharper signal testing** <!-- 🔧 NEW -->

---

> This project represents my **first real step into the world of AI-powered trading**, and I'm excited to keep building on it!

---

#### Tags

`#python` `#pytorch` `#machinelearning` `#trading` `#ai` `#dataanalysis` `#quantitativefinance` `#beginnerproject`
