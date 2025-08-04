# TickBot: NVDA Intraday Buy/Sell Signal Predictor

## 📌 Objective

TickBot is a real-time trading simulation tool designed to analyze 1-minute intraday NVDA stock data. It leverages a lightweight neural network to learn short-term market trends and issue Buy/Sell predictions at precise intervals.

---

## 🕒 Time Window

- **Market Hours Analyzed:** 9:30 AM – 10:30 AM (Eastern Time)
- **Data Interval:** 1-minute candles
- **Chart X-Axis:** Displays human-readable timestamps
- **Y-Axis:** NVDA Close Price

---

## 🧠 Prediction Strategy

### Rolling 4-Minute Learning Window
- At every **5th minute**, the model:
    - Looks back at the **previous 4 minutes** of price + volume data
    - Predicts whether to **Buy or Sell** at that 5th minute

### Marker Logic
- A **Buy Signal (▲ green)** is shown at time `t` if the model predicts a price increase based on `t-4` to `t-1` data
- A **Sell Signal (▼ red)** is shown at time `t` if the model predicts a price drop based on `t-4` to `t-1` data

---

## ✅ Prediction Display Rules

- **Predictions are only shown at every 5th minute** (e.g., 9:34, 9:39, etc.)
- No signal is shown at other times
- Each marker reflects the model’s decision based on learned short-term patterns

---

## 💡 Future Enhancements

- Track performance of predictions
- Add confidence thresholds to reduce false signals
- Extend to other tickers or longer trading sessions