import yfinance as yf
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from zoneinfo import ZoneInfo

# ----------------------------------------------------
# TICKBOT v3.0: NVDA Intraday Prediction (5-Min Model)
# ----------------------------------------------------
# - Uses NVDA 1-minute data from 9:30am–10:30am (Eastern)
# - Trains on rolling 4-minute windows
# - Predicts Buy/Sell at every 5th minute
# - Only plots signals at decision points (every 5th step)
# ----------------------------------------------------

# STEP 1: Download 1-minute intraday data for NVDA
ticker = "NVDA"
print(f"Downloading {ticker} 1-minute data from Yahoo Finance...")
data = yf.download(ticker, start="2025-08-01", end="2025-08-02", interval="1m")
data.index = data.index.tz_convert("US/Eastern")  # ✅ Convert to Eastern Time

# Filter to market open hour: 9:30–10:30am Eastern
start_time = "09:30:00"
end_time = "10:30:00"
data = data.between_time(start_time, end_time)

# Drop missing rows
data = data.dropna()

# STEP 2: Prepare data as tensors
X_all = data[["Close", "Volume"]].values
X_all = torch.tensor(X_all, dtype=torch.float32)
close_prices = data["Close"].values
timestamps = data.index.tolist()

# STEP 3: Define model
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 2)  # Output: 2 classes (Sell=0, Buy=1)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# STEP 4: Rolling training + prediction every 5th minute
buy_indices = []
sell_indices = []

for i in range(4, len(X_all) - 1):
    X_train = X_all[i - 4:i]
    y_train = []

    # Label each of the 4 minutes for training BASELINE
    # for j in range(i - 4, i):
    #     delta = close_prices[j + 1] - close_prices[j]
    #     if delta > 0.30:
    #         y_train.append(1)  # Buy
    #     elif delta < -0.30:
    #         y_train.append(0)  # Sell
    #     else:
    #         continue

    # Option 1 – Baseline: Buy if +$0.30, Sell if -$0.30
    # ✅ Updated logic to trade smarter with trends
    for j in range(i - 4, i):
        delta = close_prices[j + 1] - close_prices[j]

        # -----------------------------
        # ✅ NEW BUY LOGIC (less greedy)
        # I only want to BUY if the last 3 steps are going up and today's change is strong
        # BUT also I want to avoid buying too high (price spike too far from average)
        # -----------------------------
        if j >= 3:
            avg_recent_price = sum(close_prices[j - 3:j + 1]) / 4  # Get 4-min average
            if (
                    close_prices[j - 2] < close_prices[j - 1] < close_prices[j] < close_prices[j + 1] and
                    delta > 0.30 and
                    close_prices[j + 1] < avg_recent_price * 1.01  # ✅ Buy only if it's not spiking too high
            ):
                y_train.append(1)  # Buy

        # -----------------------------
        # ✅ NEW SELL LOGIC (avoid early dumps)
        # I only want to SELL if price was climbing before, and now it's falling
        # AND only if the rise before was significant (> $0.40 in last 3 minutes)
        # -----------------------------
        if j >= 3:
            prior_rise = close_prices[j] - close_prices[j - 3]  # Total gain over last 3 mins
            if (
                    close_prices[j - 2] < close_prices[j - 1] < close_prices[j] and
                    close_prices[j] > close_prices[j + 1] and
                    delta < -0.30 and
                    prior_rise > 0.40  # ✅ Only sell if the price really climbed beforehand
            ):
                y_train.append(0)  # Sell

        if len(y_train) < 2:
            continue  # Skip if not enough signal variation

    X_train = X_train[:len(y_train)]
    y_train = torch.tensor(y_train, dtype=torch.long)

    # Train model briefly
    for epoch in range(10):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

    # Predict on current 5th minute
    X_pred = X_all[i].unsqueeze(0)
    with torch.no_grad():
        output = model(X_pred)
        pred = torch.argmax(output).item()

    if pred == 1:
        buy_indices.append(i)
    elif pred == 0:
        sell_indices.append(i)

# STEP 5: Plot results (Enhanced Visual Styling)
plt.style.use('dark_background')  # Optional sleek style (can also try 'seaborn-v0_8-darkgrid')

plt.figure(figsize=(12, 5))
ax = plt.gca()

# Plot main close price line
ax.plot(timestamps, close_prices, label="Close Price", color="deepskyblue", linewidth=2)

# Plot Buy signals
for idx in buy_indices:
    ax.scatter(timestamps[idx], close_prices[idx], color="lime", marker="^", label="Buy Signal" if idx == buy_indices[0] else "")

# Plot Sell signals
for idx in sell_indices:
    ax.scatter(timestamps[idx], close_prices[idx], color="red", marker="v", label="Sell Signal" if idx == sell_indices[0] else "")

# Format X-axis to only show HH:MM (drop the 01 date)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=ZoneInfo("US/Eastern")))

# Major ticks every 10 minutes
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))

# Format Y-axis as US currency with cents
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.2f}"))

# Optional tweaks for professional look
plt.title("TickBot: NVDA 1-min Price with Predictive Buy/Sell Signals")
plt.xlabel("Time (Eastern)")
plt.ylabel("Price (USD)")
plt.legend(loc="upper left", fontsize=9)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()
