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
# - Uses NVDA 1-minute data from 9:30am – 10:30am Eastern
# - Trains on rolling 4-minute windows flattened into one vector
# - Predicts Buy/Sell for the next minute
# - Only plots signals at decision points
# ----------------------------------------------------

# STEP 1: Download data
ticker = "NVDA"
print(f"Downloading {ticker} 1-minute data from Yahoo Finance...")
data = yf.download(ticker, start="2025-08-01", end="2025-08-02", interval="1m")
# I convert the timestamps to US/Eastern for correct market times
data.index = data.index.tz_convert("US/Eastern")

# STEP 2: Filter for market open hour & clean
start_time = "09:30:00"
end_time = "10:30:00"
data = data.between_time(start_time, end_time)
data = data.dropna()

# STEP 3: Prepare tensors
# I extract Close and Volume as my features, then convert to a FloatTensor
X_all = data[["Close", "Volume"]].values
X_all = torch.tensor(X_all, dtype=torch.float32)
close_prices = data["Close"].values
timestamps = data.index.tolist()

# STEP 4: Define model
# I updated the first layer to accept an 8-dim vector (4 mins × 2 features)
model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 2)   # outputs logits for Sell=0 or Buy=1
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# STEP 5: Train & predict in a sliding window
buy_indices = []
sell_indices = []

for i in range(4, len(X_all) - 1):
    # I take the past 4 minutes, flatten them into one vector, and add a batch dim
    X_train = X_all[i-4:i].flatten().unsqueeze(0)  # shape: (1, 8)

    # I label the example by checking if the next minute’s close is higher
    next_up = 1 if close_prices[i+1] > close_prices[i] else 0
    y_train = torch.tensor([next_up], dtype=torch.long)  # shape: (1,)

    # I train the model on this single example
    optimizer.zero_grad()
    logits = model(X_train)              # shape: (1, 2)
    loss = loss_fn(logits, y_train)      # matched batch sizes
    loss.backward()
    optimizer.step()

    # I prepare the same flattening on the latest minute for prediction
    X_pred = X_all[i-3:i+1].flatten().unsqueeze(0)
    with torch.no_grad():
        output = model(X_pred)
        pred = torch.argmax(output).item()

    # I record the prediction index for plotting
    if pred == 1:
        buy_indices.append(i)
    else:
        sell_indices.append(i)

# STEP 6: Plot results
plt.style.use('dark_background')
plt.figure(figsize=(12, 5))
ax = plt.gca()

# I plot the price line
ax.plot(timestamps, close_prices, label="Close Price", color="deepskyblue", linewidth=2)

# I scatter buy points as green triangles
for idx in buy_indices:
    ax.scatter(timestamps[idx], close_prices[idx], color="lime", marker="^",
               label="Buy Signal" if idx == buy_indices[0] else "")

# I scatter sell points as red triangles
for idx in sell_indices:
    ax.scatter(timestamps[idx], close_prices[idx], color="red", marker="v",
               label="Sell Signal" if idx == sell_indices[0] else "")

# I format the X-axis for HH:MM in Eastern time
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=ZoneInfo("US/Eastern")))
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))

# I format the Y-axis as dollars and cents
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.2f}"))

plt.title("TickBot: NVDA 1-min Price with Predictive Buy/Sell Signals")
plt.xlabel("Time (Eastern)")
plt.ylabel("Price (USD)")
plt.legend(loc="upper left", fontsize=9)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("Plot_Graph.png")
plt.show()