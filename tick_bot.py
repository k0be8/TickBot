import yfinance as yf
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ----------------------------------------------------
# TICKBOT v2: Multi-Class Buy/Hold/Sell Signal Predictor
# ----------------------------------------------------
# - Pulls real 1-minute intraday data from Yahoo Finance
# - Uses Close and Volume as input features
# - Applies a configurable threshold to label future movement
# - Predicts: 0 = Sell, 1 = Hold, 2 = Buy
# ----------------------------------------------------

# STEP 1: Download intraday data
ticker = "QQQ"
data = yf.download(ticker, period="1d", interval="1m")
print(data.head())  # Sanity check: preview data

# STEP 2: Clean data
data = data.dropna()

# STEP 3: Define model aggressiveness (how small a move counts)
price_threshold = 0.01  # Set to 1 cent; adjust for more/less sensitivity

# STEP 4: Prepare features
X = data[["Close", "Volume"]].values  # Two features: Close price & Volume
X = torch.tensor(X, dtype=torch.float32)

# STEP 5: Generate labels (0 = Sell, 1 = Hold, 2 = Buy)
close_prices = data["Close"].values
y = []
for i in range(len(close_prices) - 2):
    change = close_prices[i + 2] - close_prices[i]
    if change > price_threshold:
        y.append(2)  # Buy
    elif change < -price_threshold:
        y.append(0)  # Sell
    else:
        y.append(1)  # Hold

# Convert to tensor (long for CrossEntropyLoss)
y = torch.tensor(y, dtype=torch.long)

# STEP 6: Align X to match y length
X = X[:len(y)]

# STEP 7: Define neural network (2 inputs → hidden → 3-class output)
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 3)  # 3 outputs: Sell, Hold, Buy
)

# STEP 8: Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# STEP 9: Train model
for epoch in range(100):
    optimizer.zero_grad()
    logits = model(X)  # Raw outputs (logits)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# STEP 10: Predict class labels
predicted_classes = torch.argmax(model(X), dim=1).numpy()

# STEP 11: Visualize predictions vs close price
plt.figure(figsize=(12, 5))
plt.plot(close_prices[:len(predicted_classes)], label="Close Price", color="blue")

# Overlay signals:
for i, label in enumerate(predicted_classes):
    if label == 2:  # Buy
        plt.scatter(i, close_prices[i], color="green", marker="^", label="Buy Signal" if i == 0 else "")
    elif label == 0:  # Sell
        plt.scatter(i, close_prices[i], color="red", marker="v", label="Sell Signal" if i == 0 else "")

plt.title("TickBot: QQQ 1-min Price with Predicted Buy/Sell Signals")
plt.xlabel("Time Index")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()