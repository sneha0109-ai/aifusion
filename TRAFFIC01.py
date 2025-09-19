!pip install -q xgboost scikit-learn pandas matplotlib seaborn plotly folium pyngrok streamlit joblib kaggle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

model = joblib.load("traffic_xgb.pkl")

# Define features (must match training)
features = ["hour","dow","is_weekend","vehicles_lag_1","vehicles_roll_3"]

df = pd.read_csv("/content/Traffic[1].csv")  
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

print("Data shape:", df.shape)
print(df.head())
print(df.info())

print("Missing values per column:\n", df.isna().sum())
print("Unique segments:", df['segment_id'].nunique() if 'segment_id' in df.columns else "No segment_id column")

# Plot average speed per segment (if available)
if "segment_id" in df.columns and "speed" in df.columns:
    df.groupby("segment_id")["speed"].mean().plot(kind="bar", figsize=(10,4))
    plt.title("Avg Speed per Segment")
    plt.show()

    print("Unique values in 'Date' column:", df['Date'].unique())
print("Unique values in 'Time' column:", df['Time'].unique())

y_true = df_proc["vehicles_target"]
y_pred_baseline = df_proc["vehicles_lag_1"]  # persistence model
baseline_mae = mean_absolute_error(y_true, y_pred_baseline)
print("Baseline MAE:", baseline_mae)

features = ["hour","dow","is_weekend","vehicles_lag_1","vehicles_roll_3"]
df_proc = df_proc.sort_values("DateTime")

split = int(0.8 * len(df_proc))
X_train = df_proc[features].iloc[:split]
y_train = df_proc["vehicles_target"].iloc[:split]
X_test = df_proc[features].iloc[split:]
y_test = df_proc["vehicles_target"].iloc[split:]

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {"objective":"reg:squarederror", "eta":0.1, "max_depth":6}
model = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtest,"val")], early_stopping_rounds=20)

y_pred = model.predict(dtest)

def predict_vehicles(hour, dow, previous_vehicles_lag_1, previous_vehicles_roll_3):
    is_weekend = 1 if dow in [5,6] else 0
    row = {
        "hour": hour,
        "dow": dow,
        "is_weekend": is_weekend,
        "vehicles_lag_1": previous_vehicles_lag_1,
        "vehicles_roll_3": previous_vehicles_roll_3
    }
    dmatrix = xgb.DMatrix(pd.DataFrame([row], columns=features))
    pred = model.predict(dmatrix)[0]
    return pred

print("🚦 AI Traffic Assistant 🚦")
seg = input("Enter road/segment name (example: 'Palasia_Square'): ")
# Removed current_speed input as the model predicts vehicle count
current_volume = float(input("Enter current vehicle count (per 5 min): "))
hour = int(input("Enter current hour (0-23): "))
dow = int(input("Enter day of week (0=Mon, 6=Sun): "))

# Get previous vehicle counts for lag and rolling features
previous_vehicles_lag_1 = float(input("Enter vehicle count from 1 hour ago: "))
previous_vehicles_roll_3 = float(input("Enter average vehicle count over the last 3 hours: "))


# Predict vehicle count for the next hour
pred_vehicles = predict_vehicles(hour, dow, previous_vehicles_lag_1, previous_vehicles_roll_3)

# Advisory based on predicted vehicle count
if pred_vehicles > 100:
    advisory = "⚠️ Very high traffic volume expected!"
elif pred_vehicles > 50:
    advisory = "⚠️ High traffic volume expected. Plan extra time."
elif pred_vehicles > 20:
    advisory = "📈 Moderate traffic volume expected."
else:
    advisory = "✅ Low traffic volume expected."

print(f"\n📍 Segment: {seg}")
# Removed speed related outputs
print(f"Current Vehicle Count (per 5 min): {current_volume:.1f}")
print(f"Predicted Vehicle Count (next hour): {pred_vehicles:.1f}")
print(advisory)

# Bar plot for current and predicted vehicle counts
plt.figure(figsize=(6,4))
sns.barplot(x=["Current","Predicted"], y=[current_volume, pred_vehicles], palette="coolwarm")
plt.title(f"Traffic Vehicle Count on {seg}")
plt.ylabel("Vehicle Count")
plt.show()

# Sequential prediction and plot for the next 5 hours
future_preds_vehicles = []

# We need to keep track of the last 3 values to calculate the rolling average for future steps.
# Let's initialize a list with the last known values.
# We have current_volume (t), previous_vehicles_lag_1 (t-1), and previous_vehicles_roll_3 (avg of t, t-1, t-2).
# We can approximate t-2 from these: 3 * roll_3 = t + t-1 + t-2  => t-2 = 3 * roll_3 - t - t-1

try:
    t_minus_2 = 3 * previous_vehicles_roll_3 - current_volume - previous_vehicles_lag_1
    # Ensure t_minus_2 is not negative, as vehicle count cannot be negative
    t_minus_2 = max(0, t_minus_2)
    last_3_vehicles = [t_minus_2, previous_vehicles_lag_1, current_volume]
except:
    # If calculation fails or leads to unreasonable value, initialize with available data
    last_3_vehicles = [previous_vehicles_lag_1, current_volume, current_volume] # Fallback: assume last two are the same as current

# Make the first prediction using the provided inputs
# This is already done above (pred_vehicles), so we add it to the list
future_preds_vehicles.append(pred_vehicles)
last_3_vehicles.append(pred_vehicles) # Add the new prediction to the history

# Now make subsequent predictions
for i in range(1, 5):  # Predict for the next 4 intervals (t+2 to t+5)
    # Update hour and day of week for future steps
    next_hour = (hour + i) % 24

    # Calculate lag_1 and rolling_3 for the next prediction
    next_lag_1 = last_3_vehicles[-1] # The last prediction is the lag_1 for the next step
    next_roll_3 = pd.Series(last_3_vehicles[-3:]).mean() # Rolling average of the last 3 values in history

    temp_pred = predict_vehicles(next_hour, dow, next_lag_1, next_roll_3)
    future_preds_vehicles.append(temp_pred)
    last_3_vehicles.append(temp_pred) # Add the new prediction to the history
    last_3_vehicles.pop(0) # Remove the oldest value from history


plt.figure(figsize=(8,4))
# Plot current volume and the sequence of future predictions
plt.plot(range(6), [current_volume] + future_preds_vehicles, marker="o", linestyle="-", color="blue")
plt.xticks(range(6), ["Now","t+1h","t+2h","t+3h","t+4h","t+5h"]) # Updated labels for hourly predictions
plt.title(f"Predicted Traffic Vehicle Count Trend - {seg}") # Updated title
plt.ylabel("Vehicle Count") # Updated ylabel
plt.grid(True, alpha=0.3)
plt.show()

# Advisory message plot
plt.figure(figsize=(5,2))
plt.text(0.5,0.5, advisory, fontsize=14, ha="center", va="center",
         bbox=dict(facecolor="yellow" if "⚠️" in advisory else "lightgreen", alpha=0.6))
plt.axis("off")
plt.show()