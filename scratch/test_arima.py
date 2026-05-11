import pandas as pd
import numpy as np
import pmdarima as pm
import traceback

# Create dummy data
train = pd.Series(np.random.randn(100).cumsum(), index=pd.date_range("2021-01-01", periods=100))
test = pd.Series(np.random.randn(10).cumsum(), index=pd.date_range("2021-04-11", periods=10))

print("Fitting model...")
model = pm.auto_arima(train, seasonal=False)
print(f"Best order: {model.order}")

print("\nTesting update loop...")
for i in range(len(test)):
    try:
        # Predict 1 step
        fc = model.predict(n_periods=1)
        print(f"Step {i} forecast: {fc[0]}")
        
        # Update
        val = test.iloc[i:i+1]
        print(f"Updating with: {val}")
        model.update(val)
    except Exception as e:
        print(f"FAILED at step {i}: {e}")
        traceback.print_exc()
        break
