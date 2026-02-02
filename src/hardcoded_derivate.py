import numpy as np
import pandas as pd
from data_processing.data_loader import load_data

X, U, time_step = load_data(
    "Aeroshield",
    0.01,
    True,
    [0, 1],
    plot=False,
    verbose=False
)

X_val, U_val, _ = load_data(
    "Aeroshield_val",
    0.01,
    True,
    [0, 1],
    plot=False,
    verbose=False
)

X_dot = np.gradient(X, time_step, axis=0)
X_dot_val = np.gradient(X_val, time_step, axis=0)

X_new = np.vstack([X, X_val])
X_dot_new = np.vstack([X_dot, X_dot_val])
U_new = np.vstack([U, U_val])

data = {
    "x": X_new.flatten(),
    "x_dot": X_dot_new.flatten(),
    "u": U_new.flatten()
}

file_path = "data/raw/Aeroshield_with_deriv.csv"
df = pd.DataFrame(data)  
df.to_csv(file_path, index=False)