import numpy as np
import pandas as pd
import pysindy as ps
import SINDYcLib as lib
import matplotlib.pyplot as plt

data_csv = pd.read_csv("Simulacia.csv")
data = data_csv.to_numpy()

time_step = np.mean(np.diff(data[:, 0]))

# Priprava dat na trenovanie modelu
X = np.stack((data[:, 1], data[:, 2], data[:, 3]), axis=-1)
U = np.stack((data[:, 4]), axis=-1)
X_train, X_valid, X_test = lib.split_data(X, val_size=0.2, test_size=0.2)
U_train, U_valid, U_test = lib.split_data(U, val_size=0.2, test_size=0.2)

optimizers = lib.build_SR3_optimizers(regularizers=["L0", "L1"])
differentiation_methods = lib.build_differentiation_methods(X_train.shape[0], base_windows=[51, 71, 91])

best_config, pareto_front = lib.find_optimal_parameters(
    
    x_train=X_train, x_valid=X_valid, u_train=U_train, u_valid=U_valid, dt=time_step,
    optimizers=optimizers,
    differentiation_methods=differentiation_methods
)

errs = np.array([r["rmse"] for r in pareto_front], dtype=float)  
spars = np.array([r["sparsity"] for r in pareto_front], dtype=float)  


plt.figure(figsize=(6, 4))  
plt.scatter(errs, spars, color="tab:blue", label="Pareto body")  
plt.xlabel("RMSE")  
plt.ylabel("Sparsity (počet nenulových koeficientov)")  
plt.title("Pareto front")  
plt.grid(True, alpha=0.3)  
plt.legend()  
plt.tight_layout()  
plt.show()
 

model = ps.SINDy(  
    optimizer=best_config["optimizer"],  
    feature_library=best_config["library"],  
    differentiation_method=best_config["differentiation_method"]  
)  

model.fit(x=X_train, u=U_train, t=time_step)
model.print()
print(model.score(x=X_test, u=U_test, t=time_step))