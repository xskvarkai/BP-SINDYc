import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import json
from pathlib import Path
from typing import Tuple, Union, Optional
import warnings
from utils.config_manager import ConfigManager
from utils.plots import plot_trajectory, plot_koopman_spectrum
from utils.helpers import compute_time_vector
import matplotlib.pyplot as plt

class KoopmanNeural(nn.Module):
    """
    A class for modeling dynamical systems using the Koopman operator framework
    with a neural network-based encoder/decoder architecture.
    Includes separate encoders for state variables and control inputs,
    lifting both into a shared latent Koopman space.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        X_train: np.ndarray,
        U_train: np.ndarray,
        X_test: np.ndarray,
        U_test: np.ndarray,
        dt: Union[int, float],
        lifted_dim: int = 32,
        hidden_dim: int = 128,
        num_batches: int = 10,
        rollout_steps: int = 5,
        lr: float = 5e-4,
        device: str = 'cpu',
    ):
        super(KoopmanNeural, self).__init__()

        self.scaler_X = MinMaxScaler().fit(X_train)  # Scaler for state variables
        self.scaler_U = MinMaxScaler().fit(U_train)  # Scaler for control inputs

        self.config_manager = config_manager
        self.config_manager.load_config("settings")
        self.data_export_path = Path(self.config_manager.get_path("settings.paths.data_export_dir"))

        self.state_dim     = X_train.shape[1]
        self.control_dim   = U_train.shape[1]
        self.lifted_dim    = lifted_dim
        self.device        = torch.device(device)
        self.rollout_steps = rollout_steps

        self.history = {
            "loss": [],
            "loss_rec": [],
            "loss_lin": [],
            "loss_pred": [],
            "loss_multi": [],
            "lr": []
        }

        self.data = {  # Store scaled data
            "x_train": self.scaler_X.transform(X_train),
            "u_train": self.scaler_U.transform(U_train),
            "x_ref":   self.scaler_X.transform(X_test),
            "u_ref":   self.scaler_U.transform(U_test),
            "dt":      dt
        }

        X_t = torch.tensor(self.data.get("x_train"), dtype=torch.float32)
        U_t = torch.tensor(self.data.get("u_train"), dtype=torch.float32)
        
        # Create sliding window sequences for correct multi-step rollout
        seq_len     = self.rollout_steps + 1
        num_samples = X_t.shape[0] - seq_len
        
        if num_samples > 0:
            X_seq = torch.stack([X_t[i : i + seq_len] for i in range(num_samples)]).to(self.device)
            U_seq = torch.stack([U_t[i : i + seq_len] for i in range(num_samples)]).to(self.device)

            batch_size      = max(1, num_samples // num_batches)
            dataset         = TensorDataset(X_seq, U_seq)
            self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            raise ValueError("Training data is too short for the specified rollout_steps.")

        self.oscillation_report = self._detect_oscillations(X_train, dt)

        self.encoder = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, lifted_dim)
        )

        self.encoder_u = nn.Sequential(
            nn.Linear(self.control_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.control_dim) 
        )

        self.decoder = nn.Sequential(
            nn.Linear(lifted_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.state_dim)
        )

        self.A = nn.Linear(lifted_dim, lifted_dim, bias=False)  # State transition matrix
        self.B = nn.Linear(self.control_dim, lifted_dim, bias=False) # Control influence matrix
        self._initialize_A(self.oscillation_report)

        self.to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def _detect_oscillations(self, X_train: np.ndarray, dt: float) -> dict:
        fs            = 1.0 / dt
        n_samples     = X_train.shape[0]
        freqs         = np.fft.rfftfreq(n_samples, d=dt)
        channels      = []
        any_oscillatory = False

        for ch in range(X_train.shape[1]):
            signal = X_train[:, ch]
            signal = signal - signal.mean()

            fft_mag    = np.abs(np.fft.rfft(signal))
            fft_mag[0] = 0
            dominant_freq = freqs[np.argmax(fft_mag)]
            dominant_amp  = fft_mag.max() / (n_samples / 2)

            zero_crossings = np.where(np.diff(np.sign(signal)))[0]
            zcr = len(zero_crossings) / (n_samples * dt)

            signal_std     = signal.std() + 1e-12
            is_oscillatory = (dominant_freq > 0.01) and (dominant_amp > 0.05 * signal_std)

            if is_oscillatory:
                any_oscillatory = True

            channels.append({
                "channel":        ch,
                "dominant_freq":  float(dominant_freq),
                "dominant_amp":   float(dominant_amp),
                "zcr":            float(zcr),
                "is_oscillatory": bool(is_oscillatory),
                "omega_discrete": float(2 * np.pi * dominant_freq * dt)
            })

        return {
            "channels":        channels,
            "any_oscillatory": any_oscillatory,
            "fs":              fs,
            "dt":              dt,
            "n_samples":       n_samples
        }

    def _initialize_A(self, report: dict):
        dim    = self.lifted_dim
        A_init = torch.zeros(dim, dim)

        if report["any_oscillatory"]:
            omegas = [
                ch["omega_discrete"]
                for ch in report["channels"]
                if ch["is_oscillatory"] and ch["omega_discrete"] > 1e-6
            ]

            block_idx = 0
            freq_idx  = 0

            while block_idx + 1 < dim:
                if freq_idx < len(omegas):
                    omega = omegas[freq_idx % len(omegas)]
                    freq_idx += 1
                else:
                    omega = np.random.uniform(0.05, 0.3)

                r = np.random.uniform(0.95, 0.99)
                c, s = r * np.cos(omega), r * np.sin(omega)

                A_init[block_idx,     block_idx]     =  c
                A_init[block_idx,     block_idx + 1] = -s
                A_init[block_idx + 1, block_idx]     =  s
                A_init[block_idx + 1, block_idx + 1] =  c
                block_idx += 2

            if block_idx < dim:
                A_init[block_idx, block_idx] = 0.97

        else:
            diag_vals = torch.ones(dim) * np.random.uniform(0.95, 0.99)
            A_init    = torch.diag(diag_vals)

        with torch.no_grad():
            self.A.weight.copy_(A_init)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode_u(self, u: torch.Tensor) -> torch.Tensor:
        return self.encoder_u(u)

    def decode(self, g: torch.Tensor) -> torch.Tensor:
        return self.decoder(g)

    def dynamics(self, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Computes one-step lifted state transition: g_{k+1} = A*g_k + B*φ(u_k)."""
        g_u = self.encode_u(u)
        return self.A(g) + self.B(g_u)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> Tuple:
        g_k           = self.encode(x)
        x_rec         = self.decode(g_k)
        g_k_next_pred = self.dynamics(g_k, u)
        x_next_pred   = self.decode(g_k_next_pred)
        return x_rec, g_k, g_k_next_pred, x_next_pred

    def train_model(
        self,
        epochs: int = 500,
        alpha_rec: float = 1.0,
        alpha_lin: float = 0.5,
        alpha_pred: float = 2.0,
        alpha_multi: float = 0.5,
        patience: int = 50,
        lr_scheduler: bool = True,
        lr_patience: int = 20,
        lr_factor: float = 0.5,
        lr_min: float = 1e-6,
    ) -> dict:
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode = 'min',
                factor = lr_factor,
                patience = lr_patience,
                min_lr = lr_min,
            ) if lr_scheduler else None
        )

        best_loss = float("inf")
        patience_counter = 0
        best_weights = None

        self.train()

        for epoch in range(epochs):
            total_loss       = 0.0
            total_loss_rec   = 0.0
            total_loss_lin   = 0.0
            total_loss_pred  = 0.0
            total_loss_multi = 0.0

            for batch_X_seq, batch_U_seq in self.dataloader:
                self.optimizer.zero_grad()
                
                batch_X_k = batch_X_seq[:, 0, :]
                batch_U_k = batch_U_seq[:, 0, :]
                batch_X_k_next_true = batch_X_seq[:, 1, :]

                x_rec, g_k, g_k_next_pred, x_next_pred = self(batch_X_k, batch_U_k)
                g_k_next_true = self.encode(batch_X_k_next_true)

                loss_rec  = self.criterion(x_rec, batch_X_k)
                loss_lin  = self.criterion(g_k_next_pred, g_k_next_true)
                loss_pred = self.criterion(x_next_pred, batch_X_k_next_true)

                loss_multi = torch.tensor(0.0, device=self.device)
                g_roll = g_k
                
                for step in range(self.rollout_steps):
                    curr_U        = batch_U_seq[:, step, :]
                    target_X_next = batch_X_seq[:, step + 1, :]
                    
                    g_roll      = self.dynamics(g_roll, curr_U)
                    x_roll_pred = self.decode(g_roll)
                    loss_multi += self.criterion(x_roll_pred, target_X_next)

                loss = (alpha_rec   * loss_rec
                      + alpha_lin   * loss_lin
                      + alpha_pred  * loss_pred
                      + alpha_multi * loss_multi)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss       += loss.item()
                total_loss_rec   += loss_rec.item()
                total_loss_lin   += loss_lin.item()
                total_loss_pred  += loss_pred.item()
                total_loss_multi += loss_multi.item()

            n              = len(self.dataloader)
            avg_loss       = total_loss       / n
            avg_loss_rec   = total_loss_rec   / n
            avg_loss_lin   = total_loss_lin   / n
            avg_loss_pred  = total_loss_pred  / n
            avg_loss_multi = total_loss_multi / n
            current_lr     = self.optimizer.param_groups[0]["lr"]

            self.history["loss"].append(avg_loss)
            self.history["loss_rec"].append(avg_loss_rec)
            self.history["loss_lin"].append(avg_loss_lin)
            self.history["loss_pred"].append(avg_loss_pred)
            self.history["loss_multi"].append(avg_loss_multi)
            self.history["lr"].append(current_lr)

            if scheduler is not None:
                scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss        = avg_loss
                patience_counter = 0
                best_weights     = {k: v.clone() for k, v in self.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch + 1} "
                      f"(no improvement for {patience} epochs)")
                print(f"  Best loss: {best_loss:.6f}")
                self.load_state_dict(best_weights)
                break

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:>4}/{epochs}] | "
                      f"Loss: {avg_loss:.6f} "
                      f"(Rec: {avg_loss_rec:.4f}, "
                      f"Lin: {avg_loss_lin:.4f}, "
                      f"Pred: {avg_loss_pred:.4f}, "
                      f"Multi: {avg_loss_multi:.4f}) | "
                      f"LR: {current_lr:.2e}")

        else:
            if best_weights is not None:
                self.load_state_dict(best_weights)
            print(f"\n  Training complete. Best loss: {best_loss:.6f}")

        plt.figure(figsize=(12, 4))
        plt.plot(self.history["loss"], label="Total")
        plt.plot(self.history["loss_pred"], label="Pred")
        plt.plot(self.history["loss_rec"], label="Rec")
        plt.plot(self.history["loss_multi"], label="Multi")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.title("Training loss history")
        plt.show()

        return self.history

    def evaluate(self, print_metrics: bool = False, plot: bool = True, u_plot: np.ndarray = None) -> Tuple[np.ndarray, float, float]:
        """
        Evaluates the trained Koopman model on the test set.
        """
        x_ref_orig = self.scaler_X.inverse_transform(self.data.get("x_ref"))
        u_ref_orig = self.scaler_U.inverse_transform(self.data.get("u_ref"))

        x_sim_orig = self.simulate(x_ref_orig, u_ref_orig)

        if np.isnan(x_sim_orig).any():
            print("ERROR: Prediction contains NaN — model likely diverges.")
            return {"rmse": float("nan"), "r2": float("nan")}

        rmse = root_mean_squared_error(x_ref_orig, x_sim_orig)
        r2   = r2_score(x_ref_orig, x_sim_orig)

        if print_metrics:
            print(f"Koopman model state R2 score: {r2:.3%}")
            print(f"Koopman model state RMSE: {rmse:.5f}")

        u_sim = u_plot if u_plot is not None else u_ref_orig

        if plot:
            plot_trajectory(compute_time_vector(x_sim_orig, self.data.get("dt")), x_ref_orig, x_sim_orig, u_sim, title="Validation on test data")

        return x_sim_orig, rmse, r2

    def plot_koopman_spectrum(self, exportable: bool = False):
        A_matrix       = self.A.weight.detach().cpu().numpy()
        eigenvalues, _ = np.linalg.eig(A_matrix)
        plot_koopman_spectrum(eigenvalues, exportable=exportable)

    def simulate(self, x_ref: np.ndarray, u_ref: np.ndarray) -> np.ndarray:
        """
        Simulates the Koopman model forward in time given initial conditions and control inputs.

        Args:
            x_ref (np.ndarray): Reference state variables (unscaled). Used to extract the initial state x0.
            u_ref (np.ndarray): Sequence of control inputs (unscaled).

        Returns:
            np.ndarray: Simulated state trajectory (unscaled / original scale).
        """

        x_ref_scaled = self.scaler_X.transform(x_ref)
        u_ref_scaled = self.scaler_U.transform(u_ref)

        x0 = torch.tensor(x_ref_scaled[0], dtype=torch.float32).to(self.device)
        U_seq = torch.tensor(u_ref_scaled[:-1], dtype=torch.float32).to(self.device)

        self.eval()
        with torch.no_grad():
            if x0.dim() == 1:
                x0 = x0.unsqueeze(0)

            g = self.encode(x0)
            predictions = [x0]

            for i in range(U_seq.shape[0]):
                u_curr = U_seq[i].unsqueeze(0)
                g      = self.dynamics(g, u_curr)
                predictions.append(self.decode(g))

        x_sim_scaled = torch.cat(predictions, dim=0).cpu().numpy()
        x_sim = self.scaler_X.inverse_transform(x_sim_scaled)

        return x_sim

    def export_data(self, export_file_name: str = "KoopmanNeural_operator", export_path: str = "."):
        A_matrix = self.A.weight.detach().cpu().numpy()
        B_matrix = self.B.weight.detach().cpu().numpy()

        obs_names = [f"g_{i}" for i in range(self.lifted_dim)]

        eigenvalues, eigenvectors = np.linalg.eig(A_matrix)
        w_matrix_export = [
            [str(complex(eigenvectors[row, col])) for col in range(eigenvectors.shape[1])]
            for row in range(eigenvectors.shape[0])
        ]

        payload = {
            "observables":            obs_names,
            "A matrix":               self._format_matrix(A_matrix, obs_names),
            "B_matrix":               self._format_matrix(B_matrix, obs_names),
            "Spectral decomposition": self._intepret_eigenvalues(eigenvalues),
            "Mode decomposition":     w_matrix_export
        }

        try:
            filepath = self.data_export_path / f"{export_file_name}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=5, default=str)
        except Exception as e:
            warnings.warn(str(e))

    def _intepret_eigenvalues(self, lambda_array: np.ndarray) -> list:
        interpreted_eigenvalues = []
        for i, lambda_val in enumerate(lambda_array):
            magnitude = np.abs(lambda_val)
            phase_rad = np.angle(lambda_val)

            stability_status = ""
            if magnitude < 1 - 1e-9:
                stability_status += "Dampened (stable)"
            elif magnitude > 1 + 1e-9:
                stability_status += "Growing (unstable)"
            else:
                stability_status += "Stable (on unit circle)"

            if np.abs(phase_rad) > 1e-9:
                stability_status += ", Oscillating"
            else:
                stability_status += ", Non-oscillating"

            eigenvalue_info = {
                "id":                   f"lambda_{i + 1}",
                "complex_value":        str(complex(lambda_val)),
                "real_part":            float(lambda_val.real),
                "imag_part":            float(lambda_val.imag),
                "magnitude":            float(magnitude),
                "phase_rad_per_sample": float(phase_rad),
                "interpretation":       stability_status
            }
            interpreted_eigenvalues.append(eigenvalue_info)
        return interpreted_eigenvalues

    def _format_matrix(self, matrix: np.ndarray, observables: list) -> list:
        matrix = matrix.tolist() if isinstance(matrix, np.ndarray) else matrix
        max_width = max(len(str(val)) for row in matrix for val in row)

        matrix_formatted = []
        for i, row in enumerate(matrix):
            matrix_formatted.append({
                "observable": observables[i],
                "row_values": [f"{val:{max_width}}" for val in row]
            })

        return matrix_formatted