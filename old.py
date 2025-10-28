import numpy as np
from scipy.optimize import curve_fit, root_scalar
import pandas as pd
import matplotlib.pyplot as plt


# Suppose we have data:

df = pd.read_excel("C:/Users/user/Nextcloud/Doktor/10_Paper/MDPI_applSci_Focus_Issue_ADR/Data/MKL/MeanField/GdB3_0K4.xlsx", sheet_name=0)

df["H"] = pd.to_numeric(df["H"], errors="coerce")
df["M"] = pd.to_numeric(df["M"], errors="coerce")

# Example: assume columns are "H" (Tesla) and "M" (emu/mol or A·m²/kg)
H_data = df["H"].to_numpy()
M_data = df["M"].to_numpy()

T0 = 0.4
J = 7/2

kB = 1.380649e-23
muB = 9.2740100783e-24

def coth(x):
    return np.cosh(x)/np.sinh(x)

def B_J(x, J):
    x = np.clip(x, -1e6, 1e6)
    return ((2*J+1)/(2*J))*1/coth((2*J+1)*x/(2*J)) - (1/(2*J))*1/coth(x/(2*J))


# --- Self-consistent mean-field magnetization (fixed-point iteration) ---
def M_mean_field_single(H, Ms, g, J, T, lam):
    """Solve M = Ms * B_J(gμBJ(H + λM)/kBT) via stable iteration."""
    M = 1
    for i in range(20):
        x = (g * muB * J * (H + lam * M)) / (kB * T)
        M_new = Ms * B_J(x, J)
        #print(f"{i}, {lam:.3e}, {M_new:.3e}")
        if abs(M_new - M) < 1e-8 * Ms:
            return M_new
        M = 0.5 * M + 0.5 * M_new
    return M  # return last estimate if not converged

# --- Vectorized model for curve fitting ---
def M_mean_field(H_array, Ms, lam, J, T, g=2.0):
    return np.array([M_mean_field_single(H, Ms, g, J, T, lam) for H in H_array])


# --- Define model for fitting (with fixed g=2) ---
def model_to_fit(H, Ms, lam):
    return M_mean_field(H, Ms, lam, J, T0, g=2.0)

# --- Initial guesses ---
p0 = [np.max(M_data), 0.1]  # [Ms, λ]

# --- Perform nonlinear fit ---
popt, pcov = curve_fit(model_to_fit, H_data, M_data, p0=p0, maxfev=5000)
Ms_fit, lam_fit = popt

print("=== Fitted Parameters ===")
print(f"  Ms  = {Ms_fit:.3e}")
print(f"  λ   = {lam_fit:.3e}")

# --- Generate fitted model curve ---
M_fit = model_to_fit(H_data, Ms_fit, -0.01)

# --- Plotting function ---
def plot_fit(H, M_exp, M_fit, T, params, g):
    Ms_fit, lam_fit = params
    plt.figure(figsize=(7,5))
    plt.plot(H, M_exp, 'o', label='Experimental data', alpha=0.7)
    plt.plot(H, M_fit, '-', lw=2, label='Mean-field fit (g=2)')
    plt.xlabel("Magnetic field H (T)")
    plt.ylabel("Magnetization M")
    plt.title(f"Mean-field fit at T = {T:.1f} K\n"
              f"$M_s$ = {Ms_fit:.2e},  g = {g:.1f},  λ = {lam_fit:.2e}")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

# --- Call plotting function ---
plot_fit(H_data, M_data, M_fit, T0, (Ms_fit, lam_fit), 2)

