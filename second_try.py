from cProfile import label

import numpy as np
from scipy.optimize import curve_fit, root_scalar
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel("C:/Users/user/Nextcloud/Doktor/10_Paper/MDPI_applSci_Focus_Issue_ADR/Data/MKL/MeanField/GdB3_0K4.xlsx", sheet_name=0)

df["H"] = pd.to_numeric(df["H"], errors="coerce")
df["M"] = pd.to_numeric(df["M"], errors="coerce")

# Example: assume columns are "H" (Tesla) and "M" (emu/mol or A·m²/kg)
H_data = df["H"].to_numpy()
M_data = df["M"].to_numpy()
M_data_meanField = np.zeros_like(M_data)


T = 0.4
g = 2
J = 7/2
kB = 1.380649e-23
muB = 9.2740100783e-24
alpha = 0.1 # mean field parameter, likely in range 0.05 ... 0.5
Msat = max(M_data)

def coth(x):
    return np.cosh(x)/np.sinh(x)

def B_J(x):
    return ((2*J + 1)/(2*J)) * coth(x*(2*J + 1)/(2*J)) - (1/(2*J))*coth(x/(2*J))

def to_x_meanField(H, alpha, m):
    return (H - alpha * m) * g * muB * J / (kB * T)

def to_x_classical(H):
    return g * muB * J / (kB * T) * H

def M_mean_field_single(H, Ms, alpha):
    def f(M):
        x = to_x_meanField(H, alpha, M)
        return M - Ms * B_J(x)

    sol = root_scalar(f, bracket=[-Ms, Ms], method='brentq')
    #print(sol.root - Ms * B_J(to_x_meanField(H, alpha, sol.root))) #quick test to the solution
    return sol.root


M_free_ion = Msat * B_J(to_x_classical(H_data))

def solve_model(H_data, alpha):
    for i in range(len(M_data)):
        M_data_meanField[i] = M_mean_field_single(H_data[i], Msat, alpha)
    return M_data_meanField


# --- Initial guesses ---
p0 = [-0.1]  # [alpha]

# --- Perform nonlinear fit (this also runs the "solve model" function ---
popt, pcov = curve_fit(solve_model, H_data, M_data, p0 = p0)
alpha_fit = popt
print(alpha_fit)

plt.figure(figsize=(7,5))
plt.plot(H_data, M_free_ion, '--' , label="free ion")
plt.plot(H_data, M_data_meanField, '-' , label=f"MeanField {alpha_fit}")
plt.plot(H_data, M_data, 'o', label="GdB3_0K4", alpha=0.7)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
