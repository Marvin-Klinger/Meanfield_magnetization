import numpy as np
from scipy.linalg import solve
from scipy.optimize import curve_fit, root_scalar
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import alpha

df = pd.read_excel("C:/Users/klinmarv/Nextcloud2/Doktor/10_Paper/MDPI_applSci_Focus_Issue_ADR/Data/MKL/MeanField/GdB9_2K0.xlsx", sheet_name=0)

df["H"] = pd.to_numeric(df["H"], errors="coerce")
df["M"] = pd.to_numeric(df["M"], errors="coerce")

# Example: assume columns are "H" (Tesla) and "M" (emu/mol or A·m²/kg)
H_data = df["H"].to_numpy()
M_data = df["M"].to_numpy()
M_data_meanField = np.zeros_like(M_data)

T = 2.0
g = 2
J = 7/2
kB = 1.380649e-23
muB = 9.2740100783e-24
Msat = max(M_data)

def coth(x):
    return np.cosh(x)/np.sinh(x)


def B_J(x):
    return ((2*J + 1)/(2*J)) * coth(x*(2*J + 1)/(2*J)) - (1/(2*J))*coth(x/(2*J))


def to_x_classical(H):
    return g * muB * J / (kB * T) * H


def to_x_meanField(H, alpha, m):
    return (H - alpha * m) * g * muB * J / (kB * T)


def M_mean_field_single(H, Ms, alpha):
    # f(M) must be zero for mean field solution
    def f(M):
        x = to_x_meanField(H, alpha, M)
        return M - Ms * B_J(x)

    # solve the diff-eqation and find M where f(M) = 0
    sol = root_scalar(f, bracket=[-Ms, Ms], method='brentq')
    #print(sol.root - Ms * B_J(to_x_meanField(H, alpha, sol.root))) #quick test to the solution (lower is better)
    return sol.root


# solve for every H in the experimental dataset
def solve_model(H_data, alpha):
    for i in range(len(M_data)):
        M_data_meanField[i] = M_mean_field_single(H_data[i], Msat, alpha)
    return M_data_meanField



# [program start]
# free ion model for comparison
M_free_ion = Msat * B_J(to_x_classical(H_data))

# optimize alpha for the entire dataset
p0 = [-0.1]  # initial guess for alpha
popt, pcov = curve_fit(solve_model, H_data, M_data, p0 = p0)
alpha_fit = popt

M_data_meanField = solve_model(H_data, alpha_fit)
print(f" fitted parameter alpha = {alpha_fit}")

plt.figure(figsize=(7,5))

plt.plot(H_data, M_data, 'o', label="GdB3 2K", alpha=0.7, color='green')
plt.plot(H_data, M_free_ion, '--' , label="free ion", color='black')
plt.plot(H_data, M_data_meanField, '-' , label=f"MeanField {alpha_fit}", color = 'red')

plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()


export_df = pd.DataFrame({
    'Temperature': H_data,
    'Fitted Moment': M_data_meanField
})

# Export both datasets to CSV
with pd.ExcelWriter('GdB9_2K0.xlsx') as writer:
    export_df.to_excel(writer, sheet_name='Fitted Curve', index=False)