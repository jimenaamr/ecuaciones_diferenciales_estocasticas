# import numpy as np
# import yfinance as yf
# import matplotlib.pyplot as plt
# from scipy.interpolate import make_interp_spline
# from scipy.optimize import minimize

# def calibrar_heston():
#     """
#     Calibración del modelo de Heston para simular la media funcional de precios de SPY
#     y compararla con los datos reales.
#     Utiliza el método de optimización para ajustar los parámetros del modelo.
#     """

#     # --- Descargar datos reales ---
#     spy = yf.Ticker("SPY")
#     hist = spy.history(period="1y")["Close"]
#     dias = len(hist)
#     S0 = hist.iloc[0]

#     # Volatilidad y drift anualizados
#     returns = hist.pct_change().dropna()
#     sigma = returns.std() * np.sqrt(252)
#     mu = returns.mean() * 252
#     dt = 1/252

#     # --- Función de simulación media funcional dado un set de parámetros ---
#     def simula_media_funcional(params):
#         kappa, theta, xi, rho, v0 = params
#         num_sim = 100
#         S_simuladas_heston = np.zeros((num_sim, dias))
#         for i in range(num_sim):
#             S_path = np.zeros(dias)
#             v_path = np.zeros(dias)
#             S_path[0] = S0
#             v_path[0] = v0
#             for t in range(1, dias):
#                 Z1 = np.random.normal()
#                 Z2 = np.random.normal()
#                 dW_v = Z1 * np.sqrt(dt)
#                 dW_S = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)
#                 v_path[t] = np.abs(
#                     v_path[t-1] + kappa*(theta - v_path[t-1])*dt + xi * np.sqrt(max(v_path[t-1], 0)) * dW_v
#                 )
#                 S_path[t] = S_path[t-1] * np.exp(
#                     (mu - 0.5 * v_path[t-1]) * dt + np.sqrt(max(v_path[t-1], 0)) * dW_S
#                 )
#             S_simuladas_heston[i] = S_path
#         media_puntual = S_simuladas_heston.mean(axis=0)
#         return media_puntual

#     # --- Función de error a minimizar (MSE entre media funcional y serie real) ---
#     def mse(params):
#         media_func = simula_media_funcional(params)
#         return np.mean((media_func - hist.values)**2)

#     # --- Parámetros iniciales (puedes afinar estos valores) ---
#     x0 = [1.0, sigma**2, 0.4, -0.7, sigma**2]  # kappa, theta, xi, rho, v0
#     bounds = [
#         (0.01, 10),      # kappa
#         (0.0001, 1),     # theta
#         (0.01, 1),       # xi
#         (-0.99, 0.0),    # rho (debe ser negativo para realismo)
#         (0.0001, 1)      # v0
#     ]

#     # --- Optimización (puede tardar varios minutos) ---
#     result = minimize(mse, x0, bounds=bounds, method='L-BFGS-B', options={'maxiter': 20})
#     print("Parámetros calibrados:", result.x)
#     print("MSE final:", result.fun)

#     # --- Simula con los parámetros óptimos y grafica ---
#     media_func_opt = simula_media_funcional(result.x)
#     x = np.arange(dias)
#     x_smooth = np.linspace(0, dias - 1, 10 * dias)
#     spline = make_interp_spline(x, media_func_opt, k=3)
#     media_funcional = spline(x_smooth)

#     # --- Graficar resultados ---
#     plt.figure(figsize=(11,6))
#     plt.plot(x, hist.values, color='blue', lw=2, label='Precio real SPY')
#     plt.plot(x_smooth, media_funcional, color='orange', lw=2.2, label='Media funcional simulada (calibrada)')
#     plt.title('Media funcional calibrada (Heston) vs. precio real SPY')
#     plt.xlabel('Días')
#     plt.ylabel('Precio')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     calibrar_heston()


# import numpy as np
# import yfinance as yf
# import matplotlib.pyplot as plt
# from scipy.optimize import minimize

# def calibrar_heston_classic():
#     # --- Descargar datos reales ---
#     spy = yf.Ticker("SPY")
#     hist = spy.history(period="1y")["Close"]
#     dias = len(hist)
#     S0 = hist.iloc[0]

#     # Volatilidad y drift anualizados
#     returns = hist.pct_change().dropna()
#     sigma = returns.std() * np.sqrt(252)
#     mu = returns.mean() * 252
#     dt = 1 / 252

#     # --- Transformaciones suaves de parámetros ---
#     def unpack_params(x):
#         # x: vector en R^5, todo sin restricciones
#         kappa = np.exp(x[0])               # [0, +inf)
#         theta = 0.0001 + 0.9999 * sigmoid(x[1])  # [0.0001, 1]
#         xi    = 0.01 + 0.99 * sigmoid(x[2])      # [0.01, 1]
#         rho   = -0.99 + 0.99 * sigmoid(x[3])     # [-0.99, 0]
#         v0    = 0.0001 + 0.9999 * sigmoid(x[4])  # [0.0001, 1]
#         return kappa, theta, xi, rho, v0

#     def sigmoid(x):
#         return 1 / (1 + np.exp(-x))

#     # --- Simulación Heston ---
#     def simula_media_funcional(params, seed=42):
#         kappa, theta, xi, rho, v0 = params
#         np.random.seed(seed)  # Fijamos semilla para reproducibilidad
#         num_sim = 100          # Puedes subir para resultados más suaves
#         S_simuladas = np.zeros((num_sim, dias))
#         for i in range(num_sim):
#             S_path = np.zeros(dias)
#             v_path = np.zeros(dias)
#             S_path[0] = S0
#             v_path[0] = v0
#             for t in range(1, dias):
#                 Z1 = np.random.normal()
#                 Z2 = np.random.normal()
#                 dW_v = Z1 * np.sqrt(dt)
#                 dW_S = (rho * Z1 + np.sqrt(1 - rho ** 2) * Z2) * np.sqrt(dt)
#                 v_prev = max(v_path[t-1], 0)
#                 v_path[t] = abs(
#                     v_path[t-1] + kappa*(theta - v_prev)*dt + xi * np.sqrt(max(v_prev, 0)) * dW_v
#                 )
#                 S_path[t] = S_path[t-1] * np.exp(
#                     (mu - 0.5 * v_prev) * dt + np.sqrt(max(v_prev, 0)) * dW_S
#                 )
#             S_simuladas[i] = S_path
#         media_puntual = S_simuladas.mean(axis=0)
#         return media_puntual

#     # --- Función de error a minimizar ---
#     def mse(x):
#         params = unpack_params(x)
#         media_func = simula_media_funcional(params, seed=123)  # semilla fija
#         error = np.mean((media_func - hist.values) ** 2)
#         print(f"Probing: kappa={params[0]:.4f}, theta={params[1]:.4f}, xi={params[2]:.4f}, rho={params[3]:.4f}, v0={params[4]:.4f} | MSE={error:.2f}")
#         return error

#     # --- Parámetros iniciales en el espacio transformado ---
#     x0 = np.zeros(5)

#     # --- Optimización ---
#     result = minimize(mse, x0, method='L-BFGS-B', options={'maxiter': 30, 'disp': True})

#     print("\nParámetros calibrados:")
#     kappa, theta, xi, rho, v0 = unpack_params(result.x)
#     print(f"kappa={kappa:.4f}, theta={theta:.6f}, xi={xi:.4f}, rho={rho:.4f}, v0={v0:.6f}")
#     print(f"MSE final: {result.fun:.4f}")

#     # --- Simula y grafica con los parámetros óptimos ---
#     media_func_opt = simula_media_funcional(unpack_params(result.x), seed=999)
#     x = np.arange(dias)
#     plt.figure(figsize=(11, 6))
#     plt.plot(x, hist.values, color='blue', lw=2, label='Precio real SPY')
#     plt.plot(x, media_func_opt, color='orange', lw=2.2, label='Media funcional simulada (calibrada)')
#     plt.title('Media funcional calibrada (Heston) vs. precio real SPY')
#     plt.xlabel('Días')
#     plt.ylabel('Precio')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     calibrar_heston_classic()


import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def unpack_params(x):
    kappa = np.exp(x[0])
    theta = 0.0001 + 0.9999 * sigmoid(x[1])
    xi    = 0.01 + 0.99 * sigmoid(x[2])
    rho   = -0.99 + 0.99 * sigmoid(x[3])
    v0    = 0.0001 + 0.9999 * sigmoid(x[4])
    return kappa, theta, xi, rho, v0

def simula_media_funcional(params, S0, dias, mu, dt, seed=42):
    kappa, theta, xi, rho, v0 = params
    np.random.seed(seed)
    num_sim = 100
    S_simuladas = np.zeros((num_sim, dias))
    for i in range(num_sim):
        S_path = np.zeros(dias)
        v_path = np.zeros(dias)
        S_path[0] = S0
        v_path[0] = v0
        for t in range(1, dias):
            Z1 = np.random.normal()
            Z2 = np.random.normal()
            dW_v = Z1 * np.sqrt(dt)
            dW_S = (rho * Z1 + np.sqrt(1 - rho ** 2) * Z2) * np.sqrt(dt)
            v_prev = max(v_path[t-1], 0)
            v_path[t] = abs(
                v_path[t-1] + kappa*(theta - v_prev)*dt + xi * np.sqrt(max(v_prev, 0)) * dW_v
            )
            S_path[t] = S_path[t-1] * np.exp(
                (mu - 0.5 * v_prev) * dt + np.sqrt(max(v_prev, 0)) * dW_S
            )
        S_simuladas[i] = S_path
    media_puntual = S_simuladas.mean(axis=0)
    return media_puntual

# Descargar datos reales
spy = yf.Ticker("SPY")
hist = spy.history(period="1y")["Close"].values
dias = len(hist)

# Dividir en dos mitades
split = dias // 2
hist_train = hist[:split]
hist_test = hist[split:]
S0_train = hist_train[0]
S0_test = hist_train[-1]
dias_test = len(hist_test)

# Estadísticos SOLO del primer tramo (entrenamiento)
returns_train = np.diff(hist_train) / hist_train[:-1]
sigma_train = np.std(returns_train) * np.sqrt(252)
mu_train = np.mean(returns_train) * 252
dt = 1/252

# Calibración SOLO con la primera mitad
def mse(x):
    params = unpack_params(x)
    media_func = simula_media_funcional(params, S0_train, len(hist_train), mu_train, dt)
    return np.mean((media_func - hist_train) ** 2)

x0 = np.zeros(5)
result = minimize(mse, x0, method='L-BFGS-B', options={'maxiter': 30, 'disp': True})
params = unpack_params(result.x)
print("Parámetros calibrados con la primera mitad:")
print(f"kappa={params[0]:.4f}, theta={params[1]:.4f}, xi={params[2]:.4f}, rho={params[3]:.4f}, v0={params[4]:.4f}")

# Predecir la segunda mitad: simular usando S0_test y parámetros calibrados
pred_media = simula_media_funcional(params, S0_test, dias_test, mu_train, dt, seed=2024)

# Graficar resultado
x = np.arange(dias)
plt.figure(figsize=(11,6))
plt.plot(np.arange(split), hist_train, label='Real (primera mitad)', color='blue')
plt.plot(np.arange(split, dias), hist_test, label='Real (segunda mitad)', color='green')
plt.plot(np.arange(split, dias), pred_media, label='Predicción Heston (calibrado en 1ª mitad)', color='orange')
plt.axvline(split, color='grey', linestyle='--', alpha=0.6)
plt.xlabel('Días')
plt.ylabel('Precio')
plt.title('Heston: calibrado en la primera mitad, predice la segunda')
plt.legend()
plt.tight_layout()
plt.show()
