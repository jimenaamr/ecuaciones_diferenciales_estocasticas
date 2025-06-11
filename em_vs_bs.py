import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf

# Elegimos una empresa del S&P 500
ticker = 'WMT'  # Cambia por cualquier empresa del S&P 500
data = yf.Ticker(ticker)

# Precio actual
S0 = data.history(period='1d')['Close'].iloc[-1]

# Volatilidad histórica anualizada
hist_prices = data.history(period='6mo')['Close']
returns = hist_prices.pct_change().dropna()
sigma = returns.std() * np.sqrt(252)

# Parámetros
K = S0 # Precio de ejercicio
T = 1.0 # Tiempo hasta el vencimiento (en años)
r = 0.045 # Tasa de interés libre de riesgo (anualizada)
N = 10000 # Número de simulaciones
dt = 1/252 # Paso de tiempo (1 día)
M = int(T/dt) # Número de pasos
num_paths = 100 # Número de trayectorias a graficar

# Trayectorias (Euler-Maruyama) Cálculo de precios paso a paso
# Es un código menos eficiente pero más claro para entender el método de Euler-Maruyama
S_paths = np.zeros((num_paths, M + 1))
S_paths[:, 0] = S0
for t in range(1, M + 1):
    Z = np.random.normal(size=num_paths)
    S_paths[:, t] = S_paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# Precio Monte Carlo
S = np.full(N, S0)
for _ in range(M):
    Z = np.random.normal(size=N)
    S *= np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
payoffs = np.maximum(S - K, 0)
call_price_mc = np.exp(-r*T) * np.mean(payoffs)

# Precio Black-Scholes
d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
call_price_bs = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

# Comparación de modelos
print("\nCOMPARACIÓN DE MODELOS PARA", ticker)
print(f"{'Modelo':35} {'Precio estimado'}")
print("-"*50)
print(f"{'Black-Scholes':35} {call_price_bs:.2f}")
print(f"{'Monte-Carlo':35} {call_price_mc:.2f}")
print("-"*50)
print(f"{'Diferencia porcentual':35} {abs(((call_price_mc - call_price_bs) / call_price_bs)) * 100:.2f}%")