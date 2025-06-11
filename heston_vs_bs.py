import numpy as np
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt

# --- PARÁMETROS DE LA ACCIÓN ---
ticker = 'WMT'  # Cambia por cualquier empresa del S&P 500
data = yf.Ticker(ticker)
S0 = data.history(period='1d')['Close'].iloc[-1]

# --- ESTIMACIÓN DE VOLATILIDAD HISTÓRICA ---
hist_prices = data.history(period='6mo')['Close']
returns = hist_prices.pct_change().dropna()
sigma = returns.std() * np.sqrt(252)     # anualizada

# --- OTROS PARÁMETROS ---
K = S0 # Precio de ejercicio (ATM)
T = 1.0 # Tiempo hasta el vencimiento (en años)
r = 0.045 # Tasa de interés libre de riesgo (anualizada)

# --- BLACK-SCHOLES ---
d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
call_bs = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

# --- PARÁMETROS HESTON ---
v0 = sigma**2 # varianza inicial
kappa = 2.0      # velocidad de reversión
theta = v0*1.2     # media de varianza a largo plazo
xi = 0.15         # volatilidad de la volatilidad
rho = -0.3       # correlación entre procesos

# --- SIMULACIÓN HESTON ---
N = 10000    # simulaciones
dt = 1/252 # Se considera que hay 252 días de mercado al año
M = int(T/dt)
payoffs = []

for _ in range(N):
    S = S0
    v = v0
    for _ in range(M):
        Z1 = np.random.normal()
        Z2 = np.random.normal()
        dW_v = Z1 * np.sqrt(dt)
        dW_S = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)
        v = np.abs(v + kappa * (theta - v) * dt + xi * np.sqrt(max(v, 0)) * dW_v)
        S = S * np.exp((r - 0.5 * v) * dt + np.sqrt(max(v, 0)) * dW_S)
    payoff = max(S - K, 0)
    payoffs.append(payoff)

call_heston = np.exp(-r * T) * np.mean(payoffs)

# --- COMPARACIÓN TABULADA ---
print("\nCOMPARACIÓN DE MODELOS PARA", ticker)
print(f"{'Modelo':35} {'Precio estimado'}")
print("-"*50)
print(f"{'Black-Scholes (clásico)':35} {call_bs:.2f}")
print(f"{'Heston (volatilidad est.)':35} {call_heston:.2f}")
print("-"*50)
print(f"{'Diferencia porcentual':35} {abs(((call_heston - call_bs) / call_bs)) * 100:.2f}%")