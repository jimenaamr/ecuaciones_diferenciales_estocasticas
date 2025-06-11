import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# COMPARACIÓN TRAYECTORIAS REALES VS SIMULADAS

def black_scholes_vs_real():
    """
    Simula trayectorias de precios usando el modelo Black-Scholes y las compara con los precios reales del ETF SPY.
    Utiliza el método de Monte Carlo para generar trayectorias y luego calcula la media de las simulaciones.
    Gráfica la media de las trayectorias simuladas y la trayectoria real del ETF SPY.
    """

    # Descarga precios históricos del ETF SPY - 1 año
    spy = yf.Ticker("SPY")
    year = 1
    hist = spy.history(period=str(year)+"y")["Close"]
    dias = len(hist)
    S0 = hist.iloc[0]
    dias_negocio = year * 252  # Asumiendo 252 días de negocio al año

    # Volatilidad y drift anualizados
    returns = hist.pct_change().dropna()
    sigma = returns.std() * np.sqrt(dias_negocio)
    mu = returns.mean() * dias_negocio
    dt = 1/dias_negocio  # Paso de tiempo diario

    # Simulaciones Black-Scholes
    num_sim = 500
    S_simuladas = np.zeros((num_sim, dias))
    for i in range(num_sim):
        W = np.cumsum(np.random.normal(0, np.sqrt(dt), size=dias-1))
        S_sim = S0 * np.exp((mu - 0.5 * sigma**2) * np.linspace(0, 1, dias-1) + sigma * W)
        S_simuladas[i, 0] = S0
        S_simuladas[i, 1:] = S_sim

    # Calcula la media simulada en cada instante de tiempo
    media_simulada = S_simuladas.mean(axis=0)

    # Gráfica
    plt.figure(figsize=(11,6))
    # Opcional: pinta todas las trayectorias en gris claro
    for i in range(num_sim):
        plt.plot(S_simuladas[i], color='gray', alpha=0.05)
    # Media simulada
    plt.plot(media_simulada, color='orange', lw=2.2, label='Media simulada Black-Scholes')
    # Trayectoria real
    plt.plot(hist.values, color='blue', lw=2, label='Precio real SPY')
    plt.title('Trayectorias simuladas vs. precio real SPY')
    plt.xlabel('Días')
    plt.ylabel('Precio')
    plt.legend()
    plt.tight_layout()
    plt.show()


def euler_maruyama_vs_real():
    """
    Simula trayectorias de precios usando el método de Euler-Maruyama y las compara con los precios reales del ETF SPY.
    Utiliza el modelo de Black-Scholes para generar trayectorias y luego calcula la media de las simulaciones.
    Gráfica la media de las trayectorias simuladas y la trayectoria real del ETF SPY.
    """

    # --- Descargar datos reales ---
    spy = yf.Ticker("SPY")
    hist = spy.history(period="1y")["Close"]
    dias = len(hist)
    S0 = hist.iloc[0]

    # Volatilidad y drift anualizados
    returns = hist.pct_change().dropna()
    sigma = returns.std() * np.sqrt(252)
    mu = returns.mean() * 252
    dt = 1/252

    # --- Simulaciones usando Euler-Maruyama ---
    num_sim = 500
    S_simuladas_em = np.zeros((num_sim, dias))
    for i in range(num_sim):
        S_path = np.zeros(dias)
        S_path[0] = S0
        for t in range(1, dias):
            Z = np.random.normal()
            # Versión multiplicativa (log-normal)
            S_path[t] = S_path[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            # Si quieres versión aditiva (menos precisa para precios): 
            # S_path[t] = S_path[t-1] + mu * S_path[t-1] * dt + sigma * S_path[t-1] * np.sqrt(dt) * Z
        S_simuladas_em[i] = S_path

    # --- Media simulada en cada instante de tiempo ---
    media_simulada_em = S_simuladas_em.mean(axis=0)

    # --- Gráfica ---
    plt.figure(figsize=(11,6))
    for i in range(num_sim):
        plt.plot(S_simuladas_em[i], color='gray', alpha=0.05)
    plt.plot(media_simulada_em, color='orange', lw=2.2, label='Media simulada Euler-Maruyama')
    plt.plot(hist.values, color='blue', lw=2, label='Precio real SPY')
    plt.title('Trayectorias simuladas (Euler-Maruyama) vs. precio real SPY')
    plt.xlabel('Días')
    plt.ylabel('Precio')
    plt.legend()
    plt.tight_layout()
    plt.show()


def heston_vs_real():
    """
    Simula trayectorias de precios usando el modelo Heston y las compara con los precios reales del ETF SPY.
    Gráfica la media de las trayectorias simuladas y la trayectoria real del ETF SPY.
    """

    # --- Descargar datos reales ---
    spy = yf.Ticker("SPY")
    hist = spy.history(period="1y")["Close"]
    dias = len(hist)
    S0 = hist.iloc[0]

    # Volatilidad y drift anualizados
    returns = hist.pct_change().dropna()
    sigma = returns.std() * np.sqrt(252)
    mu = returns.mean() * 252
    dt = 1/252

    # --- Parámetros Heston ---
    v0 = sigma**2          # Varianza inicial
    kappa = 1.0            # Velocidad de reversión
    theta = v0             # Media de varianza a largo plazo
    xi = 0.4              # Volatilidad de la volatilidad
    rho = -0.7             # Correlación

    # --- Simulación Heston ---
    num_sim = 100
    S_simuladas_heston = np.zeros((num_sim, dias))
    for i in range(num_sim):
        S_path = np.zeros(dias)
        v_path = np.zeros(dias)
        S_path[0] = S0
        v_path[0] = v0
        for t in range(1, dias):
            Z1 = np.random.normal()
            Z2 = np.random.normal()
            # Brownianos correlacionados
            dW_v = Z1 * np.sqrt(dt)
            dW_S = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)
            # Actualización varianza (Euler-Maruyama, asegurando no-negatividad)
            v_path[t] = np.abs(
                v_path[t-1] + kappa*(theta - v_path[t-1])*dt + xi * np.sqrt(max(v_path[t-1], 0)) * dW_v
            )
            # Actualización precio
            S_path[t] = S_path[t-1] * np.exp(
                (mu - 0.5 * v_path[t-1]) * dt + np.sqrt(max(v_path[t-1], 0)) * dW_S
            )
        S_simuladas_heston[i] = S_path

    # --- Media simulada en cada instante de tiempo ---
    media_simulada_heston = S_simuladas_heston.mean(axis=0)

    # --- Gráfica ---
    plt.figure(figsize=(11,6))
    for i in range(num_sim):
        plt.plot(S_simuladas_heston[i], color='gray', alpha=0.05)
    plt.plot(media_simulada_heston, color='orange', lw=2.2, label='Media simulada Heston')
    plt.plot(hist.values, color='blue', lw=2, label='Precio real SPY')
    plt.title('Trayectorias simuladas (Heston) vs. precio real SPY')
    plt.xlabel('Días')
    plt.ylabel('Precio')
    plt.legend()
    plt.tight_layout()
    plt.show()


def euler_maruyama_vs_real_trayectoria():
    """
    Simula trayectorias de precios usando el método de Euler-Maruyama y las compara con los precios reales del ETF SPY.
    Encuentra la trayectoria simulada más parecida a la real (mínimo MSE) y la grafica junto con las demás simuladas.
    """

    # --- Descargar datos reales ---
    spy = yf.Ticker("SPY")
    hist = spy.history(period="1y")["Close"]
    dias = len(hist)
    S0 = hist.iloc[0]

    # Volatilidad y drift anualizados
    returns = hist.pct_change().dropna()
    sigma = returns.std() * np.sqrt(252)
    mu = returns.mean() * 252
    dt = 1/252

    # --- Simulaciones usando Euler-Maruyama ---
    num_sim = 100
    S_simuladas_em = np.zeros((num_sim, dias))
    for i in range(num_sim):
        S_path = np.zeros(dias)
        S_path[0] = S0
        for t in range(1, dias):
            Z = np.random.normal()
            S_path[t] = S_path[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        S_simuladas_em[i] = S_path

    # --- Encuentra la trayectoria más parecida a la real (mínimo MSE) ---
    hist_array = hist.values
    mse_trayectorias = np.mean((S_simuladas_em - hist_array)**2, axis=1)
    idx_mas_parecida = np.argmin(mse_trayectorias)
    tray_mas_parecida = S_simuladas_em[idx_mas_parecida]
    mse_min = mse_trayectorias[idx_mas_parecida]

    # --- Gráfica ---
    plt.figure(figsize=(11,6))
    for i in range(num_sim):
        plt.plot(S_simuladas_em[i], color='gray', alpha=0.05)
    plt.plot(tray_mas_parecida, color='red', lw=2, label='Trayectoria simulada más parecida')
    plt.plot(hist.values, color='blue', lw=2, label='Precio real SPY')
    plt.title(f'Euler-Maruyama: Trayectorias simuladas vs. precio real SPY\n'
              f'MSE tray. más parecida: {mse_min:.2f}')
    plt.xlabel('Días')
    plt.ylabel('Precio')
    plt.legend()
    plt.tight_layout()
    plt.show()


def milstein_vs_real_trayectoria():
    """
    Simula trayectorias de precios usando el método de Milstein y las compara con los precios reales del ETF SPY.
    Encuentra la trayectoria simulada más parecida a la real (mínimo MSE) y la grafica junto con las demás simuladas.
    """

    # --- Descargar datos reales ---
    spy = yf.Ticker("SPY")
    hist = spy.history(period="1y")["Close"]
    dias = len(hist)
    S0 = hist.iloc[0]

    # Volatilidad y drift anualizados
    returns = hist.pct_change().dropna()
    sigma = returns.std() * np.sqrt(252)
    mu = returns.mean() * 252
    dt = 1/252

    # --- Simulaciones usando Milstein ---
    num_sim = 100
    S_simuladas_mil = np.zeros((num_sim, dias))
    for i in range(num_sim):
        S_path = np.zeros(dias)
        S_path[0] = S0
        for t in range(1, dias):
            Z = np.random.normal()
            S_path[t] = S_path[t-1] * np.exp(
                (mu - 0.5 * sigma**2) * dt
                + sigma * np.sqrt(dt) * Z
                + 0.5 * sigma**2 * dt * (Z**2 - 1)
            )
        S_simuladas_mil[i] = S_path

    # --- Encuentra la trayectoria más parecida a la real (mínimo MSE) ---
    hist_array = hist.values
    mse_trayectorias = np.mean((S_simuladas_mil - hist_array)**2, axis=1)
    idx_mas_parecida = np.argmin(mse_trayectorias)
    tray_mas_parecida = S_simuladas_mil[idx_mas_parecida]
    mse_min = mse_trayectorias[idx_mas_parecida]

    # --- Gráfica ---
    plt.figure(figsize=(11,6))
    for i in range(num_sim):
        plt.plot(S_simuladas_mil[i], color='gray', alpha=0.05)
    plt.plot(tray_mas_parecida, color='red', lw=2, label='Trayectoria simulada más parecida')
    plt.plot(hist.values, color='blue', lw=2, label='Precio real SPY')
    plt.title(f'Milstein: Trayectorias simuladas vs. precio real SPY\n'
              f'MSE tray. más parecida: {mse_min:.2f}')
    plt.xlabel('Días')
    plt.ylabel('Precio')
    plt.legend()
    plt.tight_layout()
    plt.show()


def black_scholes_vs_real_trayectoria():
    """
    Simula trayectorias de precios usando el modelo Black-Scholes y las compara con los precios reales del ETF SPY.
    Encuentra la trayectoria simulada más parecida a la real (mínimo MSE) y la grafica junto con las demás simuladas.
    """

    # Descarga precios históricos del ETF SPY - 1 año
    spy = yf.Ticker("SPY")
    year = 1
    hist = spy.history(period=str(year)+"y")["Close"]
    dias = len(hist)
    S0 = hist.iloc[0]
    dias_negocio = year * 252  # Asumiendo 252 días de negocio al año

    # Volatilidad y drift anualizados
    returns = hist.pct_change().dropna()
    sigma = returns.std() * np.sqrt(dias_negocio)
    mu = returns.mean() * dias_negocio
    dt = 1/dias_negocio  # Paso de tiempo diario

    # Simulaciones Black-Scholes
    num_sim = 100
    S_simuladas = np.zeros((num_sim, dias))
    for i in range(num_sim):
        W = np.cumsum(np.random.normal(0, np.sqrt(dt), size=dias-1))
        S_sim = S0 * np.exp((mu - 0.5 * sigma**2) * np.linspace(0, 1, dias-1) + sigma * W)
        S_simuladas[i, 0] = S0
        S_simuladas[i, 1:] = S_sim

    # Encuentra la trayectoria simulada más parecida a la real (mínimo MSE)
    hist_array = hist.values
    mse_trayectorias = np.mean((S_simuladas - hist_array)**2, axis=1)
    idx_mas_parecida = np.argmin(mse_trayectorias)
    tray_mas_parecida = S_simuladas[idx_mas_parecida]
    mse_min = mse_trayectorias[idx_mas_parecida]

    # Gráfica
    plt.figure(figsize=(11,6))
    for i in range(num_sim):
        plt.plot(S_simuladas[i], color='gray', alpha=0.05)
    plt.plot(tray_mas_parecida, color='red', lw=2, label='Trayectoria simulada más parecida')
    plt.plot(hist.values, color='blue', lw=2, label='Precio real SPY')
    plt.title(f'Black-Scholes: Trayectorias simuladas vs. precio real SPY\n'
              f'MSE tray. más parecida: {mse_min:.2f}')
    plt.xlabel('Días')
    plt.ylabel('Precio')
    plt.legend()
    plt.tight_layout()
    plt.show()


def heston_vs_real_trayectoria():
    """
    Simula trayectorias de precios usando el modelo Heston y las compara con los precios reales del ETF SPY.
    Encuentra la trayectoria simulada más parecida a la real (mínimo MSE) y la grafica junto con las demás simuladas.
    """

    # --- Descargar datos reales ---
    spy = yf.Ticker("SPY")
    hist = spy.history(period="1y")["Close"]
    dias = len(hist)
    S0 = hist.iloc[0]

    # Volatilidad y drift anualizados
    returns = hist.pct_change().dropna()
    sigma = returns.std() * np.sqrt(252)
    mu = returns.mean() * 252
    dt = 1/252

    # --- Parámetros Heston ---
    v0 = sigma**2          # Varianza inicial
    kappa = 1.0            # Velocidad de reversión
    theta = v0         # Media de varianza a largo plazo
    xi = 0.4               # Volatilidad de la volatilidad
    rho = -0.7             # Correlación

    # --- Simulación Heston ---
    num_sim = 100
    S_simuladas_heston = np.zeros((num_sim, dias))
    for i in range(num_sim):
        S_path = np.zeros(dias)
        v_path = np.zeros(dias)
        S_path[0] = S0
        v_path[0] = v0
        for t in range(1, dias):
            Z1 = np.random.normal()
            Z2 = np.random.normal()
            # Brownianos correlacionados
            dW_v = Z1 * np.sqrt(dt)
            dW_S = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)
            # Actualización varianza (Euler-Maruyama, asegurando no-negatividad)
            v_path[t] = np.abs(
                v_path[t-1] + kappa*(theta - v_path[t-1])*dt + xi * np.sqrt(max(v_path[t-1], 0)) * dW_v
            )
            # Actualización precio
            S_path[t] = S_path[t-1] * np.exp(
                (mu - 0.5 * v_path[t-1]) * dt + np.sqrt(max(v_path[t-1], 0)) * dW_S
            )
        S_simuladas_heston[i] = S_path

    # --- Encuentra la trayectoria más parecida a la real (mínimo MSE) ---
    hist_array = hist.values
    mse_trayectorias = np.mean((S_simuladas_heston - hist_array)**2, axis=1)
    idx_mas_parecida = np.argmin(mse_trayectorias)
    tray_mas_parecida = S_simuladas_heston[idx_mas_parecida]
    mse_min = mse_trayectorias[idx_mas_parecida]

    # --- Gráfica ---
    plt.figure(figsize=(11,6))
    for i in range(num_sim):
        plt.plot(S_simuladas_heston[i], color='gray', alpha=0.05)
    plt.plot(tray_mas_parecida, color='red', lw=2, label='Trayectoria simulada más parecida')
    plt.plot(hist.values, color='blue', lw=2, label='Precio real SPY')
    plt.title(f'Heston: Trayectorias simuladas vs. precio real SPY\n'
              f'MSE tray. más parecida: {mse_min:.2f}')
    plt.xlabel('Días')
    plt.ylabel('Precio')
    plt.legend()
    plt.tight_layout()
    plt.show()


def black_scholes_vs_real_splines():
    """
    Simula trayectorias de precios usando el modelo Black-Scholes y las compara con los precios reales del ETF SPY.
    Utiliza el método de Monte Carlo para generar trayectorias, calcula la media funcional de las simulaciones y aplica una interpolación spline cúbica.
    Gráfica la media de las trayectorias simuladas y la trayectoria real del ETF SPY.
    """

    # Descarga precios históricos del ETF SPY - 1 año
    spy = yf.Ticker("SPY")
    year = 1
    hist = spy.history(period=str(year)+"y")["Close"]
    dias = len(hist)
    S0 = hist.iloc[0]
    dias_negocio = year * 252  # Asumiendo 252 días de negocio al año

    # Volatilidad y drift anualizados
    returns = hist.pct_change().dropna()
    sigma = returns.std() * np.sqrt(dias_negocio)
    mu = returns.mean() * dias_negocio
    dt = 1/dias_negocio  # Paso de tiempo diario

    # Simulaciones Black-Scholes
    num_sim = 50
    S_simuladas = np.zeros((num_sim, dias))
    for i in range(num_sim):
        W = np.cumsum(np.random.normal(0, np.sqrt(dt), size=dias-1))
        S_sim = S0 * np.exp((mu - 0.5 * sigma**2) * np.linspace(0, 1, dias-1) + sigma * W)
        S_simuladas[i, 0] = S0
        S_simuladas[i, 1:] = S_sim

    # Media punto a punto
    media_simulada = S_simuladas.mean(axis=0)
    x = np.arange(dias)

    # Interpolación spline cúbica sobre la media simulada
    from scipy.interpolate import make_interp_spline
    x_smooth = np.linspace(0, dias-1, 10*dias)
    spline_media = make_interp_spline(x, media_simulada, k=3)
    media_funcional = spline_media(x_smooth)

    # Gráfica
    plt.figure(figsize=(11,6))
    for i in range(num_sim):
        plt.plot(x, S_simuladas[i], color='gray', alpha=0.05)
    plt.plot(x_smooth, media_funcional, color='orange', lw=2.2, label='Media funcional Black-Scholes (spline)')
    plt.plot(x, hist.values, color='blue', lw=2, label='Precio real SPY')
    plt.title('Trayectorias simuladas vs. precio real SPY')
    plt.xlabel('Días')
    plt.ylabel('Precio')
    plt.legend()
    plt.tight_layout()
    plt.show()


def euler_maruyama_vs_real_splines():
    """
    Simula trayectorias de precios usando el método de Euler-Maruyama y las compara con los precios reales del ETF SPY.
    Utiliza el método de Euler-Maruyama para generar trayectorias, calcula la media funcional de las simulaciones y aplica una interpolación spline cúbica.
    Gráfica la media de las trayectorias simuladas y la trayectoria real del ETF SPY.
    """

    # --- Descargar datos reales ---
    spy = yf.Ticker("SPY")
    hist = spy.history(period="1y")["Close"]
    dias = len(hist)
    S0 = hist.iloc[0]

    # Volatilidad y drift anualizados
    returns = hist.pct_change().dropna()
    sigma = returns.std() * np.sqrt(252)
    mu = returns.mean() * 252
    dt = 1/252

    # --- Simulaciones usando Euler-Maruyama ---
    num_sim = 50
    S_simuladas_em = np.zeros((num_sim, dias))
    for i in range(num_sim):
        S_path = np.zeros(dias)
        S_path[0] = S0
        for t in range(1, dias):
            Z = np.random.normal()
            S_path[t] = S_path[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        S_simuladas_em[i] = S_path

    # --- Media punto a punto ---
    media_simulada_em = S_simuladas_em.mean(axis=0)
    x = np.arange(dias)

    # Interpolación spline cúbica sobre la media simulada
    x_smooth = np.linspace(0, dias-1, 10*dias)
    spline_media = make_interp_spline(x, media_simulada_em, k=3)
    media_funcional = spline_media(x_smooth)

    # --- Gráfica ---
    plt.figure(figsize=(11,6))
    for i in range(num_sim):
        plt.plot(x, S_simuladas_em[i], color='gray', alpha=0.05)
    plt.plot(x_smooth, media_funcional, color='orange', lw=2.2, label='Media funcional Euler-Maruyama (spline)')
    plt.plot(x, hist.values, color='blue', lw=2, label='Precio real SPY')
    plt.title('Trayectorias simuladas (Euler-Maruyama) vs. precio real SPY')
    plt.xlabel('Días')
    plt.ylabel('Precio')
    plt.legend()
    plt.tight_layout()
    plt.show()


def heston_vs_real_splines():
    """
    Simula trayectorias de precios usando el modelo Heston y las compara con los precios reales del ETF SPY.
    Utiliza el modelo Heston para generar trayectorias, calcula la media funcional de las simulaciones y aplica una interpolación spline cúbica.
    Gráfica la media de las trayectorias simuladas y la trayectoria real del ETF SPY.
    """

    # --- Descargar datos reales ---
    spy = yf.Ticker("SPY")
    hist = spy.history(period="1y")["Close"]
    dias = len(hist)
    S0 = hist.iloc[0]

    # Volatilidad y drift anualizados
    returns = hist.pct_change().dropna()
    sigma = returns.std() * np.sqrt(252)
    mu = returns.mean() * 252
    dt = 1/252

    # --- Parámetros Heston ---
    v0 = sigma**2          # Varianza inicial
    kappa = 1.0            # Velocidad de reversión
    theta = v0             # Media de varianza a largo plazo
    xi = 0.4               # Volatilidad de la volatilidad
    rho = -0.7             # Correlación

    # --- Simulación Heston ---
    num_sim = 50
    S_simuladas_heston = np.zeros((num_sim, dias))
    for i in range(num_sim):
        S_path = np.zeros(dias)
        v_path = np.zeros(dias)
        S_path[0] = S0
        v_path[0] = v0
        for t in range(1, dias):
            Z1 = np.random.normal()
            Z2 = np.random.normal()
            # Brownianos correlacionados
            dW_v = Z1 * np.sqrt(dt)
            dW_S = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)
            # Actualización varianza (Euler-Maruyama, asegurando no-negatividad)
            v_path[t] = np.abs(
                v_path[t-1] + kappa*(theta - v_path[t-1])*dt + xi * np.sqrt(max(v_path[t-1], 0)) * dW_v
            )
            # Actualización precio
            S_path[t] = S_path[t-1] * np.exp(
                (mu - 0.5 * v_path[t-1]) * dt + np.sqrt(max(v_path[t-1], 0)) * dW_S
            )
        S_simuladas_heston[i] = S_path

    # --- Media punto a punto ---
    media_simulada_heston = S_simuladas_heston.mean(axis=0)
    x = np.arange(dias)

    # Interpolación spline cúbica sobre la media simulada
    x_smooth = np.linspace(0, dias-1, 10*dias)
    spline_media = make_interp_spline(x, media_simulada_heston, k=3)
    media_funcional = spline_media(x_smooth)

    # --- Gráfica ---
    plt.figure(figsize=(11,6))
    for i in range(num_sim):
        plt.plot(x, S_simuladas_heston[i], color='gray', alpha=0.05)
    plt.plot(x_smooth, media_funcional, color='orange', lw=2.2, label='Media funcional Heston (spline)')
    plt.plot(x, hist.values, color='blue', lw=2, label='Precio real SPY')
    plt.title('Trayectorias simuladas (Heston) vs. precio real SPY')
    plt.xlabel('Días')
    plt.ylabel('Precio')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # black_scholes_vs_real()
    # euler_maruyama_vs_real()
    # heston_vs_real()
    # black_scholes_vs_real_trayectoria()
    # euler_maruyama_vs_real_trayectoria()
    # heston_vs_real_trayectoria()
    milstein_vs_real_trayectoria()
    # black_scholes_vs_real_splines()
    # euler_maruyama_vs_real_splines()
    # heston_vs_real_splines()
