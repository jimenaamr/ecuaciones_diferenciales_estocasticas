import numpy as np
import matplotlib.pyplot as plt

def euler_maruyama(lambda_, mu, Xzero, dW, W):
    """
    Método de Euler-Maruyama para dX = lambda * X dt + mu * X dW.
    Utiliza un camino Browniano compartido para comparar con la solución exacta.

    Parámetros:
    - lambda_: Tasa de crecimiento
    - mu: Volatilidad
    - Xzero: Valor inicial de X
    - dW: Incrementos del camino Browniano
    - W: Camino Browniano acumulado

    Retorna:
    - t: Vector de tiempos
    - Xtrue: Solución exacta
    - Xem: Aproximación de Euler-Maruyama
    - emerr: Error final de Euler-Maruyama
    """

    T = 1
    N = len(dW)
    R = 4
    dt = T / N
    Dt = R * dt
    L = N // R

    # Solución exacta
    t = np.linspace(0, T, N)
    Xtrue = Xzero * np.exp((lambda_ - 0.5 * mu**2) * t + mu * W)

    # Euler-Maruyama
    Xem = np.zeros(L)
    Xtemp = Xzero
    for j in range(L):
        Winc = np.sum(dW[R * j : R * (j + 1)])
        Xtemp = Xtemp + Dt * lambda_ * Xtemp + mu * Xtemp * Winc
        Xem[j] = Xtemp

    # Gráfica
    plt.plot(t, Xtrue, 'b', label="Solución exacta")
    plt.plot(np.linspace(0, T, L), Xem, 'r*', label="Euler-Maruyama")
    plt.xlabel('t', fontsize=12)
    plt.ylabel('X', fontsize=12)
    plt.title(f'Euler-Maruyama con λ = {lambda_}, μ = {mu}', fontsize=14)
    plt.legend()
    plt.show()

    emerr = abs(Xem[-1] - Xtrue[-1])
    return t, Xtrue, Xem, emerr


def milstein(lambda_, mu, Xzero, dW, W):
    """
    Método de Milstein para dX = lambda * X dt + mu * X dW.
    En este caso, es diferente de Euler-Maruyama porque incluye un término adicional
    que mejora la precisión al considerar la variación cuadrática del proceso.

    Parámetros:
    - lambda_: Tasa de crecimiento
    - mu: Volatilidad
    - Xzero: Valor inicial de X
    - dW: Incrementos del camino Browniano
    - W: Camino Browniano acumulado

    Retorna:
    - t: Vector de tiempos
    - Xtrue: Solución exacta
    - Xmil: Aproximación de Milstein
    - emerr: Error final de Milstein
    """

    T = 1
    N = len(dW)
    R = 4
    dt = T / N
    Dt = R * dt
    L = N // R

    # Solución exacta
    t = np.linspace(0, T, N)
    Xtrue = Xzero * np.exp((lambda_ - 0.5 * mu**2) * t + mu * W)

    # Milstein
    Xmil = np.zeros(L)
    Xtemp = Xzero
    for j in range(L):
        Winc = np.sum(dW[R * j : R * (j + 1)])
        Xtemp = (Xtemp
                 + Dt * lambda_ * Xtemp
                 + mu * Xtemp * Winc
                 + 0.5 * mu**2 * Xtemp * (Winc**2 - Dt))
        Xmil[j] = Xtemp

    # Gráfica
    plt.plot(t, Xtrue, 'b', label="Solución exacta")
    plt.plot(np.linspace(0, T, L), Xmil, 'g*', label="Milstein")
    plt.xlabel('t', fontsize=12)
    plt.ylabel('X', fontsize=12)
    plt.title(f'Milstein con λ = {lambda_}, μ = {mu}', fontsize=14)
    plt.legend()
    plt.show()

    emerr = abs(Xmil[-1] - Xtrue[-1])
    return t, Xtrue, Xmil, emerr


def solve_all(lambda_, mu, Xzero):
    """
    Resuelve la SDE dX = lambda * X dt + mu * X dW
    utilizando la solución exacta, Euler-Maruyama y Milstein
    sobre el mismo camino Browniano.
    """

    # Parámetros
    T = 1
    N = 2**8
    R = 4
    dt = T / N
    Dt = R * dt
    L = N // R

    # Generar camino Browniano
    dW = np.sqrt(dt) * np.random.randn(N)
    W = np.cumsum(dW)

    # Tiempo fino y grueso
    t = np.linspace(0, T, N)
    t_coarse = np.linspace(0, T, L)

    # Solución exacta
    Xtrue = Xzero * np.exp((lambda_ - 0.5 * mu**2) * t + mu * W)

    # Euler-Maruyama
    Xem = np.zeros(L)
    Xtemp_em = Xzero
    for j in range(L):
        Winc = np.sum(dW[R*j : R*(j+1)])
        Xtemp_em += Dt * lambda_ * Xtemp_em + mu * Xtemp_em * Winc
        Xem[j] = Xtemp_em

    # Milstein
    Xmil = np.zeros(L)
    Xtemp_mil = Xzero
    for j in range(L):
        Winc = np.sum(dW[R*j : R*(j+1)])
        Xtemp_mil += Dt * lambda_ * Xtemp_mil \
                     + mu * Xtemp_mil * Winc \
                     + 0.5 * mu**2 * Xtemp_mil * (Winc**2 - Dt)
        Xmil[j] = Xtemp_mil

    # GRAFICAR TODO JUNTO
    plt.figure(figsize=(10, 6))
    plt.plot(t, Xtrue, 'b-', label="Solución exacta")
    plt.plot(t_coarse, Xem, 'r-', label="Euler-Maruyama")
    plt.plot(t_coarse, Xmil, 'g-', label="Milstein")
    plt.xlabel('t', fontsize=12)
    plt.ylabel('X', fontsize=12)
    plt.title(f'Comparación de métodos con λ = {lambda_}, μ = {mu}', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Errores finales
    emerr = abs(Xem[-1] - Xtrue[-1])
    milerr = abs(Xmil[-1] - Xtrue[-1])

    print(f"Error final Euler-Maruyama: {emerr}")
    print(f"Error final Milstein: {milerr}")
    print(f"Valor final exacto: {Xtrue[-1]}")
    print(f"Valor Euler: {Xem[-1]}")
    print(f"Valor Milstein: {Xmil[-1]}")

if __name__ == "__main__":
    # Parámetros del experimento
    lambda_ = 0
    mu = 1
    Xzero = 1
    # T = 1
    # N = 2**8
    # dt = T / N

    # # Generar camino browniano compartido
    # dW = np.sqrt(dt) * np.random.randn(N)
    # W = np.cumsum(dW)

    # # Ejecutar ambos métodos con el mismo camino
    # t, Xtrue1, Xem, err_em = euler_maruyama(lambda_, mu, Xzero, dW, W)
    # t, Xtrue2, Xmil, err_mil = milstein(lambda_, mu, Xzero, dW, W)

    # print("Error final Euler-Maruyama:", err_em)
    # print("Aproximación Euler-Maruyama en el último punto:", Xem[-1])
    # print("Error final Milstein:", err_mil)
    # print("Aproximación Milstein en el último punto:", Xmil[-1])

    solve_all(lambda_, mu, Xzero)