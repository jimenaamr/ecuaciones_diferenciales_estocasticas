
# Modelos Estocásticos para Simulación Financiera

Este repositorio contiene scripts en Python para modelar y comparar procesos financieros utilizando diferentes métodos estocásticos. Los principales modelos cubiertos incluyen el **Modelo Heston** y el de **Black-Scholes**, los métodos **Euler-Maruyama** y **Milstein**. El objetivo de estos scripts es mostrar la resolución de EDEs con distintos métodos, además de simular trayectorias de precios de acciones y comparar su precisión con datos históricos reales del ETF SPY.

## Contenidos

1. **calibrate_heston.py**: 
   - Este script realiza la calibración de los parámetros del modelo Heston utilizando datos reales del precio de SPY desde Yahoo Finance.
   - Los parámetros del modelo se optimizan para ajustar los datos históricos reales, incluyendo la volatilidad y el drift.

2. **comparacion_spy.py**:
   - Este script compara los resultados de diferentes modelos y métodos estocásticos (Euler-Maruyama,Milstein, Heston y Black-Scholes) con los datos reales de SPY.
   - Proporciona visualizaciones de las trayectorias simuladas y su comparación con los precios reales de las acciones.

3. **em_vs_bs.py**:
   - Este script compara el método Euler-Maruyama con el modelo de Black-Scholes para la simulación de precios de acciones.
   - Evalúa la precisión de ambos métodos y obtiene el error relativo de las simulaciones.

4. **em_vs_milstein.py**:
   - Este script compara el método Euler-Maruyama con el método Milstein para resolver una ecuación diferencial estocástica (EDE).
   - Grafica las soluciones simuladas de ambos métodos y muestra los errores de cada uno con la solución analítica.

5. **heston_vs_bs.py**:
   - Este script compara el modelo Heston con el modelo Black-Scholes para simular los precios del ETF SPY.
   - Calcula y muestra el error relativo entre las simulaciones de ambos modelos.

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tuusuario/nombre-del-repositorio.git
   ```

2. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```

   Las dependencias requeridas son:
   - `numpy`
   - `matplotlib`
   - `yfinance`

3. Asegúrate de tener instalada una versión de Python 3.x.

## Notas

- Los scripts utilizan datos reales de SPY obtenidos desde Yahoo Finance para calibrar y comparar los modelos.
- Los parámetros de cada modelo se optimizan para mejorar la precisión, y el error de la simulación se calcula en cada paso.
