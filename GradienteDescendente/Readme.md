# Simulación de Descenso de Gradiente con Estimación Local Aleatoria

Este script de Python simula el proceso de descenso de gradiente aplicado a una función bidimensional compleja. La particularidad de esta implementación radica en que, en cada paso de la optimización, el gradiente se estima mediante interpolación a partir de un **subconjunto aleatorio** de puntos de una cuadrícula precalculada, en lugar de usar el gradiente exacto o la interpolación sobre toda la cuadrícula. Esto simula escenarios donde la información completa de la función o su gradiente no está disponible globalmente en cada iteración.

## Descripción

El código define una función `F(x, y)` con una estructura que presenta múltiples características (picos o valles). Calcula el gradiente numérico de esta función sobre una cuadrícula densa. Luego, ejecuta un proceso de optimización tipo descenso de gradiente comenzando desde tres puntos iniciales diferentes. En cada paso del descenso, el valor actual de la función y la dirección del gradiente se "adivinan" (estiman) usando interpoladores (`scipy.interpolate.RegularGridInterpolator`) construidos dinámicamente a partir de una pequeña muestra aleatoria de 10x10 puntos de la cuadrícula completa. Finalmente, visualiza la forma de la función y las trayectorias seguidas por el optimizador en gráficos 2D de contorno y 3D de superficie.

## Características

* Definición y evaluación de una función 2D compleja sobre una cuadrícula.
* Cálculo del gradiente numérico de la función.
* Simulación de optimización (descenso de gradiente) desde múltiples puntos de inicio.
* Estimación estocástica del valor de la función y el gradiente en cada paso mediante interpolación sobre subconjuntos aleatorios de datos.
* Criterio de convergencia basado en el cambio del valor de la función.
* Visualización de la superficie de la función y las trayectorias de optimización en gráficos de contorno y 3D.

## Requisitos

Necesitas tener Python instalado junto con las siguientes librerías:

* `numpy`
* `matplotlib`
* `scipy`

Puedes instalar estas librerías usando `pip`:

```bash
pip install numpy matplotlib scipy