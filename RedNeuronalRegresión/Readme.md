# Proyecto de Clasificación Lineal: Perros vs Gatos

Este proyecto implementa y compara dos métodos de clasificación lineal para distinguir entre imágenes de perros y gatos: la clasificación basada en la **pseudoinversa** y la **Regresión Lasso**. El código carga datos de imágenes preprocesadas, los divide en conjuntos de entrenamiento y prueba, entrena ambos clasificadores y visualiza los resultados de la predicción, así como los vectores de pesos aprendidos por cada método.

## Descripción del Proyecto

El objetivo principal es demostrar cómo se pueden aplicar técnicas de clasificación lineal a datos de imágenes (representadas como vectores de características) y visualizar los patrones (pesos) que cada clasificador encuentra como importantes para realizar la distinción entre las dos clases (perros y gatos).

Se utilizan datos de imágenes de 32x32 píxeles que han sido aplanadas en vectores de 1024 elementos.

## Características

* Carga de datos de imágenes desde archivos `.mat`.
* División de datos en conjuntos de entrenamiento y prueba.
* Implementación de un clasificador lineal usando la pseudoinversa.
* Implementación de un clasificador lineal usando Regresión Lasso con regularización L1.
* Predicción de etiquetas para el conjunto de prueba con ambos métodos.
* Visualización de las predicciones en barras.
* Visualización de los vectores de pesos (coeficientes) de cada clasificador en barras.
* Visualización de los vectores de pesos como imágenes en escala de grises para interpretar los patrones aprendidos.

## Requisitos

Asegúrate de tener Python instalado en tu sistema. Necesitarás las siguientes librerías:

* `numpy`
* `scipy`
* `matplotlib`
* `scikit-learn`

Puedes instalarlas usando pip:

```bash
pip install numpy scipy matplotlib scikit-learn