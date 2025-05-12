# Se carga el archivo .mat
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv # Importamos la pseudoinversa
from sklearn.linear_model import Lasso # Importamos Lasso

# Carga los datos de los archivos .mat
cat_data = loadmat('catData_w.mat')
dog_data = loadmat('dogData_w.mat')

# Extrae las variables 'cat_wave' y 'dog_wave' de los diccionarios cargados
cat_wave = cat_data['cat_wave']
dog_wave = dog_data['dog_wave']

# Combina los datos de perros y gatos horizontalmente (axis=1) para crear un conjunto completo (no usado directamente después, pero útil para entender la estructura)
CD = np.hstack((dog_wave, cat_wave))

# Divide los datos en conjuntos de entrenamiento y prueba
# El conjunto de entrenamiento toma las primeras 60 imagenes de entrenamiento
train = np.hstack((dog_wave[:, :60], cat_wave[:, :60]))
# El conjunto de prueba toma las columnas 60 a 79 (20 imágenes) de evaluación
test = np.hstack((dog_wave[:, 60:80], cat_wave[:, 60:80]))

# Crea las etiquetas para el conjunto de entrenamiento
# 1 para las primeras 60 muestras (perros) y -1 para las siguientes 60 muestras (gatos)
labels = np.concatenate((np.ones(60), -1 * np.ones(60)))

# --- Primera parte del código: Clasificación Lineal usando Pseudoinversa ---

# Calcula el vector de pesos A usando la pseudoinversa de la matriz de entrenamiento.
A = labels @ pinv(train) # Esto resuelve el sistema lineal A @ train = labels en el sentido de mínimos cuadrados.
# Predice las etiquetas para el conjunto de prueba usando el vector de pesos A.
test_labels = np.sign(A @ test)# np.sign(x) devuelve 1 si x es positivo, -1 si es negativo y 0 si es cero.

plt.figure(1) # Crea la figura 1 que usaremos para mostrar los resultados de clasificación y los pesos.
plt.subplot(4, 1, 1) # Define una cuadrícula de 4 filas, 1 columna y selecciona la primera celda.
plt.bar(np.arange(len(test_labels)), test_labels, color=[0.6, 0.6, 0.6], edgecolor='k') # Crea un gráfico de barras con los resultados.
plt.axis('off') # Desactiva los ejes.
plt.subplot(4, 1, 2) # Selecciona la segunda celda.
plt.bar(np.arange(len(A)), A, color=[0.6, 0.6, 0.6], edgecolor='k') # Crea un gráfico de barras con los pesos A.
plt.axis([0, 1024, -0.002, 0.002]) # Establece los límites de los ejes [xmin, xmax, ymin, ymax]. Los datos tienen 1024 características (32x32).
plt.axis('off') # Desactiva los ejes.

plt.figure(2)# Crea la figura 2 para visualizar el vector de pesos como una imagen.
plt.subplot(2, 2, 1) # Define una cuadrícula de 2 filas, 2 columnas y selecciona la primera celda.
A2 = np.flipud(A.reshape(32, 32).T) # Redimensiona el vector A a una matriz de 32x32 y la voltea verticalmente (para corregir la orientación).
plt.pcolormesh(A2, cmap='gray') # Crea un mapa de colores pseudocartográfico (heatmap) en escala de grises.
plt.axis('off') # Desactiva los ejes.

# --- Segunda parte del código (Lasso) ---

lasso = Lasso(alpha=0.1)# Inicializa un modelo de regresión Lasso con un parámetro de regularización alpha=0.1.
lasso.fit(train.T, labels.T) # Entrena el modelo Lasso. .fit() espera que los datos de entrenamiento estén en filas y las etiquetas en columnas; por lo que se transpone 'train'.
# Extrae los coeficientes del modelo Lasso. Estos coeficientes son el vector de pesos A_lasso.
A_lasso = lasso.coef_.T # Se transpone para que tenga la misma forma que el vector A anterior.
# Predice las etiquetas para el conjunto de prueba (Los 20) usando el vector de pesos A_lasso.
test_labels_lasso = np.sign(A_lasso.T @ test) # Se transpone A_lasso para la multiplicación matricial.

plt.figure(1) # Continúa en la figura 1 para añadir los resultados de Lasso.
plt.subplot(4, 1, 3) # Selecciona la tercera celda.
plt.bar(np.arange(len(test_labels_lasso)), test_labels_lasso, color=[0.6, 0.6, 0.6], edgecolor='k') # Gráfico de barras de 1 y -1.
plt.axis('off') # Desactiva los ejes.
plt.subplot(4, 1, 4) # Selecciona la cuarta celda.
plt.bar(np.arange(len(A_lasso)), A_lasso, color=[0.6, 0.6, 0.6], edgecolor='k') # Gráfico de barras con los pesos A_lasso.
plt.axis([0, 1024, -0.008, 0.008]) # Establece los límites de los ejes.
plt.axis('off') # Desactiva los ejes.

plt.figure(2)# Continúa en la figura 2 para añadir la visualización del vector de pesos Lasso.
plt.subplot(2, 2, 2) # Selecciona la segunda celda.
A2_lasso = np.flipud(A_lasso.reshape(32, 32).T) # Redimensiona y voltea el vector A_lasso.
plt.pcolormesh(A2_lasso, cmap='gray') # Crea el heatmap en escala de grises.
plt.axis('off') # Desactiva los ejes.

plt.show()# Muestra todas las figuras creadas.