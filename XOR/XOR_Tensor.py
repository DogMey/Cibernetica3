# Importar las librerías necesarias
import tensorflow as tf # La biblioteca principal de TensorFlow
from tensorflow import keras # Keras es la API de alto nivel integrada en TensorFlow, ideal para construir redes neuronales
from tensorflow.keras.layers import Dense # Importar el tipo de capa densa (totalmente conectada)
from tensorflow.keras.models import Sequential # Importar el modelo Sequential para construir la red capa por capa
from tensorflow.keras.optimizers import SGD # Importar el optimizador Descenso de Gradiente Estocástico (SGD)
from tensorflow.keras.losses import MeanSquaredError # Importar la función de pérdida Error Cuadrático Medio (MSE)
from tensorflow.keras.callbacks import EarlyStopping # Importar el callback para detener el entrenamiento anticipadamente
import numpy as np # Usado para manejar los datos y redondear la salida

# --- Preparación de Datos ---
# Definir los datos de entrada (entradas XOR)
# En TensorFlow/Keras, al igual que en PyTorch, los datos suelen tener la forma (batch_size, features)
# Usamos un arreglo de numpy para definir los datos fácilmente.
P = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=np.float32) # Forma (4 muestras, 2 características)

# Definir las salidas esperadas (objetivo) (salidas XOR)
# Usamos un arreglo de numpy.
T = np.array([[0],
              [1],
              [1],
              [0]], dtype=np.float32) # Forma (4 muestras, 1 salida)

# --- Definición de la Red ---
# Crear un modelo Sequential
# Este es un modelo lineal que apila capas una tras otra
model = Sequential([
    # Primera capa: la capa oculta
    # Dense es una capa totalmente conectada.
    # Tiene 3 neuronas (units=3) como en el ejemplo de MATLAB.
    # Usa la función de activación 'tanh' (equivalente a 'tansig' en MATLAB).
    # input_shape=(2,) especifica la forma de la entrada para la primera capa (2 características).
    Dense(units=3, activation='tanh', input_shape=(2,)),

    # Segunda capa: la capa de salida
    # Tiene 1 neurona (units=1) para la salida XOR.
    # Usa la función de activación 'tanh' para coincidir con el ejemplo de MATLAB.
    Dense(units=1, activation='tanh')
])

# --- Configuración para el Entrenamiento (Compilación del Modelo) ---
# Compilar el modelo: configurar el optimizador, la función de pérdida y las métricas
# Esto prepara el modelo para el entrenamiento.
model.compile(
    # Usar el optimizador Descenso de Gradiente Estocástico (SGD)
    optimizer=SGD(learning_rate=0.1), # Especificar la tasa de aprendizaje (lr)

    # Usar el Error Cuadrático Medio como función de pérdida
    # Esto es apropiado para la salida 'tanh' y el enfoque de regresión/aproximación.
    loss=MeanSquaredError(),

    # Especificar métricas para monitorear durante el entrenamiento (opcional pero útil)
    # 'mse' mostrará el valor del Error Cuadrático Medio.
    metrics=['mse']
)

# Mostrar un resumen de la arquitectura del modelo (opcional)
# print(model.summary())

# --- Parámetros de Entrenamiento ---
# Establecer el número de épocas de entrenamiento (iteraciones sobre todo el conjunto de datos)
epochs = 200 # Coincidiendo con la configuración de MATLAB

# Establecer el objetivo de rendimiento (detener el entrenamiento cuando la pérdida esté por debajo de este valor)
goal = 1e-9 # Coincidiendo con la configuración de MATLAB

# Configurar un callback para detener el entrenamiento anticipadamente si se alcanza el objetivo de pérdida
early_stopping = EarlyStopping(
    monitor='loss', # Monitorear el valor de la pérdida
    min_delta=0, # Detener si la pérdida mejora en menos de esta cantidad (0 para detenerse exactamente en el objetivo)
    patience=0, # Número de épocas sin mejora después del cual se detiene (0 para detenerse inmediatamente si se cumple la condición)
    baseline=goal, # El entrenamiento se detendrá si la métrica monitoreada (pérdida) alcanza o supera este valor
    verbose=1 # Mostrar mensaje cuando se detenga
)

# --- Entrenamiento del Modelo ---
# Entrenar el modelo usando los datos de entrada (P) y los datos objetivo (T)
# fit() ejecuta el bucle de entrenamiento por nosotros.
# epochs: número de veces que iterar sobre todo el conjunto de datos.
# callbacks: lista de funciones a ejecutar en puntos específicos durante el entrenamiento (aquí para detenerse).
# verbose=1: muestra una barra de progreso y métricas por época.
print("\nEntrenando la red...")
history = model.fit(
    P, T,
    epochs=epochs,
    callbacks=[early_stopping], # Aplicar el callback de detención anticipada
    verbose=1 # Muestra el progreso del entrenamiento
)
print("Entrenamiento finalizado.")

# --- Simulación (Prueba) ---
# Después del entrenamiento, usar el modelo entrenado para hacer predicciones sobre los datos de entrada P
# predict() devuelve las salidas del modelo para las entradas dadas.
Y = model.predict(P)

# --- Mostrar Resultados ---
# Imprimir las salidas esperadas (objetivos)
print('\nSalidas esperadas (Targets):')
print(T.T) # Imprimir transpuesto para coincidir con la visualización horizontal de MATLAB/anterior

# Imprimir las salidas reales de la red (antes de redondear)
print('\nSalidas de la red (Network Outputs):')
# Transponemos Y para que coincida con el formato de visualización (1 fila, 4 columnas)
print(Y.T)

# Imprimir las salidas redondeadas de la red para una comparación más fácil con los objetivos 0/1
# Usamos la función round de numpy.
print('\nSalidas de la red redondeadas:')
# Aplicamos el redondeo de numpy a las predicciones Y, y luego las transponemos para visualización.
print(np.round(Y).T)

# Calcular y mostrar el rendimiento (Pérdida MSE final)
# La pérdida final suele ser el último valor reportado en el historial de entrenamiento.
# Opcionalmente, puedes evaluar explícitamente: loss, mse = model.evaluate(P, T, verbose=0)
final_loss = history.history['loss'][-1] # Obtener el último valor de pérdida del historial
print('\nRendimiento (Pérdida MSE Final):')
print(final_loss)