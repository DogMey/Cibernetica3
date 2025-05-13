# Importar librerías necesarias de PyTorch
import torch
import torch.nn as nn # Proporciona módulos para redes neuronales como capas y funciones de activación
import torch.optim as optim # Proporciona algoritmos de optimización como SGD
import numpy as np # Usado para redondear la salida para la comparación de visualización

# --- Preparación de Datos ---
# Definir los datos de entrada (entradas XOR)
# En PyTorch, los datos suelen tener la forma (batch_size, features)
# P en MATLAB es [features, batch_size], por lo que la transponemos.
P = torch.tensor([[0, 0, 1, 1],
                  [0, 1, 0, 1]], dtype=torch.float32).T # Transponer para obtener la forma (4, 2)
# P ahora se ve así:
# [[0., 0.],
#  [0., 1.],
#  [1., 0.],
#  [1., 1.]]

# Definir las salidas esperadas (objetivo) (salidas XOR)
# Transponer T de forma similar a P
T = torch.tensor([[0, 1, 1, 0]], dtype=torch.float32).T # Transponer para obtener la forma (4, 1)
# T ahora se ve así:
# [[0.],
#  [1.],
#  [1.],
#  [0.]]

# --- Definición de la Red ---
# Definir una clase para la red neuronal
# Hereda de nn.Module, la clase base para todos los módulos de redes neuronales en PyTorch
class XORNet(nn.Module):
    # Constructor de la clase
    def __init__(self):
        # Llamar al constructor de la clase padre (nn.Module)
        super(XORNet, self).__init__()

        # Definir la primera capa lineal (capa de entrada a capa oculta)
        # Toma 2 características de entrada (de P) y produce 3 características de salida (como en [3 1] de MATLAB)
        self.fc1 = nn.Linear(in_features=2, out_features=3)

        # Definir la segunda capa lineal (capa oculta a capa de salida)
        # Toma 3 características de entrada (de la capa oculta) y produce 1 característica de salida (como en [3 1] de MATLAB)
        self.fc2 = nn.Linear(in_features=3, out_features=1)

        # Definir la función de activación para la capa oculta
        # MATLAB usa 'tansig', que es equivalente a Tanh en PyTorch
        self.tanh = nn.Tanh()

        # Aunque MATLAB especificó 'tansig' también para la capa de salida,
        # es práctica común para regresión/pérdida MSE no tener activación o tener una lineal en la salida.
        # Sin embargo, para coincidir exactamente con la estructura de MATLAB que *sí* aplicó tanh en la salida,
        # la aplicaremos aquí también. Esto significa que la salida estará entre -1 y 1, necesitando redondeo
        # para la comparación con 0/1 más adelante.
        self.output_activation = nn.Tanh()

    # Definir el paso hacia adelante (forward pass) de la red
    # Este método define la secuencia de operaciones por las que pasan los datos
    def forward(self, x):
        # Pasar la entrada a través de la primera capa lineal
        x = self.fc1(x)
        # Aplicar la función de activación Tanh a la salida de la primera capa
        x = self.tanh(x)
        # Pasar el resultado a través de la segunda capa lineal
        x = self.fc2(x)
        # Aplicar la función de activación Tanh a la salida de la segunda capa (coincidiendo con MATLAB)
        x = self.output_activation(x)
        # Devolver la salida final de la red
        return x

# --- Inicialización y Configuración de la Red ---
# Crear una instancia del modelo de red
model = XORNet()

# Definir la función de pérdida
# 'perform' de MATLAB para redes feedforward a menudo implica el Error Cuadrático Medio (MSE - Mean Squared Error)
# especialmente cuando se trata de salidas continuas como las de tansig antes del redondeo.
# MSELoss es apropiado aquí ya que estamos comparando salidas float con objetivos float.
criterion = nn.MSELoss()

# Definir el optimizador
# 'traingd' de MATLAB es descenso de gradiente por lotes (batch gradient descent).
# En PyTorch, usamos optim.SGD con una tasa de aprendizaje.
# Pasamos los parámetros del modelo al optimizador para que sepa qué actualizar.
# Se necesita establecer una tasa de aprendizaje (lr); 0.1 es un punto de partida común.
optimizer = optim.SGD(model.parameters(), lr=0.1) # Puede que necesites ajustar la tasa de aprendizaje

# --- Parámetros de Entrenamiento ---
# Establecer el número de épocas de entrenamiento (iteraciones sobre todo el conjunto de datos)
epochs = 1000 # Coincidiendo con la configuración de MATLAB

# Establecer el objetivo de rendimiento (detener el entrenamiento cuando la pérdida esté por debajo de este valor)
goal = 1e-9 # Coincidiendo con la configuración de MATLAB

# --- Bucle de Entrenamiento ---
# Bucle para el número de épocas especificado
for epoch in range(epochs):
    # Realizar un paso hacia adelante (forward pass): obtener predicciones del modelo
    outputs = model(P)

    # Calcular la pérdida entre las salidas del modelo y los valores objetivo
    loss = criterion(outputs, T)

    # Verificar si la pérdida ha alcanzado el objetivo, y romper el bucle si es así
    if loss.item() < goal: # .item() obtiene el valor escalar del tensor de pérdida
        print(f'Objetivo alcanzado en la época {epoch+1}')
        break # Salir del bucle de entrenamiento

    # Poner a cero los gradientes antes del paso hacia atrás
    # PyTorch acumula gradientes por defecto, así que los limpiamos antes de calcular los nuevos
    optimizer.zero_grad()

    # Realizar un paso hacia atrás (backward pass): calcular los gradientes de la pérdida con respecto a los parámetros del modelo
    loss.backward()

    # Realizar un solo paso de optimización: actualizar los parámetros del modelo basándose en los gradientes
    optimizer.step()

    # Imprimir la pérdida cada 100 épocas para monitorear el progreso (opcional)
    if (epoch + 1) % 100 == 0:
        print(f'Época [{epoch+1}/{epochs}], Pérdida: {loss.item():.9f}')

# --- Simulación (Prueba) ---
# Después del entrenamiento, usar el modelo entrenado para hacer predicciones sobre los datos de entrada P
# No necesitamos calcular gradientes aquí, por lo que envolvemos esto en torch.no_grad()
with torch.no_grad():
    # Obtener las salidas finales de la red entrenada
    Y = model(P)

# --- Mostrar Resultados ---
# Imprimir las salidas esperadas (objetivos)
print('\nSalidas esperadas (Targets):')
print(T.T) # Imprimir transpuesto para coincidir con la visualización horizontal de MATLAB

# Imprimir las salidas reales de la red
print('\nSalidas de la red (Network Outputs):')
print(Y.T) # Imprimir transpuesto

# Imprimir las salidas redondeadas de la red para una comparación más fácil con los objetivos 0/1
# Usar la función round de numpy para un redondeo simple
# .detach().numpy() convierte el tensor de PyTorch a un arreglo de numpy
print('\nSalidas de la red redondeadas:')
print(np.round(Y.detach().numpy()).T) # Aplicar redondeo y luego transponer

# Imprimir el valor de pérdida final (rendimiento)
# Este es el MSE entre la salida de la red y el objetivo
print('\nRendimiento (Pérdida MSE Final):')
print(loss.item())