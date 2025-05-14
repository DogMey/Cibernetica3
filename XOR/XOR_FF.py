# Importar librerías necesarias de PyTorch
import torch
import torch.nn as nn # Proporciona módulos para redes neuronales como capas y funciones de activación
import torch.optim as optim # Proporciona algoritmos de optimización como SGD
import numpy as np # Usado para redondear la salida para la comparación de visualización

# --- Preparación de Datos ---
# Definir los datos de entrada (entradas XOR)
P = torch.tensor([[0, 0, 1, 1],
                  [0, 1, 0, 1]], dtype=torch.float32).T # Transponer para obtener la forma (4, 2)

# Definir las salidas esperadas (objetivo) (salidas XOR)
T = torch.tensor([[0, 1, 1, 0]], dtype=torch.float32).T # Transponer para obtener la forma (4, 1)

# --- Definición de la Red ---
# Definir una clase para la red neuronal
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__() # Llamar al constructor de la clase padre (nn.Module)

        # Definir la primera capa lineal (capa de entrada a capa oculta)
        self.fc1 = nn.Linear(in_features=2, out_features=3)

        # Definir la segunda capa lineal (capa oculta a capa de salida)
        self.fc2 = nn.Linear(in_features=3, out_features=1)

        self.tanh = nn.Tanh() # Definir la función de activación para la capa oculta
        self.output_activation = nn.Tanh() #Aplicamos tanh para que la salida este enre -1 y 1

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

criterion = nn.MSELoss() # Definir la función de pérdida

optimizer = optim.SGD(model.parameters(), lr=0.1) # Definir el optimizador descenso del gradiente por lotes

# --- Parámetros de Entrenamiento ---
epochs = 1000 # Establecer el número de épocas de entrenamiento (iteraciones sobre todo el conjunto de datos)

goal = 1e-9 # Establecer el objetivo de rendimiento (detener el entrenamiento cuando la pérdida esté por debajo de este valor)

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

print('\nSalidas de la red (Network Outputs):')
print(Y.T) # Imprimir transpuesto

print('\nSalidas de la red redondeadas:')
print(np.round(Y.detach().numpy()).T) # Aplicar redondeo y luego transponer

print('\nRendimiento (Pérdida MSE Final):')
print(loss.item())