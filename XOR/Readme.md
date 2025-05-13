# Proyecto XOR con Redes Neuronales (PyTorch vs TensorFlow)

Este proyecto demuestra la implementación del clásico problema de la Puerta Lógica XOR utilizando redes neuronales Feedforward (FFNN) multicapa, comparando dos de las librerías de Deep Learning más populares: PyTorch y TensorFlow/Keras.

El problema XOR no es linealmente separable, lo que requiere al menos una capa oculta en la red neuronal para ser resuelto.

## Objetivo

El objetivo principal de este proyecto es:
* Implementar una red neuronal Feedforward simple para resolver el problema XOR.
* Mostrar cómo se implementan conceptos similares (definición de red, entrenamiento, predicción) en PyTorch y TensorFlow/Keras.
* Servir como ejemplo comparativo entre las APIs básicas de ambas librerías para una tarea fundamental.

## Características de la Red Neuronal

Ambas implementaciones utilizan una red con la siguiente arquitectura:

* **Capa de Entrada:** 2 neuronas (para las 2 entradas de XOR).
* **Capa Oculta:** 3 neuronas.
* **Capa de Salida:** 1 neurona (para la salida de XOR).
* **Función de Activación:** `tanh` en la capa oculta y en la capa de salida (equivalente a `tansig` en MATLAB, como referencia del código original).
* **Función de Pérdida:** Error Cuadrático Medio (MSE - Mean Squared Error), adecuado para este enfoque de aproximación de salida continua antes del redondeo.
* **Optimizador:** Descenso de Gradiente Estocástico (SGD - Stochastic Gradient Descent).

## Requisitos

Para ejecutar este código, necesitas tener instalado:

* **Python:** Se recomienda una versión **3.9, 3.10 o 3.11**. TensorFlow y PyTorch tienen requisitos de versión de Python; versiones muy recientes (como 3.13 al momento de escribir esto) o muy antiguas pueden no ser compatibles con las últimas versiones de las librerías.
* **PyTorch:** La librería PyTorch.
* **TensorFlow:** La librería TensorFlow (incluye Keras).

Se recomienda encarecidamente el uso de **entornos virtuales** para gestionar las dependencias de manera aislada para cada proyecto.

## Instalación

Sigue estos pasos para configurar tu entorno.

1.  **Clonar el Repositorio (Opcional):** Si tu código está en un repositorio (como Git), clónalo:
    ```bash
    git clone <url_del_repositorio>
    cd <nombre_de_la_carpeta>
    ```
    Si solo tienes los archivos `.py`, asegúrate de estar en la carpeta donde se encuentran.

2.  **Crear un Entorno Virtual (Recomendado):**
    Abre tu terminal en la carpeta del proyecto. Ejecuta el siguiente comando. Asegúrate de que el comando `python` o `py -x.x` apunte a una versión compatible (3.9, 3.10, 3.11). Si tienes varias versiones y `python` no es la correcta, usa `py -3.11 -m venv .venv` (cambia `3.11` por tu versión compatible).
    ```bash
    python -m venv .venv
    ```
    Esto creará una carpeta `.venv` con el entorno virtual.

3.  **Activar el Entorno Virtual:**
    * **En Windows (CMD o PowerShell):**
        ```bash
        .\.venv\Scripts\activate
        ```
    * **En macOS o Linux (Bash/Zsh):**
        ```bash
        source .venv/bin/activate
        ```
    Verás el nombre del entorno virtual (ej: `(.venv)`) al principio de tu prompt, indicando que está activo.

4.  **Instalar las Librerías Necesarias:**
    Con el entorno virtual **activo**, instala PyTorch y TensorFlow:
    ```bash
    pip install torch tensorflow numpy
    ```
    (Incluimos `numpy` aunque a menudo se instala como dependencia, es explícito).

**Nota para usuarios de Windows sin Entorno Virtual (No Recomendado):**
Si por alguna razón no puedes o no quieres usar un entorno virtual y sabes que tu versión de Python 3.11.x está instalada pero no en el PATH global (como nos pasó en la conversación), puedes intentar instalar directamente usando el lanzador de Python:
```bash
py -3.11 -m pip install torch tensorflow numpy