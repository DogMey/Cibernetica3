import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

h = 0.1 # Define el paso de la cuadrícula
# Creamos el plano tridimensional
x_range = np.arange(-6, 6 + h, h) # Crea un array con los valores de x desde -6 a 6 en pasos de 0.1
y_range = np.arange(-6, 6 + h, h) # Crea un array con los valores de y desde -6 a 6 en pasos de 0.1
n = len(x_range) # Obtiene el número de puntos en una dimensión de la cuadrícula
X, Y = np.meshgrid(x_range, y_range) # Crea las matrices X e Y para formar la cuadrícula 2D (el plano cartesiano)

# --- Define la función F(x, y) y calcula su gradiente ---

F1 = 1.5 - 1.6 * np.exp(-0.05 * (3 * (X + 3)**2 + (Y + 3)**2)) # Define el primer componente de la función
F = F1 + (0.5 - np.exp(-0.1 * (3 * (X - 3)**2 + (Y - 3)**2))) # Define la función completa F, sumando F1 y otro componente

# Calcula el gradiente numérico de F sobre toda la cuadrícula (dF/dx, dF/dy)
dFx, dFy = np.gradient(F, h, h) # Este es el gradiente de F calculado numéricamente

# --- Valores de los puntos iniciales (w_0) para evaluar y sus colores de graficación ---
x0 = [4, 0, -5] # Coordenadas x de los 3 puntos iniciales
y0 = [0, -5, 2] # Coordenadas y de los 3 puntos iniciales
colors = ['ro', 'bo', 'mo'] # Colores y marcadores para graficar las trayectorias

# --- Bucle principal de optimización ---
trajectories_x = [] # Lista para almacenar las coordenadas x de las trayectorias de cada w_0
trajectories_y = [] # Lista para almacenar las coordenadas y de las trayectorias de cada w_0
trajectories_f = [] # Lista para almacenar los valores de F a lo largo de las trayectorias

# Itera sobre cada uno de los 3 puntos iniciales definidos en x0, y0
for jj in range(3):
    # Se eligen 10 puntos para realizar proceso estocastico
    i_indices = np.sort(np.random.permutation(n)[:10])# Selecciona 10 índices de fila aleatorios y los ordena
    j_indices = np.sort(np.random.permutation(n)[:10])# Selecciona 10 índices de columna aleatorios y los ordena

    # Crea las subcuadrículas de coordenadas correspondientes a los índices seleccionados
    sub_x = x_range[i_indices]
    sub_y = y_range[j_indices]

    # Extrae los valores de F para la subcuadrícula de 10x10 definida por los índices aleatorios
    sub_F = F[np.ix_(i_indices, j_indices)] # Es una matriz de 10x10 con los valores de F en la porción aleatoriamente seleccionada.
    sub_dFx = dFx[np.ix_(i_indices, j_indices)] # Extrae los valores de dFx con las pendientes (gradientes) en la misma porción aleatoria.
    sub_dFy = dFy[np.ix_(i_indices, j_indices)] # Extrae los valores de dFy para la subcuadrícula de 10x10

    # Estas funciones interpoladoras que estiman F, dFx, dFy en puntos arbitrarios dentro de la región muestreada.
    interp_f = RegularGridInterpolator((sub_x, sub_y), sub_F, bounds_error=False, fill_value=None)
    interp_dfx = RegularGridInterpolator((sub_x, sub_y), sub_dFx, bounds_error=False, fill_value=None)
    interp_dfy = RegularGridInterpolator((sub_x, sub_y), sub_dFy, bounds_error=False, fill_value=None)

    # Inicializa las listas para la trayectoria actual con el punto de partida
    x_traj = [x0[jj]] # Inicializan las listas que guardarán las coordenadas X e Y a lo largo de la trayectoria de optimización
    y_traj = [y0[jj]]

    # Estas líneas preparan las listas de trayectoria y calculan el valor de la función y el gradiente en el punto de partida.
    f_traj = [interp_f([x_traj[-1], y_traj[-1]])[0]] # Estima F en el punto inicial usando el interpolador actual
    dfx = interp_dfx([x_traj[-1], y_traj[-1]])[0] # Estima dFx en el punto inicial usando el interpolador actual
    dfy = interp_dfy([x_traj[-1], y_traj[-1]])[0] # Estima dFy en el punto inicial usando el interpolador actual

    tau = 2 # Define el tamaño del paso (tasa de aprendizaje) para el descenso de gradiente

    # Bucle para las iteraciones de descenso de gradiente (hasta 50 pasos)
    for j in range(50):
        # --- PASO DE DESCENSO DE GRADIENTE ---
        x_next = x_traj[-1] - tau * dfx # Actualiza x e y usando la regla de descenso de gradiente
        y_next = y_traj[-1] - tau * dfy
        x_traj.append(x_next) # Añade el nuevo punto calculado a las trayectorias
        y_traj.append(y_next)

        # *** RE-SELECCIÓN ALEATORIA DE SUBSET Y RE-INTERPOLACIÓN ***
        i_indices = np.sort(np.random.permutation(n)[:10]) # Selecciona nuevos índices aleatoriamente para una nueva subcuadrícula en cada paso
        j_indices = np.sort(np.random.permutation(n)[:10])

        # Crea las nuevas subcuadrículas y valores de subfunciones basados en los nuevos índices aleatorios
        sub_x = x_range[i_indices]
        sub_y = y_range[j_indices]
        sub_F = F[np.ix_(i_indices, j_indices)]
        sub_dFx = dFx[np.ix_(i_indices, j_indices)]
        sub_dFy = dFy[np.ix_(i_indices, j_indices)]

        # Crea nuevas funciones interpoladoras a partir de esta *nueva* subcuadrícula aleatoria
        interp_f = RegularGridInterpolator((sub_x, sub_y), sub_F, bounds_error=False, fill_value=None)
        interp_dfx = RegularGridInterpolator((sub_x, sub_y), sub_dFx, bounds_error=False, fill_value=None)
        interp_dfy = RegularGridInterpolator((sub_x, sub_y), sub_dFy, bounds_error=False, fill_value=None)

        # Estima el valor de la función y el gradiente en el nuevo punto (x_next, y_next)
        f_next = interp_f([x_next, y_next])[0]
        f_traj.append(f_next) # Añade el valor de F estimado a la trayectoria
        dfx = interp_dfx([x_next, y_next])[0] # Estima el gradiente dFx en el nuevo punto
        dfy = interp_dfy([x_next, y_next])[0] # Estima el gradiente dFy en el nuevo punto

        # Comprueba la convergencia: si el cambio en el valor de la función es muy pequeño
        if abs(f_traj[-1] - f_traj[-2]) < 1e-6:
             break # Si converge, sale del bucle de iteraciones

    # Después del bucle de iteraciones, guarda las trayectorias completas de este punto inicial
    trajectories_x.append(x_traj)
    trajectories_y.append(y_traj)
    trajectories_f.append(f_traj)

# --- Graficación ---
plt.figure(1) # Crea la primera figura para el gráfico de contorno
contour = plt.contour(X, Y, F - 1, 10, colors='k') # Dibuja las líneas de contorno de la función F (se resta 1 a F para ajustar las etiquetas)
plt.clabel(contour, inline=True, fontsize=8, fmt='%1.1f') # Añade etiquetas a las líneas de contorno mostrando el valor de F
# Itera sobre las 3 trayectorias guardadas
for i in range(3):
    plt.plot(trajectories_x[i], trajectories_y[i], colors[i], linewidth=2) # Grafica la trayectoria (x, y) con el color y marcador especificado y línea gruesa
    plt.plot(trajectories_x[i], trajectories_y[i], 'k:', linewidth=2) # Grafica la misma trayectoria como línea punteada negra para hacerla más visible
# Configura las etiquetas de los ejes y el título
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gráfico de Contorno de F y Trayectorias de Optimización')
plt.gca().tick_params(axis='both', which='major', labelsize=18) # Ajusta el tamaño de la fuente para las etiquetas de los ticks de los ejes
plt.figure(2) # Crea la segunda figura para el gráfico 3D
ax = plt.axes(projection='3d') # Configura ejes 3D

# Grafica la superficie 3D de la función F
ax.plot_surface(X, Y, F, cmap='gray', alpha=0.7) # usa mapa de color gris y transparencia
# Itera sobre las 3 trayectorias guardadas
for i in range(3):
    # Grafica la trayectoria en 3D (x, y, F). Se suma 0.1 a F para levantar la línea de color ligeramente sobre la superficie
    ax.plot3D(trajectories_x[i], trajectories_y[i], np.array(trajectories_f[i]) + 0.1, colors[i][0], linewidth=2)
    # Grafica la trayectoria en 3D como línea punteada negra, directamente sobre la superficie
    ax.plot3D(trajectories_x[i], trajectories_y[i], trajectories_f[i], 'k:', linewidth=2)
# Configura las etiquetas de los ejes 3D
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('F')
# Configura los límites de los ejes para la vista
ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
# Configura el ángulo de visión de la gráfica 3D
ax.view_init(elev=60, azim=-25) # elevación 60 grados, azimut -25 grados
# Ajusta el tamaño de la fuente para las etiquetas de los ticks de los ejes 3D
ax.tick_params(axis='both', which='major', labelsize=18)
plt.title('Gráfico de Superficie de F y Trayectorias de Optimización') # Título para el gráfico 3D

plt.show() # Muestra todas las figuras creadas