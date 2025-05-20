# Johnny Svensson Attractor
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Definimos el atractor de Johnny Svensson (modelo dinámico 2D)
class JohnnySvensson:
    def __init__(self, a=1.4, b=1.56, c=1.4, d=-6.56):
        self.a, self.b, self.c, self.d = a, b, c, d # Parámetros del sistema

    def step(self, x, y):
        # Calculamos el siguiente punto de la trayectoria
        x_next = self.d * torch.sin(self.a * x) - torch.sin(self.b * y)
        y_next = self.c * torch.cos(self.a * x) + torch.cos(self.b * y)
        return x_next, y_next

# Función para generar datos de entrenamiento a partir del atractor
def generate_data(n_trajectories=200, traj_len=300, device='cpu'):
    model = JohnnySvensson()            # Instancia del atractor
    xs, ys = [], []
    for _ in range(n_trajectories):
        x = (torch.rand(2, device=device)*4 - 2)    # Punto inicial aleatorio en [-2, 2]
        traj = [x.unsqueeze(0)]                     # Guardamos primer punto
        for _ in range(traj_len-1):
            x_next = torch.stack(model.step(x[0], x[1]))    # Paso siguiente
            traj.append(x_next.unsqueeze(0))                # Guardamos punto
            x = x_next                                      # Avanzamos
        traj = torch.cat(traj, dim=0)   # Convertimos lista a tensor
        xs.append(traj[:-1])            # Entrada: todos menos el último
        ys.append(traj[1:])             # Salida: todos menos el primero (siguiente paso)
    X = torch.cat(xs)
    Y = torch.cat(ys)

    # Normalizamos entradas y salidas (media 0, desviación 1)
    mean, std = X.mean(dim=0), X.std(dim=0)
    Xn = (X - mean) / std
    Yn = (Y - mean) / std
    return Xn, Yn, mean, std        # También devolvemos media y desviación para des-normalizar luego

# Definimos una capa personalizada de activación Radial (tipo RBF)
class RadialBasis(nn.Module):
    def forward(self, x):
        return torch.exp(-x**2)     # Función gaussiana sin centro fijo

# Red neuronal con capas densas y activación radial
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),       # Capa 1: 2 entradas → 20 neuronas
            nn.Sigmoid(),           # Activación no lineal sigmoidal
            nn.Linear(20, 20),      # Capa 2: 20 → 20 neuronas
            RadialBasis(),          # Activación radial (gaussiana): simula RBF
            nn.Linear(20, 20),      # Capa 3: 20 → 20 (lineal)
            nn.Linear(20, 2),       # Capa final: 20 → 2 salidas (x,y)
        )

    def forward(self, x):
        return self.net(x)

# Entrenamiento del modelo
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Usamos GPU si está disponible
X, Y, m, s = generate_data(device=device)   # Obtenemos datos ya normalizados
ds = torch.utils.data.TensorDataset(X, Y)   # Empaquetamos en Dataset
loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)  # Batch size 256

model = Net().to(device)    # Inicializamos red neuronal
opt = optim.Adam(model.parameters(), lr=1e-3)  # Optimizador Adam
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-5) # Scheduler para reducir LR
loss_fn = nn.MSELoss()  # Función de pérdida: error cuadrático medio

# Ciclo de entrenamiento
for epoch in range(1, 201): # 200 épocas
    model.train()
    running = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)                                # Predicción
        loss = loss_fn(pred, yb)                        # Cálculo del error
        opt.zero_grad(); loss.backward(); opt.step()    # Retropropagación
        running += loss.item() * xb.size(0)
    running /= len(ds)
    sched.step(running) # Actualizamos el scheduler
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d}  Loss: {running:.6f}  LR: {opt.param_groups[0]['lr']:.5f}")

# Predicción autoregresiva usando la red neuronal entrenada
model.eval()    # Modo evaluación
n_trajectories = 50         # Cantidad de trayectorias a simular
steps_per_traj  = 5000      # Longitud de cada trayectoria
burn_in         = 500       # Se descartan los primeros 500 puntos (fase transitoria)

all_pts = []
with torch.no_grad():       # No se calcula gradiente durante predicción
    for _ in range(n_trajectories):
        x = (torch.rand(2, device=device)*4 - 2)    # Punto inicial aleatorio
        traj = []
        for i in range(steps_per_traj):
            xn = (x - m.to(device)) / s.to(device)      # Normalizamos
            y_hat = model(xn.unsqueeze(0)).squeeze(0)   # Predecimos siguiente paso
            x = y_hat * s.to(device) + m.to(device)     # Desnormalizamos
            traj.append(x.cpu().numpy())                # Guardamos
        traj = np.array(traj)
        all_pts.append(traj[burn_in:])      # Guardamos solo parte estacionaria

pts = np.vstack(all_pts)    # Unimos todas las trayectorias

# Gráfico en 2D
plt.figure(figsize=(6,6))
plt.scatter(pts[:,0], pts[:,1], s=0.1, alpha=0.4)   # Puntos muy pequeños y semitransparentes
plt.title("Johnny Svensson Attractor (predicción NN, muy denso)")
plt.axis('equal')           # Ejes proporcionales
plt.show()