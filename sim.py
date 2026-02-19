import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve

# ============================================
# PARAMETROS MODIFICABLES - ¡CAMBIA AQUÍ!
# ============================================

# TIEMPO DE SIMULACION - Cambia esto para simular más o menos tiempo
TIEMPO_FINAL = 10.0     # [s] - Tiempo final de simulación
dt = 0.01             # [s] - Paso temporal

# CONDICION INICIAL - ¡NUEVO PARAMETRO MODIFICABLE!
CONSTANTE_INICIAL = 100.0  # [n/(cm²·s)] - Flujo inicial constante en todo el reactor

# PARAMETROS FISICOS
v = 1.0               # Velocidad neutrones [cm/s]
D = 0.5              # Coeficiente de difusion [cm]

# GEOMETRIA DEL REACTOR
R_reactor = 5.0       # Radio del reactor [cm]
R_barra_comb = 0.1    # Radio barras combustible [cm]
R_barra_ctrl = 0.5    # Radio barra control [cm]

# PROPIEDADES DE MATERIALES
sigma_comb = -30     # Seccion eficaz absorcion combustible [1/cm]
S_ctrl = -400.0        # Fuerza sumidero barra control

# PARAMETROS DE VISUALIZACION
H_reactor = 1.0       # Altura para visualizacion 3D

# DISCRETIZACION ESPACIAL
Nr = 80               # Puntos en direccion radial
Ntheta = 100          # Puntos en direccion angular

# ============================================
# GEOMETRIA DEL REACTOR
# ============================================
# Posiciones de las barras (coordenadas polares)
# Las barras se colocan a 30% del radio del reactor
posiciones = {
    'combustible': [
        {'r': R_reactor * 0.3, 'theta': np.pi/4},
        {'r': R_reactor * 0.3, 'theta': 3*np.pi/4},
        {'r': R_reactor * 0.3, 'theta': 5*np.pi/4},
        {'r': R_reactor * 0.3, 'theta': 7*np.pi/4},
    ],
    'control': [{'r': 0.0, 'theta': 0.0}]
}

# ============================================
# DISCRETIZACION ESPACIAL
# ============================================
print(f"Configuracion de simulacion:")
print(f"  Tiempo final: {TIEMPO_FINAL} s")
print(f"  Paso temporal: {dt} s")
print(f"  Flujo inicial constante: {CONSTANTE_INICIAL} n/(cm²·s)")
print(f"  Puntos temporales: {int(TIEMPO_FINAL/dt) + 1}")
print(f"  Radio reactor: {R_reactor} cm")

N = Nr * Ntheta  # Total puntos

dr = R_reactor / (Nr - 1)
dtheta = 2 * np.pi / Ntheta

# Crear malla 2D en polares
r = np.linspace(0, R_reactor, Nr)
theta = np.linspace(0, 2*np.pi - dtheta, Ntheta)
R, Theta = np.meshgrid(r, theta, indexing='ij')

# ============================================
# FUNCIONES DE MATERIALES
# ============================================
def crear_sigma_a(R, Theta):
    sigma = np.zeros_like(R)
    for barra in posiciones['combustible']:
        r0 = barra['r']
        theta0 = barra['theta']
        dist2 = R**2 + r0**2 - 2*R*r0*np.cos(Theta - theta0)
        sigma[dist2 <= R_barra_comb**2] = sigma_comb
    return sigma

def crear_sumideros(R, Theta):
    S = np.zeros_like(R)
    for barra in posiciones['control']:
        r0 = barra['r']
        if r0 == 0:
            mask = R <= R_barra_ctrl
        else:
            theta0 = barra['theta']
            dist2 = R**2 + r0**2 - 2*R*r0*np.cos(Theta - theta0)
            mask = dist2 <= R_barra_ctrl**2
        S[mask] = S_ctrl
    return S

sigma_a = crear_sigma_a(R, Theta)
S = crear_sumideros(R, Theta)

# ============================================
# CONSTRUCCION DE MATRICES
# ============================================
print("\nConstruyendo matrices del sistema...")

# Matriz Laplaciano en coordenadas polares
main_diag_r = -2 * np.ones(Nr)
off_diag_r = np.ones(Nr-1)
main_diag_r[-1] = -1  # Condicion Neumann en r=R

L_r = diags([off_diag_r, main_diag_r, off_diag_r], [-1, 0, 1], shape=(Nr, Nr), format='csr')

# Evitar division por cero en r=0
r_inv = np.zeros_like(r)
r_inv[1:] = 1.0 / r[1:]  # Solo para r > 0

# Derivada primera en r
main_diag_dr = np.zeros(Nr)
L_dr = diags([-np.ones(Nr-1), main_diag_dr, np.ones(Nr-1)], [-1, 0, 1], shape=(Nr, Nr), format='csr')
L_dr[-1, -2] = 0  # Neumann
L_dr = diags(r_inv, 0) @ L_dr / (2*dr)

# Parte angular (condiciones periodicas)
main_diag_theta = -2 * np.ones(Ntheta)
off_diag_theta = np.ones(Ntheta-1)
L_theta = diags([off_diag_theta, main_diag_theta, off_diag_theta], [-1, 0, 1], shape=(Ntheta, Ntheta), format='csr')
L_theta[0, -1] = 1
L_theta[-1, 0] = 1
L_theta = L_theta / (dtheta**2)

# Operador Laplaciano completo
I_r = eye(Nr, format='csr')
I_theta = eye(Ntheta, format='csr')

# Evitar division por cero en r=0
r2_inv = np.zeros_like(r)
r2_inv[1:] = 1.0 / (r[1:]**2)  # Solo para r > 0

L_radial = L_r/(dr**2) + L_dr
L_angular = kron(I_r, L_theta)
L_angular = diags(r2_inv.repeat(Ntheta), 0) @ L_angular
Lap = kron(L_radial, I_theta) + L_angular

# Matriz del sistema
A = (1/(v*dt)) * eye(N, format='csr') - D * Lap + diags(sigma_a.flatten(), 0, format='csr')

# ============================================
# SIMULACION TEMPORAL
# ============================================
print("\nIniciando simulacion temporal...")

# CONDICION INICIAL CONSTANTE MODIFICABLE
phi0 = np.ones_like(R) * CONSTANTE_INICIAL  # [n/(cm²·s)] - Valor configurable
phi = phi0.flatten()
b_vec = S.flatten()

# Elemento de area en coordenadas polares
dA = r[:, np.newaxis] * dr * dtheta  # [cm²]

# Calcular tiempos para los 3 plots 3D
tiempo_inicio = 0.0
tiempo_medio = TIEMPO_FINAL / 2.0
tiempo_final = TIEMPO_FINAL

tiempos_3d = [tiempo_inicio, tiempo_medio, tiempo_final]
resultados_3d = []  # Para guardar los 3 tiempos para plots 3D

# Para la grafica de neutrones por segundo
tiempos_neutrones = []
neutrones_por_segundo = []

# Simular evolucion completa
total_pasos = int(TIEMPO_FINAL / dt)  # Pasos totales
puntos_totales = total_pasos + 1  # Puntos totales (incluyendo t=0)

print(f"Simulando {total_pasos} pasos hasta t={TIEMPO_FINAL} s...")
print(f"Flujo inicial constante: {CONSTANTE_INICIAL} n/(cm²·s)")

# Bucle principal de simulacion
for paso in range(total_pasos + 1):
    tiempo_actual = paso * dt
    
    # Calcular flujo actual y neutrones por segundo
    phi_2d = phi.reshape((Nr, Ntheta))
    neutrones_ps = np.sum(phi_2d * dA)  # ∫ φ dA [n/s]
    
    # Guardar TODOS los puntos para la grafica
    tiempos_neutrones.append(tiempo_actual)
    neutrones_por_segundo.append(neutrones_ps)
    
    # Guardar para plots 3D en los tiempos especificados
    for t_target in tiempos_3d:
        # Verificar si estamos en el tiempo exacto o muy cercano
        if abs(tiempo_actual - t_target) < dt/2:
            # Solo guardar si no lo tenemos ya
            tiempos_guardados = [t for t, _ in resultados_3d]
            if t_target not in tiempos_guardados:
                resultados_3d.append((tiempo_actual, phi_2d.copy()))
                print(f"  Guardado para plot 3D: t = {tiempo_actual:.3f} s")
    
    # Mostrar progreso
    if paso % max(1, total_pasos // 10) == 0 and paso > 0:
        porcentaje = (paso / total_pasos) * 100
        print(f"  Progreso: {porcentaje:.0f}% - t = {tiempo_actual:.3f} s, N = {neutrones_ps:.2e} n/s")
    
    # Avanzar en el tiempo (excepto en el ultimo paso)
    if paso < total_pasos:
        b = phi/(v*dt) + b_vec
        phi = spsolve(A, b)

print("\n" + "="*60)
print(f"SIMULACION COMPLETADA")
print(f"Puntos generados para grafica de neutrones: {len(tiempos_neutrones)}")
print("="*60)

# Asegurar que tenemos exactamente 3 tiempos para los plots 3D
if len(resultados_3d) < 3:
    print("Generando tiempos faltantes para plots 3D...")
    # Simular específicamente para los tiempos que faltan
    for t_target in tiempos_3d:
        if t_target not in [t for t, _ in resultados_3d]:
            # Recalcular el flujo para ese tiempo
            phi_temp = phi0.flatten().copy()
            pasos_necesarios = int(t_target / dt)
            
            for p in range(pasos_necesarios + 1):
                if p > 0:
                    b_temp = phi_temp/(v*dt) + b_vec
                    phi_temp = spsolve(A, b_temp)
            
            phi_2d_temp = phi_temp.reshape((Nr, Ntheta))
            resultados_3d.append((t_target, phi_2d_temp.copy()))
            print(f"  Generado plot 3D faltante: t = {t_target:.3f} s")

# Ordenar resultados_3d por tiempo
resultados_3d.sort(key=lambda x: x[0])

# ============================================
# FUNCION PARA DIBUJAR CILINDROS 3D
# ============================================
def dibujar_cilindro(ax, x_center, y_center, radio, altura, color, alpha=0.7):
    """Dibuja un cilindro 3D en la posicion especificada"""
    # Parametros para el cilindro
    z = np.linspace(0, altura, 10)
    theta_cil = np.linspace(0, 2*np.pi, 30)
    
    # Crear malla para la superficie del cilindro
    Z, Theta_cil = np.meshgrid(z, theta_cil)
    
    # Coordenadas del cilindro
    X_cil = x_center + radio * np.cos(Theta_cil)
    Y_cil = y_center + radio * np.sin(Theta_cil)
    
    # Dibujar superficie del cilindro
    ax.plot_surface(X_cil, Y_cil, Z, color=color, alpha=alpha, 
                   edgecolor='black', linewidth=0.3, shade=True)

# ============================================
# CONVERTIR A COORDENADAS CARTESIANAS
# ============================================
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

# ============================================
# 1. PRIMER PLOT 3D: TIEMPO INICIAL (t=0)
# ============================================
if len(resultados_3d) > 0:
    tiempo1, phi_snapshot1 = resultados_3d[0]
    
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    # Dibujar los cilindros de las barras
    for barra in posiciones['combustible']:
        x_bar = barra['r'] * np.cos(barra['theta'])
        y_bar = barra['r'] * np.sin(barra['theta'])
        dibujar_cilindro(ax1, x_bar, y_bar, R_barra_comb, H_reactor, 
                        color='green', alpha=0.6)
    
    # Barra de control
    dibujar_cilindro(ax1, 0, 0, R_barra_ctrl, H_reactor, 
                    color='red', alpha=0.6)
    
    # Escalar el flujo para visualizacion
    phi_scaled1 = phi_snapshot1.copy()
    phi_max_total = np.max([phi.max() for _, phi in resultados_3d])
    if phi_max_total > 0:
        phi_scaled1 = phi_scaled1 / phi_max_total * H_reactor
    
    # Crear superficie del flujo
    surf1 = ax1.plot_surface(X, Y, phi_scaled1, cmap='plasma', 
                           alpha=0.7, rcount=60, ccount=60)
    
    # Configurar vista
    ax1.view_init(elev=30, azim=45)
    ax1.set_xlabel('X [cm]', fontsize=11, labelpad=10)
    ax1.set_ylabel('Y [cm]', fontsize=11, labelpad=10)
    ax1.set_zlabel('Flujo (escalado)', fontsize=11, labelpad=10)
    ax1.set_title(f'Distribucion Inicial de Neutrones\nTiempo: t = {tiempo1:.3f} s\nFlujo inicial: {CONSTANTE_INICIAL} n/(cm²·s)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Limites
    ax1.set_xlim([-R_reactor*1.1, R_reactor*1.1])
    ax1.set_ylim([-R_reactor*1.1, R_reactor*1.1])
    ax1.set_zlim([0, H_reactor*1.2])
    
    # Barra de color
    fig1.colorbar(surf1, ax=ax1, shrink=0.6, aspect=20, pad=0.1, 
                 label='Flujo φ [n/cm²·s] (relativo)')
    
    plt.tight_layout()
    plt.show()

# ============================================
# 2. SEGUNDO PLOT 3D: TIEMPO INTERMEDIO
# ============================================
if len(resultados_3d) > 1:
    tiempo2, phi_snapshot2 = resultados_3d[1]
    
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    # Dibujar los cilindros de las barras
    for barra in posiciones['combustible']:
        x_bar = barra['r'] * np.cos(barra['theta'])
        y_bar = barra['r'] * np.sin(barra['theta'])
        dibujar_cilindro(ax2, x_bar, y_bar, R_barra_comb, H_reactor, 
                        color='green', alpha=0.6)
    
    # Barra de control
    dibujar_cilindro(ax2, 0, 0, R_barra_ctrl, H_reactor, 
                    color='red', alpha=0.6)
    
    # Escalar el flujo
    phi_scaled2 = phi_snapshot2.copy()
    if phi_max_total > 0:
        phi_scaled2 = phi_scaled2 / phi_max_total * H_reactor
    
    # Crear superficie del flujo
    surf2 = ax2.plot_surface(X, Y, phi_scaled2, cmap='plasma', 
                           alpha=0.7, rcount=60, ccount=60)
    
    # Configurar vista
    ax2.view_init(elev=30, azim=45)
    ax2.set_xlabel('X [cm]', fontsize=11, labelpad=10)
    ax2.set_ylabel('Y [cm]', fontsize=11, labelpad=10)
    ax2.set_zlabel('Flujo (escalado)', fontsize=11, labelpad=10)
    ax2.set_title(f'Distribucion Intermedia de Neutrones\nTiempo: t = {tiempo2:.3f} s ({tiempo2/TIEMPO_FINAL*100:.0f}% de {TIEMPO_FINAL} s)\nFlujo inicial: {CONSTANTE_INICIAL} n/(cm²·s)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Limites
    ax2.set_xlim([-R_reactor*1.1, R_reactor*1.1])
    ax2.set_ylim([-R_reactor*1.1, R_reactor*1.1])
    ax2.set_zlim([0, H_reactor*1.2])
    
    # Barra de color
    fig2.colorbar(surf2, ax=ax2, shrink=0.6, aspect=20, pad=0.1, 
                 label='Flujo φ [n/cm²·s] (relativo)')
    
    plt.tight_layout()
    plt.show()

# ============================================
# 3. TERCER PLOT 3D: TIEMPO FINAL
# ============================================
if len(resultados_3d) > 2:
    tiempo3, phi_snapshot3 = resultados_3d[2]
    
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection='3d')
    
    # Dibujar los cilindros de las barras
    for barra in posiciones['combustible']:
        x_bar = barra['r'] * np.cos(barra['theta'])
        y_bar = barra['r'] * np.sin(barra['theta'])
        dibujar_cilindro(ax3, x_bar, y_bar, R_barra_comb, H_reactor, 
                        color='green', alpha=0.6)
    
    # Barra de control
    dibujar_cilindro(ax3, 0, 0, R_barra_ctrl, H_reactor, 
                    color='red', alpha=0.6)
    
    # Escalar el flujo
    phi_scaled3 = phi_snapshot3.copy()
    if phi_max_total > 0:
        phi_scaled3 = phi_scaled3 / phi_max_total * H_reactor
    
    # Crear superficie del flujo
    surf3 = ax3.plot_surface(X, Y, phi_scaled3, cmap='plasma', 
                           alpha=0.7, rcount=60, ccount=60)
    
    # Configurar vista
    ax3.view_init(elev=30, azim=45)
    ax3.set_xlabel('X [cm]', fontsize=11, labelpad=10)
    ax3.set_ylabel('Y [cm]', fontsize=11, labelpad=10)
    ax3.set_zlabel('Flujo (escalado)', fontsize=11, labelpad=10)
    ax3.set_title(f'Distribucion Final de Neutrones\nTiempo: t = {tiempo3:.3f} s (final)\nFlujo inicial: {CONSTANTE_INICIAL} n/(cm²·s)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Limites
    ax3.set_xlim([-R_reactor*1.1, R_reactor*1.1])
    ax3.set_ylim([-R_reactor*1.1, R_reactor*1.1])
    ax3.set_zlim([0, H_reactor*1.2])
    
    # Barra de color
    fig3.colorbar(surf3, ax=ax3, shrink=0.6, aspect=20, pad=0.1, 
                 label='Flujo φ [n/cm²·s] (relativo)')
    
    plt.tight_layout()
    plt.show()

# ============================================
# 4. GRAFICA DE NEUTRONES POR SEGUNDO
# ============================================
fig4, ax4 = plt.subplots(figsize=(12, 7))

# Calcular cuantos puntos mostrar
puntos_totales = len(tiempos_neutrones)
mostrar_cada = max(1, puntos_totales // 100)  # Maximo 100 puntos visibles

# Graficar puntos seleccionados
indices = range(0, puntos_totales, mostrar_cada)
ax4.scatter([tiempos_neutrones[i] for i in indices], 
           [neutrones_por_segundo[i] for i in indices], 
           s=30, color='blue', alpha=0.7, 
           label='puntos de simulación')

# Conectar con linea
ax4.plot(tiempos_neutrones, neutrones_por_segundo, 
        'b-', linewidth=1, alpha=0.5)

# Marcar los 3 tiempos de las graficas 3D
colors_3d = ['red', 'green', 'purple']
nombres_3d = ['Inicial', 'Intermedio', 'Final']
for i, (tiempo, phi_snapshot) in enumerate(resultados_3d[:3]):
    idx = np.argmin(np.abs(np.array(tiempos_neutrones) - tiempo))
    neutrones_ps = neutrones_por_segundo[idx]
    
    # Destacar estos puntos
    ax4.scatter(tiempo, neutrones_ps, 
               s=150, color=colors_3d[i], 
               edgecolor='black', linewidth=2, 
               zorder=10, label=f'{nombres_3d[i]}: t = {tiempo:.3f} s')
    
    # Linea vertical
    ax4.axvline(x=tiempo, color=colors_3d[i], 
               linestyle='--', alpha=0.5, linewidth=1)
    
    # Anotacion
    offset_y = 15 if i % 2 == 0 else -15
    ax4.annotate(f'{neutrones_ps:.3e} n/s', 
                xy=(tiempo, neutrones_ps),
                xytext=(5, offset_y),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                color=colors_3d[i],
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='white', 
                         alpha=0.9,
                         edgecolor=colors_3d[i]))

# Configurar grafica
ax4.set_xlabel('Tiempo [s]', fontsize=12)
ax4.set_ylabel('Neutrones por Segundo [n/s]', fontsize=12)
ax4.set_title(f'Neutrones por Segundo en el Reactor\nT_final = {TIEMPO_FINAL} s, dt = {dt} s, φ₀ = {CONSTANTE_INICIAL} n/(cm²·s)', 
             fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10, loc='lower right')
ax4.set_xlim([-TIEMPO_FINAL*0.05, TIEMPO_FINAL*1.05])

# Informacion de configuracion
info_text = f'Configuración:\nT_final = {TIEMPO_FINAL} s\ndt = {dt} s\nφ₀ = {CONSTANTE_INICIAL} n/(cm²·s)'
ax4.text(0.02, 0.82, info_text,
        transform=ax4.transAxes, fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

# Calcular estadisticas
if len(neutrones_por_segundo) > 0:
    neutrones_inicial = neutrones_por_segundo[0]
    neutrones_final = neutrones_por_segundo[-1]
    diferencia = neutrones_final - neutrones_inicial
    porcentaje_total = (diferencia / neutrones_inicial) * 100 if neutrones_inicial != 0 else 0
    
    # Estadisticas en grafica
    stats_text = f'Estadísticas:\nInicial: {neutrones_inicial:.3e} n/s\nFinal: {neutrones_final:.3e} n/s\nCambio: {diferencia:.3e} n/s\n({porcentaje_total:.1f}%)'
    ax4.text(0.5, 0.85, stats_text,
            transform=ax4.transAxes, fontsize=10,
            verticalalignment='top',
            horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

# ============================================
# RESUMEN EN CONSOLA
# ============================================
print("\n" + "="*60)
print("RESUMEN DE RESULTADOS:")
print("="*60)
print(f"Tiempo final configurado: {TIEMPO_FINAL} s")
print(f"Paso temporal: {dt} s")
print(f"Condicion inicial: φ₀ = {CONSTANTE_INICIAL} n/(cm²·s)")
print(f"Puntos calculados: {len(tiempos_neutrones)}")

# Calcular area total del reactor
area_total = np.pi * R_reactor**2
neutrones_inicial_teorico = CONSTANTE_INICIAL * area_total

print(f"\nArea total del reactor: {area_total:.2f} cm²")
print(f"Neutrones iniciales teoricos: {neutrones_inicial_teorico:.2e} n/s")

if len(neutrones_por_segundo) > 0:
    neutrones_inicial = neutrones_por_segundo[0]
    neutrones_final = neutrones_por_segundo[-1]
    diferencia = neutrones_final - neutrones_inicial
    porcentaje_total = (diferencia / neutrones_inicial) * 100 if neutrones_inicial != 0 else 0
    
    print(f"\nNeutrones por segundo (simulados):")
    print(f"  Inicial (t=0.0s):    {neutrones_inicial:.4e} n/s")
    print(f"  Intermedio (t={TIEMPO_FINAL/2:.3f}s): {neutrones_por_segundo[int((TIEMPO_FINAL/2)/dt)]:.4e} n/s")
    print(f"  Final (t={TIEMPO_FINAL:.3f}s):    {neutrones_final:.4e} n/s")
    print(f"  Cambio:              {diferencia:.4e} n/s ({porcentaje_total:.1f}%)")
    
    # Comparacion con teorico
    diferencia_teorico = neutrones_inicial - neutrones_inicial_teorico
    porcentaje_error = (diferencia_teorico / neutrones_inicial_teorico) * 100 if neutrones_inicial_teorico != 0 else 0
    print(f"\nComparacion con valor teorico:")
    print(f"  Teorico: {neutrones_inicial_teorico:.4e} n/s")
    print(f"  Simulado: {neutrones_inicial:.4e} n/s")
    print(f"  Diferencia: {diferencia_teorico:.4e} n/s ({porcentaje_error:.2f}%)")

print("\n" + "="*60)
print("GRAFICAS GENERADAS:")
print("="*60)
print("1. Plot 3D - Tiempo inicial (t=0.0 s)")
print(f"2. Plot 3D - Tiempo intermedio (t={TIEMPO_FINAL/2:.3f} s)")
print(f"3. Plot 3D - Tiempo final (t={TIEMPO_FINAL:.3f} s)")
print(f"4. Grafica 2D - Neutrones por segundo ({len(tiempos_neutrones)} puntos)")
print("="*60)