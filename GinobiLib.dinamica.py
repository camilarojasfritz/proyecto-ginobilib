import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar el video
cap = cv2.VideoCapture('Tiro_dinamica1.mov')

# Leer el primer frame para seleccionar la ROI
ret, frame = cap.read()

# Seleccionar la ROI manualmente
bbox = cv2.selectROI("Selecciona el área de la pelota", frame)
cv2.destroyAllWindows()

# Inicializar el tracker
tracker = cv2.TrackerCSRT.create()
tracker.init(frame, bbox)

# Variables para almacenar posiciones de la pelota
positions = []
times = []

# Definir los tiempos de contacto en segundos
t1 = 0.03  # tiempo inicial de contacto en segundos
t2 = 0.22  # tiempo final de contacto en segundos

# Procesar el video frame por frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Actualizar el tracker
    ret, bbox = tracker.update(frame)
    if ret:
        x, y, w, h = [int(v) for v in bbox]
        center_x = x + w // 2
        center_y = y + h // 2
        center_y = frame.shape[0] - (y + h // 2)
        print(center_x, center_y)

        # Guardar la posición y el tiempo
        positions.append((center_x, center_y))
        times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)  # tiempo en segundos

        # Dibujar la caja de seguimiento en el frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
        cv2.imshow("Tracking", frame)

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

# Convertir listas a arrays numpy
positions = np.array(positions)
times = np.array(times)

# Verificar los tiempos capturados
# print("Tiempos capturados:", times)

# Asegurarse de que t1 y t2 estén dentro del rango de tiempos disponibles
if t1 >= times[0] and t2 <= times[-1]:
    # Encontrar las posiciones en los tiempos t1 y t2
    idx_t1 = np.argmin(np.abs(times - t1))
    idx_t2 = np.argmin(np.abs(times - t2))

    posicion_t1 = positions[idx_t1]
    posicion_t2 = positions[idx_t2]

    print(f"Posicion t1:{posicion_t1}. Posicion t2: {posicion_t2}")
    
    x1, y1 = positions[idx_t1]
    x2, y2 = positions[idx_t2]

    # Calcular la velocidad en píxeles por segundo
    vx = (x2 - x1) / (t2 - t1)
    vy = (y2 - y1) / (t2 - t1)

    # Masa de la pelota
    masa_pelota = 0.6  # kg

    # Calcular el impulso (cambio de momento)
    impulso_x = masa_pelota * vx
    impulso_y = masa_pelota * vy

    # Calcular la fuerza aplicada durante el contacto
    tiempo_contacto = t2 - t1
    fuerza_x = impulso_x / tiempo_contacto
    fuerza_y = impulso_y / tiempo_contacto

    # Constante de gravedad
    g = 9.81  # m/s^2

    # Calcular la fuerza de gravedad (peso)
    peso = masa_pelota * g

    # Parámetros para el rozamiento del aire
    coef_arrastre = 0.5  # coeficiente de arrastre
    area_referencia = 0.045  # área de referencia en m^2 (asumiendo un diámetro de 24 cm)
    densidad_aire = 1.225  # densidad del aire en kg/m^3

    # Calcular la fuerza de arrastre
    '''
        Cuando un objeto se mueve a través de un fluido (aire, agua,…) el mismo ejerce una fuerza de
        resistencia (conocida como fuerza de arrastre) que tiende a reducir su velocidad.
        La fuerza de arrastre depende de propiedades del fluido y del tamaño, forma y velocidad del objeto
        relativa al fluido.
    '''
    fuerza_arrastre_x = 0.5 * coef_arrastre * area_referencia * densidad_aire * (vx**2)
    fuerza_arrastre_y = 0.5 * coef_arrastre * area_referencia * densidad_aire * (vy**2)

    # Mostrar resultados
    print(f"Impulso en x: {round(impulso_x, 3)} Ns")
    print(f"Impulso en y: {round(impulso_y, 3)} Ns")
    print(f"Fuerza en x: {round(fuerza_x, 3)} N")
    print(f"Fuerza en y: {round(fuerza_y, 3)} N")
    print(f"Peso de la pelota: {peso} N")
    print(f"Fuerza de arrastre en x: {round(fuerza_arrastre_x, 3)} N")
    print(f"Fuerza de arrastre en y: {round(fuerza_arrastre_y, 3)} N")

    # Visualización opcional
    plt.plot(times, positions[:, 0], label='Posición en x')
    plt.plot(times, positions[:, 1], label='Posición en y')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Posición (píxeles)')
    plt.legend()
    plt.show()

    # ------------ RECORTAR GRAFICA ENTRE T1 Y T2, REVISAR ------------ #

    # mask = (times >= t1) & (times <= t2)
    # filtered_times = times[mask]
    # filtered_positions_x = positions[:, 0][mask]
    # filtered_positions_y = positions[:, 1][mask]

    # # Graficar las posiciones filtradas
    # plt.plot(filtered_times, filtered_positions_x, label='Posición en x')
    # plt.plot(filtered_times, filtered_positions_y, label='Posición en y')
    # plt.xlabel('Tiempo (s)')
    # plt.ylabel('Posición (píxeles)')
    # plt.legend()
    # plt.show()
else:
    print(f"Los tiempos t1 ({t1}s) y t2 ({t2}s) están fuera del rango del video.")
