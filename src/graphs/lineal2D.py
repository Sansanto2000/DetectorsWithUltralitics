import matplotlib.pyplot as plt

# Coordenadas de los 3 puntos
x = [640, 1024, 2048]
y = [0.835, 0.833, 0.812]

# Graficar puntos unidos por líneas
plt.plot(x, y, marker='o')

# Etiquetas opcionales
for i in range(len(x)):
    plt.text(x[i], y[i], f'({x[i]}, {y[i]})')

# Configuración básica
plt.xlabel("N")
plt.ylabel("Map50-95")
plt.title("Redimension en relacion al rendimiento")
plt.grid(True)

# Guardar gráfico
plt.savefig(r"/home/sponte/Repositorios/DetectorsWithUltralitics/src/graphs/show/lineal2D.png", dpi=300, bbox_inches='tight')

# Mostrar gráfico
plt.show()