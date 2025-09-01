import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import os

def leeJugadores(nombreFichero: str) -> np.ndarray:
    """
    Lee un fichero de datos con dos columnas separadas por tabulador:
    - Columna 1: horas de juego
    - Columna 2: horas de chat
    Retorna una matriz numpy con shape (n, 2).
    """
    with open(nombreFichero, encoding="utf-8") as fichero:
        lineas = [(l.strip()).split('\t') for l in fichero if l.strip()]
        horasJuego = [float(x[0]) for x in lineas]
        horasChat = [float(x[1]) for x in lineas]
        matriz = np.column_stack((horasJuego, horasChat))
    return matriz


# ========================
# Lectura de datos
# ========================
datos = leeJugadores('data/players.data')

# ========================
# Clustering jerárquico
# ========================
enlace = linkage(datos, method='complete', metric='euclidean')

# Ejemplo de salida de los primeros enlaces
print("Primeros enlaces del clustering jerárquico:")
for i, row in enumerate(enlace[:6]):  # mostramos los primeros 6
    id1, id2, dist, cluster_size = row
    print(f"id1={int(id1)}")
    print(f"id2={int(id2)}")
    print(f"distancia={dist:.4f}")
    print(f"tamaño del cluster={int(cluster_size)}")
    print("-----------------------------------")

# Crear carpeta de salida si no existe
os.makedirs("output", exist_ok=True)

# Dendrograma
plt.figure(figsize=(12, 8))
plt.title("Dendrograma de agrupamiento jerárquico", fontsize=18)
plt.xlabel("Índice de la muestra", fontsize=14)
plt.ylabel("Distancia", fontsize=14)

dendrogram(
    enlace,
    orientation='top',
    labels=np.arange(1, datos.shape[0] + 1),
    distance_sort='descending',
    show_leaf_counts=True
)

plt.tight_layout()
plt.savefig('output/dendrograma.png')
plt.close()


# ========================
# K-means clustering
# ========================
k = 4
kmedios = KMeans(n_clusters=k, n_init=10, random_state=42)
kmedios.fit(datos)

print("\nEtiquetas de los primeros 100 elementos:")
print(kmedios.labels_[:100])

print("\nCentros de los clusters:")
print(kmedios.cluster_centers_)

print("\nPredicción de nuevos puntos [[2,3], [4,1]]:")
print(kmedios.predict([[2, 3], [4, 1]]))

# Visualización K-means
plt.figure(figsize=(10, 7))
plt.scatter(datos[:, 0], datos[:, 1], c=kmedios.labels_, cmap='viridis', s=50, alpha=0.7)
plt.scatter(kmedios.cluster_centers_[:, 0], kmedios.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, label='Centroides')
plt.title("Clustering K-means", fontsize=18)
plt.xlabel("Horas de Juego", fontsize=14)
plt.ylabel("Horas de Chat", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('output/kmeans.png')
plt.close()


## Spectral Clustering

from sklearn.cluster import SpectralClustering

SC = SpectralClustering(n_clusters=4, affinity='nearest_neighbors', n_neighbors=10, random_state=42)
SC.fit(datos)

print("\nEtiquetas de los primeros 100 elementos (Spectral Clustering):")
print(SC.labels_[:100])

plt.figure(figsize=(10, 7))
plt.scatter(datos[:, 0], datos[:, 1], c=SC.labels_, cmap='viridis', s=50, alpha=0.7)
plt.title("Clustering Espectral", fontsize=18)
plt.xlabel("Horas de Juego", fontsize=14)
plt.ylabel("Horas de Chat", fontsize=14)
plt.tight_layout()
plt.savefig('output/spectral_clustering.png')
plt.close()