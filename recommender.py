from surprise import Dataset, Reader, KNNBasic, KNNWithMeans
from surprise.model_selection import KFold, cross_validate

# Ruta al dataset MovieLens 100k
fich = 'ml-100k/u.data'

# Definición de columnas del dataset
columnas = 'user item rating timestamp'

# Configurar el lector
lector = Reader(line_format=columnas, sep='\t')

# Cargar los datos
datos = Dataset.load_from_file(fich, reader=lector)

# Crear conjunto de entrenamiento completo
trainset = datos.build_full_trainset()

# Definir el modelo (KNN básico)
metodo = KNNBasic()
metodo.fit(trainset)

# Realizar predicciones individuales
usuario = str(196)
elemento = str(242)
prediccion = metodo.predict(usuario, elemento, r_ui=3, verbose=True)
print("Estimada =", prediccion.est, 'real =', prediccion.r_ui)

usuario = str(196)
elemento = str(302)
prediccion = metodo.predict(usuario, elemento, verbose=True)
print("Estimada =", prediccion.est)

# ---------------------------------------------------------
# Validación cruzada con otro método (KNNWithMeans)
# ---------------------------------------------------------

# Opciones de similitud
similOpts = {"name": "pearson", "user_based": True}

metodo2 = KNNWithMeans(k=20, sim_options=similOpts)

# Evaluar con validación cruzada
results = cross_validate(metodo2, datos, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print("\nResultados promedio:")
print("RMSE:", results['test_rmse'].mean())
print("MAE :", results['test_mae'].mean())
