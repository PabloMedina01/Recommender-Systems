from surprise import Dataset, Reader, KNNBasic, KNNWithMeans, SVD
from surprise.model_selection import cross_validate, GridSearchCV

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

# ---------------------------------------------------------
# Modelo KNN básico (predicciones individuales)
# ---------------------------------------------------------
metodo = KNNBasic()
metodo.fit(trainset)

usuario = str(196)
elemento = str(242)
prediccion = metodo.predict(usuario, elemento, r_ui=3, verbose=True)
print("Estimada =", prediccion.est, 'real =', prediccion.r_ui)

usuario = str(196)
elemento = str(302)
prediccion = metodo.predict(usuario, elemento, verbose=True)
print("Estimada =", prediccion.est)

# ---------------------------------------------------------
# Validación cruzada con KNNWithMeans
# ---------------------------------------------------------
similOpts = {"name": "pearson", "user_based": True}
metodo2 = KNNWithMeans(k=20, sim_options=similOpts)

results = cross_validate(metodo2, datos, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print("\nResultados promedio (KNNWithMeans):")
print("RMSE:", results['test_rmse'].mean())
print("MAE :", results['test_mae'].mean())

# ---------------------------------------------------------
# Validación cruzada con SVD
# ---------------------------------------------------------
metodo3 = SVD()

results_svd = cross_validate(metodo3, datos, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print("\nResultados promedio (SVD):")
print("RMSE:", results_svd['test_rmse'].mean())
print("MAE :", results_svd['test_mae'].mean())

# ---------------------------------------------------------
# Búsqueda en cuadrícula (Grid Search) con SVD
# ---------------------------------------------------------
param_grid = {
    'n_factors': [50, 100, 200],
    'n_epochs': [20, 40, 60],
    'lr_all': [0.002, 0.005, 0.01],
    'reg_all': [0.2, 0.4, 0.6]
}

grid_search = GridSearchCV(SVD, param_grid, measures=['RMSE', 'MAE'], cv=5)
grid_search.fit(datos)

print("\nResultados de la búsqueda en cuadrícula (SVD):")
print("Mejor RMSE:", grid_search.best_score['RMSE'])
print("Mejor MAE :", grid_search.best_score['MAE'])
print("Mejores hiperparámetros:", grid_search.best_params['RMSE'])
