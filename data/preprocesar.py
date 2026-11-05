import pandas as pd

# variables globales
ARCHIVO_ENTRADA = "data_capec_multilabel.csv"
ARCHIVO_SALIDA = "dataset.parquet"

# Leer el archivo CSV
print(f"Leyendo {ARCHIVO_ENTRADA}...")
df = pd.read_csv(ARCHIVO_ENTRADA)

# Ver filas y columnas tiene el dataset
filas, columnas = df.shape
print(f"El dataset tiene {filas} filas y {columnas} columnas.")

# Guardar los datos en formato Parquet 
df.to_parquet(ARCHIVO_SALIDA, index=False)
print(f"Datos guardados exitosamente en: {ARCHIVO_SALIDA}")