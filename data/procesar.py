import pandas as pd

ARCHIVO_ENTRADA = "dataset.parquet"
ARCHIVO_SALIDA = "dataset_limpio.parquet"

# configuracion
UMBRAL_NULOS = 0.90

# Leer el archivo Parquet
print(f"Leyendo {ARCHIVO_ENTRADA}...")
df = pd.read_parquet(ARCHIVO_ENTRADA)


print("################  Analisis de datos ################")
def analizar_dataset(df):
    # Ver filas y columnas tiene el dataset
    filas, columnas = df.shape
    print(f"El dataset tiene {filas} filas y {columnas} columnas.\n")

    # Crear una tabla resumen combinando Tipos, Nulos y Porcentaje
    tabla_resumen = pd.DataFrame({
        'Tipo de Dato': df.dtypes,
        'Cantidad Nulos': df.isnull().sum(),
        '% de Nulos': (df.isnull().mean() * 100).round(2) # Agregamos el % para ver el umbral
    })
    print(tabla_resumen.to_string())

analizar_dataset(df=df)

print("################  Limpieza de datos  ################")
df = df.dropna(subset=['src_ip']) # eliminar 2 filas nulas

# eliminar columnas nulas
porcentaje_nulos = df.isnull().mean()
COLUMNAS_PROTEGIDAS = ["request_body"]
columnas_a_borrar = [
    c for c in porcentaje_nulos[porcentaje_nulos > UMBRAL_NULOS].index
    if c not in COLUMNAS_PROTEGIDAS
]
print(f"\nEliminando columnas con muchos nulos:")
for col in columnas_a_borrar:
    print(f"- {col}: {porcentaje_nulos[col] * 100:.2f}% nulls/vacia")
df = df.drop(columns=columnas_a_borrar)


print("################  Transformar datos  ################")
# convertir datos a entero
columnas_enteras = ['src_port', 'dst_port', 'response_http_status_code', 'response_content_length']
for col in columnas_enteras:
    df[col] = df[col].fillna(0).astype('int32')

# Rellenar los nullos con ""
columnas_texto = df.select_dtypes(include=['str']).columns
df[columnas_texto] = df[columnas_texto].fillna("")

columnas_etiquetas = [
    '000 - Normal', '272 - Protocol Manipulation', '242 - Code Injection',
    '88 - OS Command Injection', '126 - Path Traversal', '66 - SQL Injection',
    '16 - Dictionary-based Password Attack', '310 - Scanning for Vulnerable Software',
    '153 - Input Data Manipulation', '248 - Command Injection',
    '274 - HTTP Verb Tampering', '194 - Fake the Source of Data',
    '34 - HTTP Response Splitting', '33 - HTTP Request Smuggling'
]
df['is_attack'] = (df['000 - Normal'] == 0).astype('int32') # Variable binaria (1 = Ataque, 0 = Normal)

# tipo de ataque q puede estar 
columnas_solo_ataques = [col for col in columnas_etiquetas if col != '000 - Normal']
nombres_limpios = {col: col.split(' - ', 1)[1] for col in columnas_solo_ataques}
df['attack_type'] = df[columnas_solo_ataques].apply(
    lambda fila: ", ".join([nombres_limpios[col] for col in columnas_solo_ataques if fila[col] == 1]),
    axis=1
)
df.loc[df['attack_type'] == "", 'attack_type'] = "Normal" # si quedaron "" pasan a "Normal"
df = df.drop(columns=columnas_etiquetas)

analizar_dataset(df=df)

# Realizar prueba de integridad
print("################ Realizar prueba de integridad ################")
is_atk_arr = df['is_attack'].to_numpy()
atk_type_arr = df['attack_type'].to_numpy()

cond_error1 = (is_atk_arr == 0) & (atk_type_arr != "Normal")
cond_error2 = (is_atk_arr == 1) & (atk_type_arr == "Normal")

# Test 1 (Falsos Ataques)
if not cond_error1.any():
    print("(test 1) Sin errores logicos: 0 normales mal etiquetados.")
else:
    errores1 = df[cond_error1]
    print(f"(test 1) Errores logicos encontrados {len(errores1)} filas")
    print(errores1[['is_attack', 'attack_type']].head(10))

# Test 2 (Ataques Ocultos)
if not cond_error2.any():
    print("(test 2) Sin errores logicos: 0 ataques mal etiquetados.")
else:
    errores2 = df[cond_error2]
    print(f"(test 2) Errores logicos encontrados {len(errores2)} filas")
    print(errores2[['is_attack', 'attack_type']].head(10))

# Guardar el dataset procesado
df.to_parquet(ARCHIVO_SALIDA, index=False)
print(f"Dataset final guardado en {ARCHIVO_SALIDA}")