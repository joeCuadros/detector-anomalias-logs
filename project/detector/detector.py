import json
import os
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
import requests
import warnings

warnings.filterwarnings("ignore")
OLLAMA_URL = os.getenv("OLLAMA_URL", "url_ollama_no_encontrado")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "modelo_no_encontrado")
MODO = "llm"  # "ml" | "llm" | "hibrida"

## Modelos de machine learning
USAR_RESPUESTA = True
RUTA_MODELS = Path(__file__).resolve().parent / "models"

#modelo_rf = joblib.load(RUTA_MODELS / "modelo_random_forest_con_respuesta.pkl")
#modelo_logistica = joblib.load(RUTA_MODELS / "modelo_logistica_con_respuesta.pkl")
modelo_xgb = xgb.XGBClassifier()
modelo_xgb.load_model(RUTA_MODELS / "modelo_xgboost_con_respuesta.json")
#modelo_lgb_base = lgb.Booster(model_file=str(RUTA_MODELS / "modelo_lightgbm_base_con_respuesta.txt"))
#modelo_lgb_balanced = lgb.Booster(model_file=str(RUTA_MODELS / "modelo_lightgbm_balanced_con_respuesta.txt"))
tfidf = joblib.load(RUTA_MODELS / "vectorizador_tfidf.pkl")

MODELOS = {
#    "random_forest": {"usar": True, "modelo": modelo_rf},
#    "logistica": {"usar": True, "modelo": modelo_logistica},
    "xgboost": {"usar": True, "modelo": modelo_xgb},
#    "lightgbm_base": {"usar": True, "modelo": modelo_lgb_base},
#    "lightgbm_balanced": {"usar": True, "modelo": modelo_lgb_balanced},
}

#######################################################################################
# Recibe el diccionario y lo procesa para el modelo a usar
def preprocesar(r: dict):
    metodo = str(r.get("method") or "")
    uri = str(r.get("uri") or "")
    body = str(r.get("body") or "")
    texto = metodo + " " + uri + " " + body

    num_list = [len(uri), len(body)]

    if USAR_RESPUESTA:
        status = r.get("status")
        response_size = r.get("response_size")
        if status is None or response_size is None:
            raise ValueError("Faltan campos de respuesta HTTP.")
        num_list += [int(status), int(response_size)]

    X_num = csr_matrix(np.array([num_list], dtype=np.float32))
    X_tfidf = tfidf.transform([texto])
    return hstack([X_num, X_tfidf]).toarray().astype(np.float32)

# usando el modelo
def procesar(X_input):
    resultados = {}
    for nombre, config in MODELOS.items():
        if not config.get("usar", False):
            continue

        m = config["modelo"]

        if isinstance(m, lgb.Booster):
            p = float(m.predict(X_input)[0])
            pred = int(p >= 0.5)
            prob = [1.0 - p, p]
        elif hasattr(m, "predict_proba"):
            pred = int(m.predict(X_input)[0])
            prob = m.predict_proba(X_input)[0].tolist()
        else:
            pred = int(m.predict(X_input)[0])
            prob = []

        resultados[nombre] = {
            "prediccion": pred,
            "probabilidades": prob,
        }
    return resultados

def procesar_llm(log_sospechoso):
    from urllib.parse import unquote_plus
    # Solo los campos que sirven para decidir; IP, id y status confunden a un modelo pequeño
    try:
        r = json.loads(log_sospechoso)
        metodo = str(r.get("method") or "")
        uri = unquote_plus(str(r.get("uri") or ""))
        body = unquote_plus(str(r.get("body") or ""))
    except (json.JSONDecodeError, TypeError, AttributeError):
        metodo, uri, body = "", str(log_sospechoso), ""

    prompt = f"""Eres un analista de seguridad que revisa peticiones HTTP de una aplicación web.

REGLA PRINCIPAL: la mayoría de las peticiones son legítimas. Marca es_alerta=true SOLO si la URI o el cuerpo contienen una carga maliciosa explícita. Una ruta o un parámetro normal (como ?id=1, /my-account, /manifest.json) NO es un ataque.

Señales de ataque:
- SQL Injection: comillas con operadores (' OR '1'='1), UNION SELECT, comentarios (-- o #), DROP/INSERT/SELECT dentro de un parámetro.
- XSS: <script>, onerror=, onload=, javascript: dentro de un parámetro.
- Path Traversal: ../ repetido, /etc/passwd, ..\\windows.
- Otros: inyección de comandos (; cat, | whoami, $(...)).

Ejemplos:
Petición: GET /api/Products?q=apple
{{"razon": "Búsqueda normal sin caracteres sospechosos", "es_alerta": false, "tipo_ataque": "Ninguno"}}

Petición: GET /rest/products/search?q=' UNION SELECT username,password FROM users--
{{"razon": "UNION SELECT dentro del parámetro q", "es_alerta": true, "tipo_ataque": "SQL Injection"}}

Petición: GET /manifest.json
{{"razon": "Recurso estático legítimo", "es_alerta": false, "tipo_ataque": "Ninguno"}}

Petición: GET /search?q=<script>alert(1)</script>
{{"razon": "Etiqueta script en el parámetro q", "es_alerta": true, "tipo_ataque": "XSS"}}

Petición: GET /download?file=../../../../etc/passwd
{{"razon": "Secuencia ../ para leer /etc/passwd", "es_alerta": true, "tipo_ataque": "Path Traversal"}}

Petición: GET /my-account/profile
{{"razon": "Ruta de navegación normal", "es_alerta": false, "tipo_ataque": "Ninguno"}}

Petición: POST /login  cuerpo: email=admin' OR '1'='1&password=x
{{"razon": "Condición OR '1'='1' en el campo email", "es_alerta": true, "tipo_ataque": "SQL Injection"}}

Ahora analiza esta petición. Responde ÚNICAMENTE con un JSON con las claves razon (máximo 15 palabras), es_alerta y tipo_ataque.

Petición: {metodo} {uri}  cuerpo: {body if body else "(vacío)"}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",                      # fuerza a Ollama a retornar solo sintaxis JSON
        "options": {"temperature": 0, "num_predict": 150}
    }

    try:
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=30)
        response.raise_for_status()

        data_ollama = response.json()
        texto_generado = data_ollama.get("response", "")
        resultado_json = json.loads(texto_generado)

        es_alerta = bool(resultado_json.get("es_alerta", False))
        tipo_ataque = resultado_json.get("tipo_ataque", "Desconocido")
        razon = resultado_json.get("razon", "Sin detalle")

        # Coherencia: no hay alerta si el propio modelo dice que no hay tipo de ataque
        if str(tipo_ataque).strip().lower() in ("ninguno", "none", ""):
            es_alerta = False

        return {
            "es_alerta": es_alerta,
            "tipo_ataque": tipo_ataque,
            "razon": razon
        }

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Fallo de conexión con Ollama: {e}")
        return {"es_alerta": False, "tipo_ataque": "Error", "razon": "Error HTTP"}
    except json.JSONDecodeError as e:
        print(f"[ERROR] El LLM no devolvió un JSON parseable: {e}")
        return {"es_alerta": False, "tipo_ataque": "Error", "razon": "Respuesta inválida"}

# alerta
def alerta(id_alerta, ip_alerta, modelo, explicacion):
    alerta_data = {
        "id_alerta": id_alerta,
        "ip_alerta": ip_alerta,
        "modelo": modelo,
        "explicacion": explicacion
    }
    print("[ALERTA DETECTADA]:", json.dumps(alerta_data), flush=True)

# Funciona para procesar log solo con modelos de machine learning
def procesar_log_solo_ML(linea):
    r = json.loads(linea)
    X = preprocesar(r)
    resultado = procesar(X)
    print(f"LOG: {r.get('method')} {r.get('uri')} (id={r.get('id')})", flush=True)
    print(" -> RESULTADO:", json.dumps(resultado), flush=True)

    # modelos 
    for nombre, datos in resultado.items():
        prediccion = datos["prediccion"]
        probabilidades = datos["probabilidades"]
        if prediccion == 1:
            confiabilidad = probabilidades[prediccion] if probabilidades else 0.0
            alerta(
                id_alerta=r.get("id"),
                ip_alerta=r.get("ip"),
                modelo=nombre,
                explicacion = f"{confiabilidad * 100:.2f} %"
            )

# Funciona para procesar log solo con modelo LLM
def procesar_log_solo_LLM(linea):
    r = json.loads(linea)
    print(f"LOG: {r.get('method')} {r.get('uri')} (id={r.get('id')})", flush=True)
    resultado = procesar_llm(linea)
    print(" -> RESULTADO LLM:", json.dumps(resultado, ensure_ascii=False), flush=True)
    if resultado["es_alerta"]:
        alerta(
            id_alerta=r.get("id"),
            ip_alerta=r.get("ip"),
            modelo=OLLAMA_MODEL,
            explicacion=r.get("razon"),
        )

# Funciona para procesar log con ML como filtro y LLM como confirmación
def procesar_log_hibrida(linea):
    r = json.loads(linea)
    X = preprocesar(r)
    resultado = procesar(X)
    print(f"LOG: {r.get('method')} {r.get('uri')} (id={r.get('id')})", flush=True)
    print(" -> RESULTADO ML:", json.dumps(resultado), flush=True)

    modelos_sospechosos = []
    confiabilidad = 0.0
    for nombre, datos in resultado.items():
        if datos["prediccion"] == 1:
            modelos_sospechosos.append(nombre)
            probabilidades = datos["probabilidades"]
            if probabilidades:
                confiabilidad = max(confiabilidad, probabilidades[1])

    if not modelos_sospechosos:
        print(" -> ML: legítimo, no pasa al LLM", flush=True)
        return

    resultado_llm = procesar_llm(linea)
    print(" -> RESULTADO LLM:", json.dumps(resultado_llm, ensure_ascii=False), flush=True)

    if resultado_llm["es_alerta"]:
        alerta(
            id_alerta=r.get("id"),
            ip_alerta=r.get("ip"),
            modelo="+".join(modelos_sospechosos) + "+" + OLLAMA_MODEL,
            explicacion=f"{confiabilidad * 100:.2f}% ->" + r.get("razon"),
        )
    
def procesar_log_hibrida(linea):
    r = json.loads(linea)

def procesar_log(linea):
    if MODO == "ml":
        procesar_log_solo_ML(linea)
    elif MODO == "llm":
        procesar_log_solo_LLM(linea)
    elif MODO == "hibrida":
        procesar_log_hibrida(linea)
    else:
        raise ValueError(f"MODO inválido: {MODO} (usa ml, llm o hibrida)")

def escuchar_nginx():
    print(f"Esperando logs de nginx... (Ctrl+C para salir) [modelo={OLLAMA_MODEL}]", flush=True)

    for linea in iter(sys.stdin.readline, ""):
        linea = linea.strip()
        if not linea:
            continue
        try:
            procesar_log(linea)
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
            print(f"-> ERROR: {e}", file=sys.stderr, flush=True)


def prueba():
    ejemplo = {
        "id": "test-001",
        "method": "GET",
        "uri": "/api/Products",
        "body": "",
        "status": 200,
        "response_size": 512
    }

    print("--- PRUEBA DE EJEMPLO ---")
    try:
        procesar_log_solo_ML(json.dumps(ejemplo))
    except (ValueError, TypeError, KeyError) as e:
        print(f"-> ERROR: {e}")


if __name__ == "__main__":
    if "--test" in sys.argv:
        prueba()
    else:
        escuchar_nginx()
