import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
import warnings

warnings.filterwarnings("ignore")

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

# alerta
def alerta(id_alerta, ip_alerta, modelo, confiabilidad):
    alerta_data = {
        "id_alerta": id_alerta,
        "ip_alerta": ip_alerta,
        "modelo": modelo,
        "confiabilidad": round(float(confiabilidad) * 100, 2),
    }
    print("[ALERTA DETECTADA]:", json.dumps(alerta_data), flush=True)

# Funciona para procesar log solo con modelos de machine learning
def procesar_log(linea):
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
                confiabilidad=confiabilidad
            )

def escuchar_nginx():
    print("Esperando logs de nginx... (Ctrl+C para salir)", flush=True)

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
        procesar_log(json.dumps(ejemplo))
    except (ValueError, TypeError, KeyError) as e:
        print(f"-> ERROR: {e}")


if __name__ == "__main__":
    if "--test" in sys.argv:
        prueba()
    else:
        escuchar_nginx()
