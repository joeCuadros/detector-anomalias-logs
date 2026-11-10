"""Panel web del detector: sirve index.html y los eventos que escribe detector.py (solo biblioteca estándar)."""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
EVENTOS = Path(os.getenv("EVENTOS", "/datos/eventos.jsonl"))
PUERTO = int(os.getenv("PUERTO", "8090"))
MAX_EVENTOS = 400  # el panel muestra los más recientes


def leer_eventos():
    if not EVENTOS.exists():
        return []
    lineas = EVENTOS.read_text(encoding="utf-8", errors="replace").splitlines()
    eventos = []
    for linea in lineas:
        try:
            eventos.append(json.loads(linea))
        except json.JSONDecodeError:
            continue  # línea a medio escribir
    return eventos


class Manejador(BaseHTTPRequestHandler):
    def responder(self, codigo, tipo, cuerpo):
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_GET(self):
        ruta = self.path.split("?", 1)[0]
        if ruta in ("/", "/index.html"):
            self.responder(200, "text/html; charset=utf-8", (RAIZ / "index.html").read_bytes())
        elif ruta == "/api/eventos":
            ev = leer_eventos()
            datos = {"total": len(ev), "eventos": ev[-MAX_EVENTOS:], "resumen": resumen(ev)}
            self.responder(200, "application/json; charset=utf-8", json.dumps(datos, ensure_ascii=False).encode("utf-8"))
        else:
            self.responder(404, "text/plain; charset=utf-8", b"no encontrado")

    def log_message(self, *args):
        pass  # sin registro por petición: el panel consulta cada segundo


def resumen(ev):
    """Totales sobre todos los eventos, no solo los mostrados."""
    r = {"total": len(ev), "legitima": 0, "descartada": 0, "alerta": 0, "no_confirmada": 0, "enviadas_llm": 0, "ms_ml": [], "ms_llm": []}
    for e in ev:
        r[e.get("veredicto", "legitima")] = r.get(e.get("veredicto", "legitima"), 0) + 1
        if e.get("ms_ml") is not None:
            r["ms_ml"].append(e["ms_ml"])
        if e.get("ms_llm") is not None:
            r["ms_llm"].append(e["ms_llm"])
            r["enviadas_llm"] += 1
    for k in ("ms_ml", "ms_llm"):
        v = r[k]
        r[k] = sum(v) / len(v) if v else None
    return r


if __name__ == "__main__":
    print(f"Panel en http://127.0.0.1:{PUERTO} (eventos: {EVENTOS})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PUERTO), Manejador).serve_forever()
