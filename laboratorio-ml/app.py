"""API e interface web. Uso: python app.py -> http://localhost:5000"""
import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from config import DADOS, MODELOS, PROBLEMAS

if not (MODELOS / "metricas.json").exists():  # primeira execução: gera os dados e treina
    import gerar_dados
    import treinar
    if not (DADOS / "churn.csv").exists():
        gerar_dados.main()
    treinar.main()

METRICAS = json.loads((MODELOS / "metricas.json").read_text(encoding="utf-8"))
MODELOS_CARREGADOS = {nome: joblib.load(MODELOS / f"{nome}.joblib") for nome in PROBLEMAS}

app = Flask(__name__)
app.json.ensure_ascii = False
app.json.sort_keys = False


def validar(problema, dados):
    """Nunca confie na entrada: confere tipos, limites e opções de cada campo."""
    linha, erros = {}, []
    for campo, regra in problema["features"].items():
        valor = dados.get(campo)
        if valor in (None, ""):  # vazio é permitido: o pipeline preenche
            linha[campo] = None
        elif regra["tipo"] == "categoria":
            if valor not in regra["opcoes"]:
                erros.append(f"{campo} deve ser um de: {', '.join(regra['opcoes'])}")
            linha[campo] = valor
        else:
            try:
                linha[campo] = float(valor)
                if not regra["min"] <= linha[campo] <= regra["max"]:
                    erros.append(f"{campo} deve estar entre {regra['min']} e {regra['max']}")
            except (TypeError, ValueError):
                erros.append(f"{campo} deve ser numérico")
    return linha, erros


@app.get("/")
def pagina():
    return render_template("index.html")


@app.get("/api/problemas")
def problemas():
    return jsonify({nome: {**p, "metricas": METRICAS[nome]} for nome, p in PROBLEMAS.items()})


@app.post("/api/prever/<nome>")
def prever(nome):
    if nome not in PROBLEMAS:
        return jsonify(erro="Problema não encontrado"), 404
    problema = PROBLEMAS[nome]
    linha, erros = validar(problema, request.get_json(silent=True) or {})
    if erros:
        return jsonify(erro="; ".join(erros)), 400
    X = pd.DataFrame([linha], columns=list(problema["features"])).astype(
        {c: float for c, r in problema["features"].items() if r["tipo"] == "numero"})
    modelo = MODELOS_CARREGADOS[nome]
    if problema["tipo"] == "classificacao":
        return jsonify(probabilidade=float(modelo.predict_proba(X)[0, 1]))
    return jsonify(valor=float(modelo.predict(X)[0]))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
