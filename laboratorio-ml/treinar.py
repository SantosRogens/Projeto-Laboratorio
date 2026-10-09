"""Compara algoritmos, avalia no teste e salva os modelos. Uso: python treinar.py"""
import json

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score,
                             root_mean_squared_error)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from config import DADOS, MODELOS, PROBLEMAS, SEMENTE


def candidatos(tipo):
    if tipo == "classificacao":
        return {
            "Regressão Logística": LogisticRegression(max_iter=1000),
            "Árvore de Decisão": DecisionTreeClassifier(max_depth=5, random_state=SEMENTE),
            "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10,
                                                    min_samples_leaf=5, random_state=SEMENTE),
            "Gradient Boosting": HistGradientBoostingClassifier(learning_rate=.05, random_state=SEMENTE),
        }
    return {
        "Regressão Linear": LinearRegression(),
        "Árvore de Decisão": DecisionTreeRegressor(max_depth=8, random_state=SEMENTE),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=14,
                                               min_samples_leaf=3, random_state=SEMENTE),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=.05,
                                                           random_state=SEMENTE),
    }


def criar_pipeline(problema, modelo):
    features = problema["features"]
    numericas = [f for f in features if features[f]["tipo"] == "numero"]
    categoricas = [f for f in features if features[f]["tipo"] == "categoria"]
    preprocessamento = ColumnTransformer([
        ("numericas", Pipeline([("preencher", SimpleImputer(strategy="median")),
                                ("padronizar", StandardScaler())]), numericas),
        ("categoricas", Pipeline([("preencher", SimpleImputer(strategy="most_frequent")),
                                  ("one_hot", OneHotEncoder(handle_unknown="ignore"))]), categoricas),
    ])
    return Pipeline([("preprocessamento", preprocessamento), ("modelo", modelo)])


def dividir(problema):
    df = pd.read_csv(DADOS / problema["arquivo"])
    X = df[list(problema["features"])]  # só as features: o id e o alvo ficam de fora
    y = df[problema["alvo"]]
    estratificar = y if problema["tipo"] == "classificacao" else None
    return train_test_split(X, y, test_size=.2, random_state=SEMENTE, stratify=estratificar)


def avaliar(problema, modelo, X_teste, y_teste):
    if problema["tipo"] == "classificacao":
        prob = modelo.predict_proba(X_teste)[:, 1]
        prev = (prob >= .5).astype(int)
        return {"acuracia": accuracy_score(y_teste, prev), "precisao": precision_score(y_teste, prev),
                "recall": recall_score(y_teste, prev), "f1": f1_score(y_teste, prev),
                "roc_auc": roc_auc_score(y_teste, prob),
                "matriz_confusao": confusion_matrix(y_teste, prev).tolist()}
    prev = modelo.predict(X_teste)
    return {"mae": mean_absolute_error(y_teste, prev), "rmse": root_mean_squared_error(y_teste, prev),
            "r2": r2_score(y_teste, prev)}


def treinar(nome, problema):
    print(f"\n=== {problema['titulo']} ===")
    X_treino, X_teste, y_treino, y_teste = dividir(problema)
    classificacao = problema["tipo"] == "classificacao"
    metrica = "roc_auc" if classificacao else "neg_root_mean_squared_error"
    comparacao = {}
    for algoritmo, modelo in candidatos(problema["tipo"]).items():
        cv = cross_validate(criar_pipeline(problema, modelo), X_treino, y_treino, cv=5, scoring=metrica)
        notas = cv["test_score"] if classificacao else -cv["test_score"]  # RMSE vem negativo
        comparacao[algoritmo] = {"media": notas.mean(), "desvio": notas.std()}
        print(f"  {algoritmo:<20} {notas.mean():.4f} ± {notas.std():.4f}")
    escolhido = (max if classificacao else min)(comparacao, key=lambda a: comparacao[a]["media"])
    modelo = criar_pipeline(problema, candidatos(problema["tipo"])[escolhido]).fit(X_treino, y_treino)
    teste = avaliar(problema, modelo, X_teste, y_teste)
    print(f"  Escolhido: {escolhido} | teste: {teste}")
    joblib.dump(modelo, MODELOS / f"{nome}.joblib", compress=3)
    return {"modelo": escolhido, "metrica": "ROC AUC" if classificacao else "RMSE",
            "validacao_cruzada": comparacao, "teste": teste, "sklearn": sklearn.__version__}


def main():
    MODELOS.mkdir(exist_ok=True)
    metricas = {nome: treinar(nome, problema) for nome, problema in PROBLEMAS.items()}
    (MODELOS / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False),
                                           encoding="utf-8")


if __name__ == "__main__":
    main()
