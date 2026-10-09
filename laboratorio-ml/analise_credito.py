"""Exercício prático (Aula IAML 05): inadimplência de crédito. Uso: python analise_credito.py"""
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_validate, train_test_split

from config import DADOS, PASTA, PROBLEMAS, SEMENTE
from treinar import candidatos, criar_pipeline

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
FIG = PASTA / "figuras"
FIG.mkdir(exist_ok=True)
CV = dict(cv=5, n_jobs=-1)  # n_jobs=-1: folds em paralelo (mesmos resultados, menos tempo)
FN_CUSTO, FP_CUSTO = 8000, 1500  # inadimplente aprovado / bom cliente recusado
R = {}


def salvar(nome, **kw):
    plt.tight_layout(); plt.savefig(FIG / f"{nome}.png", dpi=150, **kw); plt.close()


problema = PROBLEMAS["credito"]
df = pd.read_csv(DADOS / "credito.csv")

# ---------------- Parte 1: análise exploratória ----------------
R["n"], R["prop_inadimplentes"] = len(df), df.inadimplente.mean()
aus = df.isna().sum()[lambda s: s > 0]
R["ausentes"] = aus.to_dict()
R["ausentes_pct"] = (aus / len(df)).to_dict()

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
cont = df.inadimplente.value_counts().sort_index()
ax[0].set_ylim(0, cont.max() * 1.12)
ax[0].bar_label(ax[0].bar(["Pagou (0)", "Inadimplente (1)"], cont, color=["#3b5bdb", "#c92a2a"]),
                labels=[f"{v} ({v / len(df):.1%})" for v in cont], padding=3)
ax[0].set(title="Distribuição do alvo", ylabel="Contratos")
ax[1].bar_label(ax[1].barh(aus.index, aus.values, color="#f08c00"),
                labels=[f"{v} ({v / len(df):.1%})" for v in aus], padding=3)
ax[1].set(title="Valores ausentes por coluna", xlim=(0, aus.max() * 1.3))
salvar("eda1_alvo_ausentes")

faixa = pd.cut(df.score_credito, [299, 500, 600, 700, 800, 1000])
taxa_score = df.groupby(faixa, observed=True).inadimplente.mean()
taxa_fin = df.groupby("finalidade").inadimplente.mean().sort_values(ascending=False)
taxa_imo = df.groupby("possui_imovel").inadimplente.mean()
R |= {"taxa_score": {str(k): v for k, v in taxa_score.items()},
      "taxa_finalidade": taxa_fin.to_dict(), "taxa_imovel": taxa_imo.to_dict()}

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
sns.boxplot(data=df, x="inadimplente", y="score_credito", ax=ax[0], palette=["#3b5bdb", "#c92a2a"])
ax[0].set_title("Score de crédito x inadimplência")
ax[1].bar(taxa_fin.index, taxa_fin * 100, color="#3b5bdb")
ax[1].set(title="Taxa de inadimplência por finalidade", ylabel="%")
ax[1].tick_params(axis="x", rotation=25)
ax[2].bar(taxa_imo.index, taxa_imo * 100, color=["#c92a2a", "#3b5bdb"])
ax[2].set(title="Taxa por posse de imóvel", ylabel="%")
for a in ax[1:]:
    a.bar_label(a.containers[0], fmt="%.1f%%", padding=2)
salvar("eda2_score_finalidade_imovel")

fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(taxa_score.index.astype(str), taxa_score * 100, marker="o", color="#c92a2a")
ax.set(title="Inadimplência por faixa de score", ylabel="%", xlabel="faixa de score")
salvar("eda3_faixa_score")

num_cols = [f for f, r in problema["features"].items() if r["tipo"] == "numero"]
corr = df[num_cols + ["inadimplente"]].corr()["inadimplente"].drop("inadimplente").sort_values()
R["correlacao"] = corr.to_dict()
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.barh(corr.index, corr, color=np.where(corr < 0, "#3b5bdb", "#c92a2a"))
ax.set_title("Correlação das features numéricas com o alvo")
salvar("eda4_correlacao")

# ---------------- Parte 2: modelagem ----------------
X, y = df[list(problema["features"])], df[problema["alvo"]]  # id_contrato e alvo ficam de fora
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=SEMENTE, stratify=y)
R["tamanhos"] = {"treino": len(X_tr), "teste": len(X_te),
                 "inad_treino": float(y_tr.mean()), "inad_teste": float(y_te.mean())}
algoritmos = candidatos("classificacao")
R["cv"] = {}
for nome, modelo in algoritmos.items():
    s = cross_validate(criar_pipeline(problema, clone(modelo)), X_tr, y_tr, scoring="roc_auc", **CV)["test_score"]
    R["cv"][nome] = {"media": float(s.mean()), "desvio": float(s.std())}
    print(f"{nome:<20} {s.mean():.4f} ± {s.std():.4f}")
R["melhor"] = melhor = max(R["cv"], key=lambda a: R["cv"][a]["media"])
novo = lambda p: criar_pipeline(p, clone(algoritmos[melhor]))  # clone: nunca reaproveita um modelo já treinado

# ---------------- Parte 3: avaliação no teste ----------------
pipe = novo(problema).fit(X_tr, y_tr)
prob_te = pipe.predict_proba(X_te)[:, 1]
prev = (prob_te >= .5).astype(int)
cm = confusion_matrix(y_te, prev)
auc_te = roc_auc_score(y_te, prob_te)
R["teste"] = {"acuracia": accuracy_score(y_te, prev), "precisao": precision_score(y_te, prev),
              "recall": recall_score(y_te, prev), "f1": f1_score(y_te, prev),
              "roc_auc": auc_te, "matriz": cm.tolist()}
print(R["teste"])

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
fpr, tpr, _ = roc_curve(y_te, prob_te)
ax[0].plot(fpr, tpr, color="#3b5bdb", lw=2, label=f"{melhor} (AUC = {auc_te:.3f})")
ax[0].plot([0, 1], [0, 1], "--", color="gray", label="Sorteio (AUC = 0,5)")
ax[0].set(xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos (recall)",
          title="Curva ROC no conjunto de teste")
ax[0].legend(loc="lower right")
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax[1],
            xticklabels=["Previsto: paga", "Previsto: inadimp."], yticklabels=["Real: paga", "Real: inadimp."])
ax[1].set_title("Matriz de confusão (limiar 0,5)")
salvar("avaliacao_roc_confusao")

# ---------------- Parte 4: limiar de decisão (sem usar o teste) ----------------
def tabela_limiares(y_real, prob, limiares):
    """Métricas e custo para vários limiares de uma vez (vetorizado, sem laço de sklearn)."""
    y_real, limiares = np.asarray(y_real, bool), np.asarray(limiares)
    recusa = prob[:, None] >= limiares            # linhas = contratos, colunas = limiares
    vp = (recusa & y_real[:, None]).sum(0)
    fp, fn = (recusa & ~y_real[:, None]).sum(0), (~recusa & y_real[:, None]).sum(0)
    rec = recusa.sum(0)
    return [{"limiar": float(t), "precisao": float(vp[i] / rec[i]) if rec[i] else 0.0,
             "recall": float(vp[i] / y_real.sum()), "recusados": int(rec[i]), "fn": int(fn[i]),
             "fp": int(fp[i]), "custo": int(fn[i] * FN_CUSTO + fp[i] * FP_CUSTO)}
            for i, t in enumerate(limiares)]


prob_cv = cross_val_predict(novo(problema), X_tr, y_tr,
                            method="predict_proba", **CV)[:, 1]
grade = np.round(np.arange(.05, .81, .01), 2)
curva = tabela_limiares(y_tr, prob_cv, grade)
R["limiar_otimo"] = otimo = min(curva, key=lambda d: d["custo"])["limiar"]
tabela = [.10, .15, otimo, .30, .50]
R["tabela_limiares"] = tabela_limiares(y_tr, prob_cv, tabela)
R["tabela_limiares_teste"] = tabela_limiares(y_te, prob_te, tabela)
R["limiar_escolhido_treino"], R["limiar_escolhido_teste"] = (
    tabela_limiares(y, p, [otimo])[0] for y, p in [(y_tr, prob_cv), (y_te, prob_te)])
for k, y_ in [("baselines_teste", y_te), ("baselines_treino", y_tr)]:
    R[k] = {"aprovar_todos": int(y_.sum()) * FN_CUSTO, "recusar_todos": int((y_ == 0).sum()) * FP_CUSTO}
print("limiar ótimo (CV no treino):", otimo, R["limiar_escolhido_treino"], R["limiar_escolhido_teste"])

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(grade, [d["custo"] / 1e6 for d in curva], color="#3b5bdb", lw=2)
ax.axvline(otimo, color="#c92a2a", ls="--", label=f"Limiar escolhido = {otimo:.2f}")
ax.set(xlabel="Limiar de recusa (prob. ≥ limiar → recusar)", ylabel="Custo total (R$ milhões)",
       title="Custo total x limiar (probabilidades de validação cruzada no treino)")
ax.legend()
salvar("limiar_custo")

# ---------------- Bônus ----------------
def com_comprometimento(d):  # parcela estimada / renda: só usa colunas da própria linha (sem vazamento)
    return d.assign(comprometimento_renda=d.valor_emprestimo * 1.33 / d.prazo_meses / d.renda_mensal)


ext = {"features": {**problema["features"], "comprometimento_renda": {"tipo": "numero"}}}
Xc_tr, Xc_te = com_comprometimento(X_tr), com_comprometimento(X_te)
cv_auc = lambda p, Xa: cross_validate(p, Xa, y_tr, scoring="roc_auc", **CV)["test_score"]
base_cv = cv_auc(novo(problema), X_tr)
ext_cv = cv_auc(novo(ext), Xc_tr)
pe = novo(ext).fit(Xc_tr, y_tr)
R["bonus_comprometimento"] = {
    "base": [float(base_cv.mean()), float(base_cv.std())],
    "com_feature": [float(ext_cv.mean()), float(ext_cv.std())], "teste_base": auc_te,
    "teste_com_feature": float(roc_auc_score(y_te, pe.predict_proba(Xc_te)[:, 1]))}

gs = GridSearchCV(criar_pipeline(ext, LogisticRegression(max_iter=1000)),
                  {"modelo__C": [.01, .1, 1, 10, 100]}, scoring="roc_auc", **CV).fit(Xc_tr, y_tr)
R["bonus_grid"] = {"melhor": gs.best_params_, "cv_auc": float(gs.best_score_),
                   "teste_auc": float(roc_auc_score(y_te, gs.predict_proba(Xc_te)[:, 1]))}


def met(**kw):
    m = criar_pipeline(ext, LogisticRegression(max_iter=1000, **kw)).fit(Xc_tr, y_tr)
    p = m.predict_proba(Xc_te)[:, 1]
    r = (p >= .5).astype(int)
    return {"precisao": float(precision_score(y_te, r)), "recall": float(recall_score(y_te, r)),
            "f1": float(f1_score(y_te, r)), "auc": float(roc_auc_score(y_te, p)),
            "custo": tabela_limiares(y_te, p, [.5])[0]["custo"]}


R["bonus_class_weight"] = {"sem": met(), "balanced": met(class_weight="balanced")}
print(json.dumps({k: v for k, v in R.items() if k.startswith("bonus")}, indent=1, ensure_ascii=False))

# coeficientes do modelo final (interpretabilidade)
R["coeficientes"] = pd.Series(pipe.named_steps["modelo"].coef_[0],
                              index=pipe.named_steps["preprocessamento"].get_feature_names_out()
                              ).sort_values().to_dict()
(PASTA / "resultados_credito.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=float),
                                              encoding="utf-8")
