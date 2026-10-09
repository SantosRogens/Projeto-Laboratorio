"""Gera o relatório PDF do exercício a partir de resultados_credito.json e figuras/."""
import json
import os

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

R = json.load(open("resultados_credito.json", encoding="utf-8"))
ss = getSampleStyleSheet()
N = ParagraphStyle("N", parent=ss["Normal"], fontSize=10, leading=14, spaceAfter=6)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=14, spaceBefore=12, spaceAfter=6,
                    textColor=colors.HexColor("#1d2330"))
T = ParagraphStyle("T", parent=ss["Title"], fontSize=18, leading=22)
SM = ParagraphStyle("SM", parent=N, fontSize=8.5, leading=11, textColor=colors.HexColor("#5b6475"))
LW = 17 * cm
C = "<font face='Courier'>{}</font>"


def cod(t):
    return C.format(t)


def fig(path, largura=LW):
    w, h = PILImage.open(path).size
    return Image(path, width=largura, height=largura * h / w)


def tabela(dados, larguras=None, destaque=None):
    t = Table(dados, colWidths=larguras, repeatRows=1)
    est = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a5568")),
           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
           ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
           ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#cbd2dc")),
           ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f5f8")])]
    if destaque is not None:
        est += [("BACKGROUND", (0, destaque), (-1, destaque), colors.HexColor("#dbe4ff")),
                ("FONTNAME", (0, destaque), (-1, destaque), "Helvetica-Bold")]
    t.setStyle(TableStyle(est))
    return t


def pct(v, c=1):
    return f"{v * 100:.{c}f}%".replace(".", ",")


def n3(v):
    return f"{v:.3f}".replace(".", ",")


def n4(v):
    return f"{v:.4f}".replace(".", ",")


def brl(v):
    return "R$ " + f"{v:,.0f}".replace(",", ".")


def lim(v):
    return f"{v:.2f}".replace(".", ",")


S = []
S += [Paragraph("Exercício Prático 2 — Aprendizado Supervisionado", T),
      Paragraph("<b>Predição de inadimplência de crédito</b> · Aula IAML 05 (Prática 2) · "
                "Inteligência Artificial e Machine Learning", N),
      Paragraph("Aluno(a): ______________________________ &nbsp;&nbsp; RA: ______________", N),
      Paragraph(f"Dados sintéticos ({cod('data/credito.csv')}, gerado por {cod('gerar_dados.py')}, semente 42). "
                f"Todos os números foram produzidos por {cod('analise_credito.py')} e são reprodutíveis. "
                "Ambiente: Python 3.12, scikit-learn 1.8.0 (a aula fixa 1.9.1; os resultados de churn e imóveis "
                "coincidem com os da aula).", SM)]

# ---------- Parte 1
S += [Paragraph("Parte 1 — Análise exploratória", H1)]
n = R["n"]
S += [Paragraph(f"<b>Proporção de inadimplentes.</b> Dos {n} contratos, {pct(R['prop_inadimplentes'])} são "
                f"inadimplentes (1) e {pct(1 - R['prop_inadimplentes'])} pagaram (0). As classes são "
                "desbalanceadas (cerca de 1 para 4), então a acurácia sozinha engana: aprovar todos os pedidos "
                "já acertaria cerca de 79%.", N)]
S += [Paragraph(f"<b>Valores ausentes.</b> Apenas duas colunas: {cod('renda_mensal')} com "
                f"{R['ausentes']['renda_mensal']} ausentes ({pct(R['ausentes_pct']['renda_mensal'])}) e "
                f"{cod('tempo_emprego_anos')} com {R['ausentes']['tempo_emprego_anos']} "
                f"({pct(R['ausentes_pct']['tempo_emprego_anos'])}). São poucos, então imputamos (mediana) dentro "
                "do Pipeline em vez de descartar linhas.", N)]
S += [fig("figuras/eda1_alvo_ausentes.png")]
ts, tf, ti = R["taxa_score"], R["taxa_finalidade"], R["taxa_imovel"]
S += [Paragraph("<b>Como a inadimplência varia.</b>", N),
      Paragraph(f"• <b>Score:</b> relação forte e monotônica. Faixa 300–500: {pct(ts['(299, 500]'])}; "
                f"500–600: {pct(ts['(500, 600]'])}; 600–700: {pct(ts['(600, 700]'])}; "
                f"700–800: {pct(ts['(700, 800]'])}; acima de 800: {pct(ts['(800, 1000]'])}. "
                "É o melhor preditor individual (correlação de −0,32 com o alvo).", N),
      Paragraph(f"• <b>Finalidade:</b> <i>negócio</i> se destaca ({pct(tf['negocio'])}); as demais ficam entre "
                f"{pct(tf['educacao'])} e {pct(tf['reforma'])}.", N),
      Paragraph(f"• <b>Posse de imóvel:</b> quem não possui imóvel inadimpliu mais ({pct(ti['nao'])} contra "
                f"{pct(ti['sim'])}); efeito moderado.", N),
      fig("figuras/eda2_score_finalidade_imovel.png")]
S += [KeepTogether([fig("figuras/eda3_faixa_score.png", 8.3 * cm)]), Spacer(1, 4)]
S += [Paragraph(f"<b>Alguma coluna deve ser descartada?</b> Sim: {cod('id_contrato')}. É um identificador único "
                "sem poder preditivo; mantê-lo permitiria ao modelo (sobretudo às árvores) decorar contratos "
                f"individuais, gerando overfitting. O alvo {cod('inadimplente')} também fica fora de X. As demais "
                "colunas existem no momento da solicitação do empréstimo, então não há indício de vazamento de "
                "dados.", N)]

# ---------- Parte 2
S += [Paragraph("Parte 2 — Modelagem", H1)]
tm = R["tamanhos"]
S += [Paragraph(f"Separação 80/20, estratificada, {cod('random_state=42')}: {tm['treino']} contratos de treino "
                f"({pct(tm['inad_treino'])} inadimplentes) e {tm['teste']} de teste ({pct(tm['inad_teste'])}). "
                "O Pipeline (ColumnTransformer) imputa pela mediana e padroniza as numéricas; imputa pela moda e "
                f"aplica one-hot nas categóricas ({cod('handle_unknown=\"ignore\"')}). Medianas, médias e categorias "
                "são aprendidas só com o treino de cada fold, o que evita vazamento. Validação cruzada de 5 folds, "
                "métrica ROC AUC:", N)]
nomes = list(R["cv"])
linhas = [["Algoritmo", "ROC AUC (média ± desvio)"]]
for a in nomes:
    linhas.append([a, f"{n4(R['cv'][a]['media'])} ± {n4(R['cv'][a]['desvio'])}"])
S += [tabela(linhas, [7 * cm, 6 * cm], destaque=1 + nomes.index(R["melhor"])), Spacer(1, 6)]
S += [Paragraph(f"<b>Escolhido: {R['melhor']}.</b> A vantagem sobre Random Forest e Gradient Boosting é pequena "
                "em relação ao desvio entre folds (~0,022). A relação entre as features e o risco é "
                "aproximadamente linear nestes dados, então os modelos mais complexos só acrescentaram variância "
                "(mesma lição do churn na aula). A regressão logística também é interpretável, o que importa em "
                "crédito (Parte 6). A Árvore de Decisão (profundidade 5) foi a pior.", N)]

# ---------- Parte 3
S += [Paragraph("Parte 3 — Avaliação no conjunto de teste", H1)]
t = R["teste"]
(vn, fp), (fn, vp) = t["matriz"]
S += [Paragraph("O teste foi usado uma única vez, com o modelo já escolhido e o limiar padrão 0,5:", N)]
S += [tabela([["Acurácia", "Precisão", "Recall", "F1", "ROC AUC"],
              [pct(t["acuracia"]), pct(t["precisao"]), pct(t["recall"]), n3(t["f1"]), n3(t["roc_auc"])]]),
      Spacer(1, 6)]
S += [tabela([["", "Previsto: paga", "Previsto: inadimplente"],
              ["Real: pagou", f"{vn} (VN)", f"{fp} (FP)"],
              ["Real: inadimplente", f"{fn} (FN)", f"{vp} (VP)"]], [4.5 * cm, 4 * cm, 4.5 * cm]), Spacer(1, 6)]
S += [fig("figuras/avaliacao_roc_confusao.png")]
S += [Paragraph(f"<b>Leitura.</b> A acurácia de {pct(t['acuracia'])} parece boa, mas engana: "
                f"{pct(1 - tm['inad_teste'])} dos contratos pagam, então aprovar todos já daria "
                f"{pct(1 - tm['inad_teste'])}. O que pesa é o recall de apenas {pct(t['recall'])}: com o limiar "
                f"0,5 o modelo deixa passar {fn} dos {fn + vp} inadimplentes. O ROC AUC de {n3(t['roc_auc'])} "
                "mostra boa capacidade de ordenar o risco (e está longe de 0,95, então não há suspeita de "
                "vazamento); o problema é o corte de 0,5, conservador demais para este negócio.", N)]
S += [Paragraph("<b>Falso positivo (FP):</b> o modelo prevê inadimplência e a fintech recusa, mas o cliente "
                "pagaria. É um <b>bom cliente recusado</b>; o custo é o lucro não obtido (R$ 1.500).", N),
      Paragraph("<b>Falso negativo (FN):</b> o modelo prevê que o cliente pagará e a fintech aprova, mas ele não "
                "paga. É um <b>inadimplente aprovado</b>; o custo é o prejuízo (R$ 8.000).", N),
      Paragraph("<b>Qual é mais caro?</b> O falso negativo: cada um custa 8.000 / 1.500 ≈ 5,3 vezes mais que um "
                "falso positivo. Por isso o recall pesa mais que a precisão aqui, e o limiar deve ficar abaixo "
                "de 0,5.", N)]

# ---------- Parte 4
S += [Paragraph("Parte 4 — Limiar de decisão", H1)]
S += [Paragraph(f"As probabilidades vieram de {cod('cross_val_predict(pipeline, X_treino, y_treino, cv=5, method=&quot;predict_proba&quot;)')}: "
                "cada previsão é feita por um modelo que não viu aquele contrato, e o teste não foi tocado. "
                "Regra: recusar se probabilidade ≥ limiar. Custo total = 8.000 × FN + 1.500 × FP.", N)]
lin = [["Limiar", "Precisão", "Recall", "Recusados", "FN", "FP", "Custo (treino, CV)"]]
dest = None
for i, r in enumerate(R["tabela_limiares"], 1):
    lin.append([lim(r["limiar"]), pct(r["precisao"]), pct(r["recall"]), r["recusados"], r["fn"], r["fp"],
                brl(r["custo"])])
    if abs(r["limiar"] - R["limiar_otimo"]) < 1e-9:
        dest = i
S += [tabela(lin, [1.8 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm, 1.6 * cm, 1.6 * cm, 4 * cm], destaque=dest),
      Spacer(1, 4)]
b = R["baselines_treino"]
S += [Paragraph(f"Referências no treino: aprovar todos custaria {brl(b['aprovar_todos'])} e recusar todos, "
                f"{brl(b['recusar_todos'])}.", SM)]
S += [fig("figuras/limiar_custo.png", 12.5 * cm)]
lt, bt = R["limiar_escolhido_teste"], R["baselines_teste"]
c05 = R["tabela_limiares_teste"][-1]["custo"]
L = lim(R["limiar_otimo"])
S += [Paragraph(f"<b>Recomendação: limiar de {L}</b> (varredura de 0,05 a 0,80 em passos de 0,01, minimizando o "
                "custo nas probabilidades de validação cruzada). Há um argumento teórico que corrobora: se as "
                "probabilidades forem bem calibradas, recusar compensa quando p × 8.000 &gt; (1 − p) × 1.500, "
                f"ou seja, p &gt; 1.500 / 9.500 ≈ 0,158. O valor empírico ({L}) fica próximo, e a curva de custo "
                "é achatada entre ~0,15 e ~0,22, então a escolha não é frágil.", N)]
S += [Paragraph(f"<b>Custo no conjunto de teste com este limiar:</b> {brl(lt['custo'])} ({lt['fn']} FN e "
                f"{lt['fp']} FP; precisão {pct(lt['precisao'])}, recall {pct(lt['recall'])}). Para comparar: com "
                f"o limiar 0,5 o custo seria {brl(c05)}; aprovar todos, {brl(bt['aprovar_todos'])}; recusar "
                f"todos, {brl(bt['recusar_todos'])}. O limiar escolhido reduz o custo em "
                f"{pct(1 - lt['custo'] / c05, 0)} frente ao 0,5 e em "
                f"{pct(1 - lt['custo'] / bt['aprovar_todos'], 0)} frente a aprovar todos.", N)]
rows = [["Limiar", "Precisão", "Recall", "FN", "FP", "Custo (teste)"]]
for r in R["tabela_limiares_teste"]:
    rows.append([lim(r["limiar"]), pct(r["precisao"]), pct(r["recall"]), r["fn"], r["fp"], brl(r["custo"])])
idx = [round(x["limiar"], 2) for x in R["tabela_limiares_teste"]].index(round(R["limiar_otimo"], 2))
S += [tabela(rows, [2 * cm, 2.4 * cm, 2.4 * cm, 1.8 * cm, 1.8 * cm, 3.6 * cm], destaque=1 + idx),
      Spacer(1, 4)]
S += [Paragraph("Nota: a tabela do teste serve só para <i>reportar</i> o resultado. Nela o limiar 0,15 custaria "
                "um pouco menos, mas a escolha foi feita sem o teste (usá-lo para escolher o limiar o tornaria "
                "otimista). A diferença (cerca de R$ 18 mil) é pequena e compatível com ruído amostral. O custo "
                "no teste (1.200 contratos) é menor que no treino (4.800) por ter 4 vezes menos linhas.", SM)]

# ---------- Parte 5
S += [Paragraph("Parte 5 — Integração ao sistema", H1)]
S += [Paragraph(f"Foi adicionado o problema {cod('credito')} ao dicionário {cod('PROBLEMAS')} em {cod('config.py')}, "
                "com as nove features, limites (mín./máx.), valores padrão e opções das categorias "
                f"({cod('possui_imovel')}: sim/nao; {cod('finalidade')}: pessoal, veiculo, reforma, educacao, "
                f"negocio). {cod('id_contrato')} ficou de fora. <b>Não foi necessário alterar {cod('app.py')} "
                f"nem {cod('index.html')}</b>: como tudo é lido de {cod('config.py')}, a API e o formulário "
                f"passaram a incluir a nova aba automaticamente. Ao rodar {cod('python treinar.py')}, o terceiro "
                f"modelo foi treinado e salvo em {cod('models/credito.joblib')}.", N)]
S += [Paragraph("As telas usam o limiar 0,5 para a cor da interface; para a decisão de negócio vale o limiar da "
                "Parte 4.", N)]
S += [Paragraph("<b>Risco alto</b> — 28 anos, renda R$ 1.800, 6 meses de emprego, score 380, 5 dívidas ativas, "
                "sem imóvel, finalidade negócio, R$ 30.000 em 12 meses: <b>98,9%</b> de probabilidade de "
                "inadimplência.", N),
      fig("figuras/app_credito_risco_alto.png", 14 * cm)]
S += [Paragraph("<b>Risco baixo</b> — 45 anos, renda R$ 12.000, 12 anos de emprego, score 850, sem dívidas, com "
                "imóvel, educação, R$ 10.000 em 36 meses: <b>0,4%</b> de probabilidade de inadimplência.", N),
      fig("figuras/app_credito_risco_baixo.png", 14 * cm)]

# ---------- Parte 6
S += [Paragraph("Parte 6 — Reflexão ética", H1)]
S += [Paragraph(
    "Modelos de crédito aprendem com decisões do passado; se um grupo já foi preterido por preconceito, o modelo "
    "reproduz e escala essa discriminação, ainda com aparência de objetividade. O risco existe mesmo sem variáveis "
    "protegidas, porque outras funcionam como <i>proxies</i>: CEP ou bairro podem refletir renda, raça e etnia, e a "
    "idade pode prejudicar jovens e idosos. Por isso eu <b>não usaria</b>, mesmo que melhorassem o desempenho, "
    "gênero, raça/cor, religião, orientação sexual, estado civil, nome, dados de saúde nem CEP/bairro; a idade só "
    "com justificativa e auditoria (ela está no dataset do exercício, mas eu a revisaria antes de ir a produção). "
    "Também mediria taxa de aprovação, recall e falsos positivos por grupo, para detectar tratamento desigual, e "
    "preferiria um modelo interpretável como a regressão logística, cujo ganho de desempenho frente aos complexos "
    "foi nulo aqui. Pela LGPD (Lei 13.709/2018): o tratamento exige finalidade específica e base legal (a proteção "
    "do crédito é uma das hipóteses do art. 7º, X), com coleta apenas dos dados necessários (art. 6º: finalidade, "
    "necessidade e não discriminação); dados sensíveis têm regime mais restrito (art. 11); e o art. 20 garante ao "
    "titular o direito de pedir a revisão de decisões tomadas unicamente por tratamento automatizado e de obter "
    "informações claras sobre os critérios usados. Na prática, o modelo deve apoiar a equipe de crédito (humano no "
    "circuito), com explicação do motivo da recusa e canal de contestação, e o desempenho e o viés devem ser "
    "monitorados ao longo do tempo.", N)]

# ---------- Bônus
S += [Paragraph("Desafios extras (bônus)", H1)]
bc, bg, bw = R["bonus_comprometimento"], R["bonus_grid"], R["bonus_class_weight"]
S += [Paragraph("<b>1. Feature \"comprometimento de renda\"</b> (parcela estimada = valor × 1,33 / prazo, dividida "
                "pela renda). Na validação cruzada o ROC AUC praticamente não mudou: "
                f"{n4(bc['base'][0])} ± {n4(bc['base'][1])} sem a feature contra "
                f"{n4(bc['com_feature'][0])} ± {n4(bc['com_feature'][1])} com ela. No teste foi de "
                f"{n3(bc['teste_base'])} para {n3(bc['teste_com_feature'])}, diferença pequena e dentro do ruído "
                "(o desvio entre folds é ~0,022). Conclusão: nestes dados e com este modelo não há ganho "
                "confiável; este experimento sozinho não permite afirmar a causa.", N)]
S += [Paragraph(f"<b>2. GridSearchCV</b> (regressão logística com a nova feature; C em 0,01; 0,1; 1; 10; 100; "
                f"5 folds): melhor C = {bg['melhor']['modelo__C']}, AUC de CV {n4(bg['cv_auc'])} e de teste "
                f"{n3(bg['teste_auc'])}. Ganho desprezível: o modelo já estava bem regularizado.", N)]
sem, bal = bw["sem"], bw["balanced"]
S += [Paragraph("<b>3. class_weight=\"balanced\"</b> (com a nova feature, limiar 0,5, no teste):", N)]
S += [tabela([["Configuração", "Precisão", "Recall", "F1", "ROC AUC", "Custo"],
              ["Sem class_weight", pct(sem["precisao"]), pct(sem["recall"]), n3(sem["f1"]), n3(sem["auc"]),
               brl(sem["custo"])],
              ["balanced", pct(bal["precisao"]), pct(bal["recall"]), n3(bal["f1"]), n3(bal["auc"]),
               brl(bal["custo"])]],
             [4 * cm, 2.4 * cm, 2.4 * cm, 2 * cm, 2.4 * cm, 3.2 * cm]), Spacer(1, 6)]
S += [Paragraph("O balanceamento <b>troca precisão por recall</b>: o recall sobe bastante, a precisão cai e o ROC "
                "AUC quase não muda, ou seja, o ranqueamento é o mesmo e o efeito equivale a deslocar o limiar "
                "para baixo. Como o falso negativo custa mais, o custo melhora em relação ao 0,5 sem "
                "balanceamento, mas escolher o limiar pelo custo (Parte 4) é mais direto e controlável.", N)]

# ---------- Arquivos
S += [Paragraph("Arquivos entregues", H1)]
S += [Paragraph(f"{cod('config.py')} (com {cod('credito')}) · {cod('gerar_dados.py')} · {cod('treinar.py')} · "
                f"{cod('app.py')} · {cod('templates/index.html')} · {cod('analise_credito.py')} (Partes 1 a 4 e "
                f"bônus) · {cod('resultados_credito.json')} · {cod('figuras/')}. Reprodução: "
                f"{cod('python gerar_dados.py')}, {cod('python treinar.py')}, {cod('python analise_credito.py')} "
                f"e {cod('python app.py')}.", N)]


def rodape(c, d):
    c.saveState()
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.grey)
    c.drawString(2 * cm, 1.2 * cm, "Exercício Prático 2 — IAML 05 — Inadimplência de crédito")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Pág. {d.page}")
    c.restoreState()


os.makedirs("/mnt/user-data/outputs", exist_ok=True)
SimpleDocTemplate("/mnt/user-data/outputs/relatorio_exercicio_credito.pdf", pagesize=A4,
                  leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm,
                  title="Exercício Prático 2 — Inadimplência de crédito").build(
    S, onFirstPage=rodape, onLaterPages=rodape)
print("ok")
