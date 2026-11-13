import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Dashboard Experimental — Percolação / Deep Learning",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# FUNÇÕES DE LIMPEZA E NORMALIZAÇÃO
# ==============================================================================
METRICAS_DISPONIVEIS = {
    "Acurácia": "Acuracia_Teste",
    "F1-Score Macro": "F1_Score_Macro",
    "Precisão Macro": "Precisao_Macro",
    "Recall Macro": "Recall_Macro",
    "Loss (Teste)": "Loss_Teste",
    "Tempo de Treino (s)": "Tempo_Treino_Seg",
}

ORDEM_JANELAS = ["Original", "L3", "L5", "L7", "L15", "L21", "L41"]


def normalizar_tipo_entrada(valor: str) -> str:
    """Padroniza pequenas diferenças de escrita entre planilhas CNN e ViT."""
    if pd.isna(valor):
        return "Desconhecido"
    s = str(valor).strip().lower()
    if s.startswith("original"):
        return "Original"

    s = re.sub(r"eucli\b", "eucl", s)
    s = re.sub(r"manha\b", "manh", s)

    s = re.sub(r"^l(\d+)", lambda m: f"L{m.group(1)}", s)
    return s


def extrair_janela(tipo_normalizado: str) -> str:
    """Extrai a escala L (Original, L3, L5, L7, L15, L21, L41)."""
    if tipo_normalizado == "Original":
        return "Original"
    match = re.match(r"^(L\d+)", tipo_normalizado)
    return match.group(1) if match else "Outro"


def extrair_descritor(tipo_normalizado: str) -> str:    
    if tipo_normalizado == "Original":
        return "RGB Original"
    return re.sub(r"^L\d+_", "", tipo_normalizado)


def carregar_planilha(fonte, familia_default: str) -> pd.DataFrame:    
    df = pd.read_excel(fonte, sheet_name="Registro_Treinamento")

    cols_metadados = [
        "Dataset",
        "Tipo_imagem",
        "Tipo_Entrada",
        "Arquitetura",
        "Estrategia_FineTuning",
        "Tamanho_Lote",
        "Otimizador",
        "Taxa_Aprendizado",
    ]
    cols_existentes = [c for c in cols_metadados if c in df.columns]
    df[cols_existentes] = df.groupby("ID_Experimento", sort=False)[cols_existentes].ffill()

    for col in METRICAS_DISPONIVEIS.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Familia"] = df["Arquitetura"].apply(
        lambda x: "Vision Transformer (ViT)"
        if "vit" in str(x).lower()
        else ("CNN" if "cnn" in str(x).lower() else familia_default)
    )
    df["Entrada_Padronizada"] = df["Tipo_Entrada"].apply(normalizar_tipo_entrada)
    df["Janela"] = df["Entrada_Padronizada"].apply(extrair_janela)
    df["Descritor"] = df["Entrada_Padronizada"].apply(extrair_descritor)

    categorias_extras = [j for j in df["Janela"].unique() if j not in ORDEM_JANELAS]
    df["Janela"] = pd.Categorical(
        df["Janela"], categories=ORDEM_JANELAS + categorias_extras, ordered=True
    )
    return df


# ==============================================================================
# SIDEBAR — CARREGAMENTO E FILTROS
# ==============================================================================
st.sidebar.title("Upload das Planilhas de Resultados")

with st.sidebar.expander("Arquivos Excel", expanded=False):
    upload_cnn = st.file_uploader("Planilha CNN (.xlsx)", type=["xlsx"], key="cnn_up")
    upload_vit = st.file_uploader("Planilha ViT (.xlsx)", type=["xlsx"], key="vit_up")

fonte_cnn = upload_cnn if upload_cnn is not None else "Registro_Experimentos_CNN.xlsx"
fonte_vit = upload_vit if upload_vit is not None else "Registro_Experimentos_ViT.xlsx"

if not isinstance(fonte_cnn, st.runtime.uploaded_file_manager.UploadedFile) and not os.path.exists(fonte_cnn):
    st.error(f"Arquivo `{fonte_cnn}` não encontrado. Faça o upload na barra lateral.")
    st.stop()

if not isinstance(fonte_vit, st.runtime.uploaded_file_manager.UploadedFile) and not os.path.exists(fonte_vit):
    st.error(f"Arquivo `{fonte_vit}` não encontrado. Faça o upload na barra lateral.")
    st.stop()

df_cnn_raw = carregar_planilha(fonte_cnn, "CNN")
df_vit_raw = carregar_planilha(fonte_vit, "Vision Transformer (ViT)")
df_completo = pd.concat([df_cnn_raw, df_vit_raw], ignore_index=True)

df_media = df_completo[df_completo["Fold"].astype(str).str.upper() == "MÉDIA"].copy()

st.sidebar.markdown("---")
st.sidebar.subheader("Métrica Principal")
metrica_label = st.sidebar.selectbox(
    "Selecione a métrica para comparação:",
    options=list(METRICAS_DISPONIVEIS.keys()),
    index=0,
)
metrica_col = METRICAS_DISPONIVEIS[metrica_label]
is_percentual = metrica_col in [
    "Acuracia_Teste",
    "F1_Score_Macro",
    "Precisao_Macro",
    "Recall_Macro",
]

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros de Experimentos")

# Filtro de Arquiteturas
arquiteturas_disponiveis = sorted(df_media["Arquitetura"].dropna().unique().tolist())
arquiteturas_sel = st.sidebar.multiselect(
    "Arquiteturas (CNN & ViT):",
    options=arquiteturas_disponiveis,
    default=arquiteturas_disponiveis,
)

janelas_presentes = [j for j in ORDEM_JANELAS if j in df_media["Janela"].astype(str).unique()]
janelas_sel = st.sidebar.multiselect(
    "Tamanhos de Janela (L):",
    options=janelas_presentes,
    default=janelas_presentes,
)

descritores_disponiveis = sorted(df_media["Descritor"].dropna().unique().tolist())
descritores_sel = st.sidebar.multiselect(
    "Descritor / Canal de Percolação:",
    options=descritores_disponiveis,
    default=descritores_disponiveis,
)

# Opção para próxima etapa: Matrizes de Confusão
st.sidebar.markdown("---")
st.sidebar.subheader("Próxima Etapa")
habilitar_cm = st.sidebar.checkbox(
    "Habilitar aba de Matrizes de Confusão",
    value=True,
    help="Ativa a visualização preparada para carregar os arquivos .npy de matriz de confusão.",
)

# Aplica filtros
df_filtrado = df_media[
    (df_media["Arquitetura"].isin(arquiteturas_sel))
    & (df_media["Janela"].astype(str).isin(janelas_sel))
    & (df_media["Descritor"].isin(descritores_sel))
].copy()

if df_filtrado.empty:
    st.warning("Nenhum experimento corresponde aos filtros selecionados na barra lateral.")
    st.stop()

# ==============================================================================
# CABEÇALHO E KPIs
# ==============================================================================
st.title("Comparativo de Experimentos CNNs vs Vision Transformers")
st.caption(
    "Análise das médias das 5 folds (`Fold == 'MÉDIA'`) "
)


melhor_idx = (
    df_filtrado[metrica_col].idxmin()
    if metrica_col in ["Loss_Teste", "Tempo_Treino_Seg"]
    else df_filtrado[metrica_col].idxmax()
)
melhor_row = df_filtrado.loc[melhor_idx]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Experimentos Filtrados", len(df_filtrado))
with col2:
    val_fmt = (
        f"{melhor_row[metrica_col]*100:.2f}%"
        if is_percentual
        else f"{melhor_row[metrica_col]:.4f}"
    )
    st.metric(f"Melhor {metrica_label}", val_fmt)
with col3:
    st.metric("Melhor Entrada / Janela", melhor_row["Entrada_Padronizada"])
with col4:
    media_geral = df_filtrado[metrica_col].mean()
    media_fmt = f"{media_geral*100:.2f}%" if is_percentual else f"{media_geral:.4f}"
    st.metric(f"Média Geral ({metrica_label})", media_fmt)

st.markdown("---")

# ==============================================================================
# ABAS DE VISUALIZAÇÃO
# ==============================================================================
nomes_abas = [
    "CNN vs. ViT por Janela (L)",
    "Comparação por Entrada Exata",
    "Mapa de Calor (Arquitetura × Entrada)",
    "Tabela Detalhada & Folds",
]
if habilitar_cm:
    nomes_abas.append("Matrizes de Confusão (Em Breve)")

abas = st.tabs(nomes_abas)

# ------------------------------------------------------------------------------
# ABA 1: DESCRITORES & DISTÂNCIAS POR JANELA (L3, L5, L7, L15, L21, L41)
# ------------------------------------------------------------------------------
with abas[0]:
    st.subheader(f"{metrica_label} por Descritor / Distância ao longo das Janelas (L)")

    def classificar_distancia(desc: str) -> str:
        if desc == "RGB Original":
            return "Baseline (Original)"
        if desc == "q_chess_eucl_manh":
            return "Combinada (Chess + Eucl + Manh)"
        if "chess" in desc:
            return "Chessboard(chess)"
        if "eucl" in desc:
            return "Euclidiana (eucl)"
        if "manh" in desc:
            return "Manhattan (manh)"
        return "Outra"

    df_aba1 = df_filtrado.copy()
    df_aba1["Distancia"] = df_aba1["Descritor"].apply(classificar_distancia)

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        distancias_disp = sorted(df_aba1["Distancia"].unique().tolist())
        distancias_sel = st.multiselect(
            "Filtrar por Métrica de Distância:",
            options=distancias_disp,
            default=distancias_disp,
            key="filtro_dist_aba1",
        )
    with col_f2:
        modo_visualizacao = st.radio(
            "Agrupar cores do gráfico por:",
            options=["Descritor (ex: cqm_eucl, q_eucl)", "Arquitetura (facetado por Descritor)"],
            index=0,
        )

    df_aba1 = df_aba1[df_aba1["Distancia"].isin(distancias_sel)]

    if df_aba1.empty:
        st.warning("Nenhum dado para a distância selecionada.")
    elif modo_visualizacao.startswith("Descritor"):
        # Compara cqm_eucl, q_eucl, etc. lado a lado em cada janela L, com painéis por Arquitetura
        fig_desc = px.bar(
            df_aba1.sort_values(["Janela", "Descritor"]),
            x="Janela",
            y=metrica_col,
            color="Descritor",
            barmode="group",
            facet_col="Arquitetura",
            facet_col_wrap=2,
            text_auto=".3f",
            category_orders={"Janela": janelas_sel},
            hover_data=["ID_Experimento", "Entrada_Padronizada", "Acuracia_Teste", "F1_Score_Macro"],
            title=f"Comparação entre Descritores por Janela (L) — {metrica_label}",
            labels={"Janela": "Janela (L)", metrica_col: metrica_label},
            height=650,
        )
        if is_percentual:
            y_min = max(0.0, df_aba1[metrica_col].min() - 0.04)
            fig_desc.update_yaxes(range=[y_min, 1.01], tickformat=".1%")
        fig_desc.update_traces(textposition="outside")
        st.plotly_chart(fig_desc, use_container_width=True)

        fig_linhas_desc = px.line(
            df_aba1.sort_values(["Janela", "Descritor"]),
            x="Janela",
            y=metrica_col,
            color="Descritor",
            line_dash="Familia",
            facet_col="Arquitetura",
            facet_col_wrap=2,
            markers=True,
            category_orders={"Janela": janelas_sel},
            title=f"Curva de Desempenho por Descritor ao Longo de L — {metrica_label}",
            labels={"Janela": "Janela (L)", metrica_col: metrica_label},
            height=620,
        )
        if is_percentual:
            fig_linhas_desc.update_yaxes(range=[y_min, 1.01], tickformat=".1%")
        fig_linhas_desc.update_traces(line=dict(width=2.5), marker=dict(size=8))
        st.plotly_chart(fig_linhas_desc, use_container_width=True)

    else:
        fig_arq = px.bar(
            df_aba1.sort_values(["Janela", "Arquitetura"]),
            x="Janela",
            y=metrica_col,
            color="Arquitetura",
            barmode="group",
            facet_col="Descritor",
            facet_col_wrap=2,
            text_auto=".3f",
            category_orders={"Janela": janelas_sel},
            title=f"CNN vs. ViT separado por Descritor — {metrica_label}",
            labels={"Janela": "Janela (L)", metrica_col: metrica_label},
            height=800,
        )
        if is_percentual:
            y_min = max(0.0, df_aba1[metrica_col].min() - 0.04)
            fig_arq.update_yaxes(range=[y_min, 1.01], tickformat=".1%")
        st.plotly_chart(fig_arq, use_container_width=True)

    # --------------------------------------------------------------------------
    # COMPARADOR DIRETO (DUELO ENTRE DOIS EXPERIMENTOS ESPECÍFICOS)
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Comparação Cenário A  x Cenário B ")

    opcoes_duelo = (
        df_aba1.sort_values(["Arquitetura", "Janela", "Descritor"])
        .apply(
            lambda r: f"{r['Arquitetura']} | {r['Entrada_Padronizada']} ({r['ID_Experimento']})",
            axis=1,
        )
        .tolist()
    )

    if len(opcoes_duelo) >= 2:
        idx_a = next((i for i, s in enumerate(opcoes_duelo) if "L3_cqm_eucl" in s), 0)
        idx_b = next((i for i, s in enumerate(opcoes_duelo) if "L15_q_eucl" in s), 1)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            cen_a = st.selectbox("Cenário A:", options=opcoes_duelo, index=idx_a, key="duelo_a")
        with col_d2:
            cen_b = st.selectbox("Cenário B:", options=opcoes_duelo, index=idx_b, key="duelo_b")

        row_a = df_aba1[
            df_aba1.apply(
                lambda r: f"{r['Arquitetura']} | {r['Entrada_Padronizada']} ({r['ID_Experimento']})" == cen_a,
                axis=1,
            )
        ].iloc[0]
        row_b = df_aba1[
            df_aba1.apply(
                lambda r: f"{r['Arquitetura']} | {r['Entrada_Padronizada']} ({r['ID_Experimento']})" == cen_b,
                axis=1,
            )
        ].iloc[0]

        val_a = row_a[metrica_col]
        val_b = row_b[metrica_col]
        diff = val_a - val_b

        m1, m2, m3 = st.columns(3)
        with m1:
            fmt_a = f"{val_a*100:.2f}%" if is_percentual else f"{val_a:.4f}"
            st.metric(f"Cenário A ({row_a['Entrada_Padronizada']})", fmt_a)
        with m2:
            fmt_b = f"{val_b*100:.2f}%" if is_percentual else f"{val_b:.4f}"
            st.metric(f"Cenário B ({row_b['Entrada_Padronizada']})", fmt_b)
        with m3:
            fmt_diff = f"{diff*100:+.2f} p.p." if is_percentual else f"{diff:+.4f}"
            st.metric("Diferença (A − B)", fmt_diff)

# ------------------------------------------------------------------------------
# ABA 2: COMPARAÇÃO DIRETA POR TIPO DE ENTRADA EXATA
# ------------------------------------------------------------------------------
with abas[1]:
    st.subheader(f"Comparação Direta por Representação de Entrada ({metrica_label})")

    apenas_comuns = st.checkbox(
        "Mostrar apenas entradas testadas simultaneamente em CNN e ViT (interseção)",
        value=True,
    )

    df_entrada = df_filtrado.copy()
    if apenas_comuns:
        entradas_cnn = set(df_entrada[df_entrada["Familia"] == "CNN"]["Entrada_Padronizada"])
        entradas_vit = set(
            df_entrada[df_entrada["Familia"] == "Vision Transformer (ViT)"]["Entrada_Padronizada"]
        )
        comuns = entradas_cnn.intersection(entradas_vit)
        df_entrada = df_entrada[df_entrada["Entrada_Padronizada"].isin(comuns)]

    if df_entrada.empty:
        st.info("Nenhuma entrada em comum encontrada com os filtros atuais.")
    else:
        fig_entrada = px.bar(
            df_entrada.sort_values(["Janela", "Entrada_Padronizada"]),
            x="Entrada_Padronizada",
            y=metrica_col,
            color="Arquitetura",
            barmode="group",
            hover_data=["ID_Experimento", "Descritor", "Acuracia_Teste", "F1_Score_Macro"],
            title=f"Desempenho por Tipo de Entrada ({metrica_label})",
            labels={"Entrada_Padronizada": "Tipo de Entrada", metrica_col: metrica_label},
            height=520,
        )
        if is_percentual:
            y_min = max(0.0, df_entrada[metrica_col].min() - 0.05)
            fig_entrada.update_yaxes(range=[y_min, 1.01], tickformat=".1%")
        fig_entrada.update_xaxes(tickangle=-35)
        st.plotly_chart(fig_entrada, use_container_width=True)

# ------------------------------------------------------------------------------
# ABA 3: HEATMAP ARQUITETURA x ENTRADA
# ------------------------------------------------------------------------------
with abas[2]:
    st.subheader(f"Matriz de Desempenho: Arquitetura × Entrada ({metrica_label})")

    pivot_df = df_filtrado.pivot_table(
        index="Entrada_Padronizada",
        columns="Arquitetura",
        values=metrica_col,
        aggfunc="mean",
    )

    fig_heat = px.imshow(
        pivot_df,
        text_auto=".4f",
        aspect="auto",
        color_continuous_scale="RdYlGn_r"
        if metrica_col in ["Loss_Teste", "Tempo_Treino_Seg"]
        else "Viridis",
        labels=dict(x="Arquitetura", y="Tipo de Entrada", color=metrica_label),
        height=max(450, len(pivot_df) * 28),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ------------------------------------------------------------------------------
# ABA 4: TABELA RESUMO E INSPEÇÃO DOS 5 FOLDS
# ------------------------------------------------------------------------------
with abas[3]:
    st.subheader("Tabela Consolidada (Médias das Folds)")

    cols_exibir = [
        "ID_Experimento",
        "Familia",
        "Arquitetura",
        "Janela",
        "Entrada_Padronizada",
        "Acuracia_Teste",
        "F1_Score_Macro",
        "Precisao_Macro",
        "Recall_Macro",
        "Loss_Teste",
        "Tempo_Treino_Seg",
    ]
    st.dataframe(
        df_filtrado[cols_exibir].sort_values(
            by=metrica_col,
            ascending=(metrica_col in ["Loss_Teste", "Tempo_Treino_Seg"]),
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### 🔎 Detalhamento por Fold (1 a 5)")
    exp_opcoes = (
        df_filtrado["ID_Experimento"]
        + " | "
        + df_filtrado["Arquitetura"]
        + " | "
        + df_filtrado["Entrada_Padronizada"]
    ).tolist()
    exp_escolhido = st.selectbox("Inspecionar folds individuais de:", exp_opcoes)
    id_sel = exp_escolhido.split(" | ")[0]
    arq_sel = exp_escolhido.split(" | ")[1]

    df_folds = df_completo[
        (df_completo["ID_Experimento"] == id_sel)
        & (df_completo["Arquitetura"] == arq_sel)
        & (df_completo["Fold"].astype(str).str.upper() != "MÉDIA")
    ]
    st.dataframe(
        df_folds[
            [
                "Fold",
                "Epoca_Early_Stopping",
                "Acuracia_Teste",
                "F1_Score_Macro",
                "Precisao_Macro",
                "Recall_Macro",
                "Loss_Teste",
                "Dir_Matriz_Confusao",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

# ------------------------------------------------------------------------------
# ABA 5: PREPARADA PARA MATRIZES DE CONFUSÃO (.npy)
# ------------------------------------------------------------------------------
if habilitar_cm:
    with abas[4]:
        st.subheader("Visualização de Matrizes de Confusão (Próxima Etapa)")
        st.info(
            "Estrutura pronta para carregar os arquivos `.npy` apontados na coluna `Dir_Matriz_Confusao` "
            "(ex: `./resultados/EXP_001_ViT_B_16/cm_fold_1.npy`) ou somar as matrizes das 5 folds."
        )

        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            exp_cm_sel = st.selectbox(
                "Selecione o Experimento:",
                options=exp_opcoes,
                key="cm_exp_select",
            )
        with col_cm2:
            fold_cm_sel = st.selectbox(
                "Selecione o Fold:",
                options=["Soma das 5 Folds", "1", "2", "3", "4", "5"],
                key="cm_fold_select",
            )

        id_cm = exp_cm_sel.split(" | ")[0]
        arq_cm = exp_cm_sel.split(" | ")[1]
        df_cm_rows = df_completo[
            (df_completo["ID_Experimento"] == id_cm)
            & (df_completo["Arquitetura"] == arq_cm)
            & (df_completo["Fold"].astype(str).str.upper() != "MÉDIA")
        ]

        caminhos_npy = df_cm_rows["Dir_Matriz_Confusao"].dropna().tolist()
        st.write("**Caminhos registrados na planilha para este experimento:**")
        st.code("\n".join(caminhos_npy) if caminhos_npy else "Nenhum caminho registrado.")

        # Carregamento automático caso a pasta ./resultados exista localmente
        matriz_carregada = None
        if fold_cm_sel == "Soma das 5 Folds":
            matrizes = [
                np.load(p.replace("\\", "/"))
                for p in caminhos_npy
                if os.path.exists(p.replace("\\", "/"))
            ]
            if matrizes:
                matriz_carregada = np.sum(matrizes, axis=0)
        else:
            row_f = df_cm_rows[df_cm_rows["Fold"].astype(str) == fold_cm_sel]
            if not row_f.empty:
                caminho = str(row_f.iloc[0]["Dir_Matriz_Confusao"]).replace("\\", "/")
                if os.path.exists(caminho):
                    matriz_carregada = np.load(caminho)

        if matriz_carregada is not None:
            classes_lc25000 = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]
            labels = (
                classes_lc25000
                if matriz_carregada.shape[0] == 5
                else [f"Classe {i}" for i in range(matriz_carregada.shape[0])]
            )
            fig_cm = px.imshow(
                matriz_carregada,
                x=labels,
                y=labels,
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(x="Predito", y="Real", color="Amostras"),
                title=f"Matriz de Confusão — {exp_cm_sel} ({fold_cm_sel})",
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.warning(
                "Usar os .npy  ./resultados/.... "

            )