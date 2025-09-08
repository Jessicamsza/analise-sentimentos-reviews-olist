import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(
    page_title="Análise de Sentimento",
    page_icon="🧠",
    layout="wide"
)

# TEMA PERSONALIZADO
st.markdown(
    """
    <style>
    .stApp {
        background-color: #08383e;
        color: #d5d4c8;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #d5d4c8;
    }
    .stButton>button {
        color: #08383e;
        background-color: #9eeca3;
        border-color: #9eeca3;
    }
    </style>
    """, unsafe_allow_html=True
)

# FUNÇÕES DE CACHE
@st.cache_resource
def load_pipeline():
    """Carrega e retorna o pipeline de classificação treinado."""
    model_path = Path(__file__).parent.parent / "models/modelo_sentimento.pkl"
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except FileNotFoundError:
        st.error(f"Arquivo do modelo não encontrado em: {model_path}")
        return None

@st.cache_data
def load_data():
    """Carrega e retorna o dataset de avaliações."""
    df = pd.read_csv("https://github.com/Jessicamsza/analise-sentimentos-reviews-olist/raw/refs/heads/main/data/olist_order_reviews_dataset.csv")
    return df

# CARREGAMENTO DOS ARQUIVOS
pipeline = load_pipeline()
df = load_data()

# LAYOUT DA APLICAÇÃO
st.title("Dashboard de Análise de Sentimento")
st.markdown("Uma ferramenta para analisar e classificar o feedback de clientes em tempo real.")


# SEÇÃO 1: VISÃO GERAL DO DATASET

st.header("📊 Análise Geral do Dataset de Avaliações")

if df is not None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Avaliações", len(df))
    col2.metric("Nota Média", round(df['review_score'].mean(), 2))
    col3.metric("% Avaliações Positivas", f"{(df['review_score'] >= 4).mean()*100:.1f}%")

    # Cria duas colunas para tabelas e gráficos
    col4, col5 = st.columns([1, 2])

    with col4:
        st.subheader("Distribuição das Notas")
        score_counts = df['review_score'].value_counts().sort_index()
        st.dataframe(score_counts)

    with col5:
        st.subheader("Gráfico da Distribuição das Notas")
        fig = px.bar(
            score_counts,
            x=score_counts.index,
            y=score_counts.values,
            title="Contagem de Avaliações por Nota (1 a 5)",
            labels={'x': 'Nota da Avaliação', 'y': 'Quantidade de Avaliações'},
            text_auto=True,
            color_discrete_sequence=['#9eeca3']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#d5d4c8',
            xaxis_title_font_color='#d5d4c8',
            yaxis_title_font_color='#d5d4c8'
        )
        st.plotly_chart(fig, use_container_width=True)


# SEÇÃO 2: CLASSIFICADOR INTERATIVO

st.header("🤖 Teste o Classificador de Sentimento")

texto_usuario = st.text_area(
    "Digite um comentário de cliente para ser classificado:",
    placeholder="Ex: 'Adorei o produto, a entrega foi muito rápida!' ou 'Horrível, atrasou e veio quebrado.'",
    height=150
)

colA, colB, colC = st.columns([1,1,1])
with colB:  # centraliza o botão
    if st.button("Classificar Sentimento", type="primary"):
        if pipeline is not None and texto_usuario.strip():
            predicao = pipeline.predict([texto_usuario])[0]
            try:
                probabilidades = pipeline.predict_proba([texto_usuario])[0]
                confianca = max(probabilidades) * 100
            except AttributeError:
                confianca = None

            st.write("### Resultado da Classificação")
            if predicao == 1:
                st.success("Sentimento Positivo 👍")
            else:
                st.error("Sentimento Negativo 👎")

            if confianca:
                st.progress(int(confianca))
                st.caption(f"Confiança do modelo: {confianca:.1f}%")
        elif not texto_usuario.strip():
            st.warning("Por favor, digite um comentário para classificar.")

st.info("""
**Qual o objetivo deste classificador?**

Enquanto a nota (1-5) já indica o sentimento em um review, o verdadeiro poder deste modelo é sua capacidade de analisar textos **sem uma nota associada**, como emails, comentários em redes sociais ou chats de suporte.

Ele atua como um **filtro inteligente**, identificando e priorizando automaticamente os feedbacks negativos de qualquer fonte para uma análise mais aprofundada, como a extração de causa-raiz com IA Generativa.
""")

# SEÇÃO 3: PRÓXIMOS PASSOS

st.header("🚀 Próximos Passos")
st.markdown("""
- 🔗 **Integração em tempo real** com dados de redes sociais e sistemas de atendimento.  
- 🧾 **Explicabilidade com IA Generativa**: gerar resumos automáticos das principais reclamações.  
- 📈 **Dashboard executivo** para acompanhamento de KPIs de satisfação.  
- 🛠 **Automação de processos**: priorizar tickets de suporte automaticamente.
""")
