# Análise de Sentimentos em Avaliações de E-commerce (Olist)

![Status do Projeto](https://img.shields.io/badge/status-concluído-green)
![Python Version](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Um projeto de Machine Learning ponta a ponta que realiza a análise exploratória, o treino de um modelo de classificação e a implantação de um dashboard interativo para analisar o sentimento de avaliações de clientes.


## Features

* **Análise Exploratória de Dados (EDA):** Investigação detalhada sobre a distribuição das notas e a correlação entre o texto e a avaliação.
* **Engenharia de Modelos:** Comparação entre um modelo baseline e um modelo otimizado com a técnica **SMOTE** para corrigir o desbalanceamento de classes, resultando num aumento significativo do *Recall* para feedbacks negativos.
* **Dashboard Interativo:** Uma interface intuitiva construída com **Streamlit** que consome o modelo treinado e permite a classificação de novos textos em tempo real.


## Tecnologias Utilizadas

* **Linguagem:** Python
* **Análise de Dados:** Pandas, Matplotlib
* **Machine Learning:** Scikit-learn, Imbalanced-learn
* **Dashboard:** Streamlit
* **Containerização:** Docker

## Como Executar

### 1. Execução Local

```bash
# 1. Clone o repositório
git clone [https://github.com/seu-usuario/analise-sentimentos-reviews-olist.git](https://github.com/seu-usuario/analise-sentimentos-reviews-olist.git)
cd analise-sentimentos-reviews-olist

# 2. Crie e ative um ambiente virtual
python -m venv .venv
source .venv/bin/activate  # (ou .\.venv\Scripts\activate no Windows)

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute a aplicação Streamlit
streamlit run app/app.py
```

### 2. Execução via Docker
Certifique-se de que tem o Docker instalado e em execução.

```bash
# 1. Construa a imagem Docker a partir da raiz do projeto
docker build -t analise-sentimentos-app .

# 2. Execute o contêiner a partir da imagem criada
docker run -p 8501:8501 analise-sentimentos-app
```
