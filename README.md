# CardioPredict AI 🏥

CardioPredict AI é um projeto completo de Machine Learning focado em prever o **Risco de Doença Cardíaca** com base em fatores clínicos, histórico familiar e estilo de vida. O projeto contempla análise exploratória de dados, tratamento de outliers, engenharia de atributos (feature engineering), experimentação e tuning usando **MLflow**, e um dashboard interativo feito com **Streamlit**.

## 🚀 Estrutura do Projeto

- `ML.py`: O script principal do aplicativo Streamlit e da pipeline do MLflow ativo.
- `desafio-final.ipynb`: Notebook com a Análise Exploratória (EDA), feature engineering e comparação de múltiplos algoritmos.
- `dataset_saude_brasil.csv`: Dataset contendo informações de saúde de múltiplos pacientes sintéticos do Brasil.
- `modelo_risco_saude.pkl`: Modelo RandomForest treinado e salvo usando `joblib`.
- `requirements.txt`: Lista de dependências e bibliotecas do projeto.

## 🛠️ Requisitos e Instalação

Para rodar o projeto localmente, primeiro clone este repositório no seu computador e crie um ambiente virtual (recomendado):

```bash
# Clone o repositório
git clone https://github.com/SeuUsuario/CardioPredict-AI.git
cd CardioPredict-AI

# Crie e ative um ambiente virtual (Windows)
python -m venv venv
venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```

## 🖥️ Como Executar

Para iniciar a interface web interativa do Streamlit, basta executar:

```bash
streamlit run ML.py
```

Isso abrirá automaticamente no seu navegador o endereço `http://localhost:8501`. Na interface, você pode inserir métricas do perfil de um paciente para verificar a inferência do modelo em tempo real, junto com a confiança e métricas cadastradas no MLflow.

## 🔬 Modelagem e MLflow

O algoritmo vencedor após a validação no formato *holdout 80/20 estratificado* foi o **RandomForestClassifier**. O registro dos modelos e parâmetros foi feito usando `mlflow` para monitoramento. No arquivo `desafio-final.ipynb` foram testadas várias outras abordagens, dentre elas o KNN, Regressão Logística, ExtraTrees e GradientBoosting.

## 🌐 Deploy (Nuvem Grátis)

### No Hugging Face Spaces:
1. Crie uma conta no [Hugging Face](https://huggingface.co/).
2. Clique em **Spaces > Create new Space**.
3. Dê um nome, escolha o License, selecione **Streamlit** no Space SDK.
4. Faça o upload dos arquivos `ML.py`, `dataset_saude_brasil.csv`, `requirements.txt` e `modelo_risco_saude.pkl`.
5. Pronto! O Hugging Face iniciará o app automaticamente e fornecerá um link público.

### No Streamlit Community Cloud:
1. Suba todo este código para um repositório público no seu **GitHub**.
2. Acesse [share.streamlit.io](https://share.streamlit.io/) e faça login.
3. Clique em **New app** e selecione o repositório, a *branch* e o arquivo principal `ML.py`.
4. Clique em **Deploy!**

---
*Projeto desenvolvido para aplicação real de MLOps, Deploy e análise preditiva na área da Saúde.*
