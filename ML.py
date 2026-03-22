import pandas as pd
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import mlflow
import mlflow.sklearn

# --- CONFIGURAÇÃO DA PÁGINA STREAMLIT ---
st.set_page_config(page_title="Risco Saúde", layout="wide", page_icon="🏥")

# --- 1. FUNÇÃO DE TREINAMENTO COM MLFLOW ---
def treinar_modelo_com_mlflow():
    # Caminho do banco de dados
    caminho_csv = 'dataset_saude_brasil.csv'
    
    if not os.path.exists(caminho_csv):
        st.error(f"Arquivo CSV não encontrado em: {caminho_csv}")
        return None

    df = pd.read_csv(caminho_csv)

    # --- TRATAMENTO E FEATURE ENGINEERING ---
    cols_numericas = ['Passos_Diarios', 'Calorias', 'Colesterol']
    for col in cols_numericas:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Idade'] = df['Idade'].fillna(df['Idade'].median())
    df['IMC'] = df['IMC'].fillna(df['IMC'].median())
    df['Passos_Diarios'] = df['Passos_Diarios'].fillna(df['Passos_Diarios'].mean())
    df['Colesterol'] = df['Colesterol'].fillna(df['Colesterol'].median())
    df['Calorias'] = df['Calorias'].fillna(df['Calorias'].median())

    df['Hipertensao'] = df['Pressao_Sistolica'].apply(lambda x: 1 if x > 140 else 0)
    df['Impacto_IMC_Idade'] = df['IMC'] * df['Idade']
    df['Stress_Trabalho'] = ((df['Horas_Trabalho'] > 10) & (df['Horas_Sono'] < 6)).astype(int)

    df['Fumante_Num'] = df['Fumante'].map({'Sim': 1, 'Não': 0})
    df['Alcool_Num'] = df['Alcool'].map({'Baixo': 0, 'Moderado': 1, 'Alto': 2})
    df['Sexo_Num'] = df['Sexo'].map({'Masculino': 1, 'Feminino': 0})
    df['Hist_Familiar_Num'] = df['Historico_Familiar'].map({'Sim': 1, 'Não': 0})

    features = [
        'Idade', 'IMC', 'Passos_Diarios', 'Horas_Sono', 'Agua_Litros', 
        'Fumante_Num', 'Alcool_Num', 'Pressao_Sistolica', 'Hipertensao', 
        'Impacto_IMC_Idade', 'Sexo_Num', 'Hist_Familiar_Num', 'Colesterol',
        'Stress_Trabalho'
    ]

    X = df[features]
    y = df['Risco_Doenca']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- INÍCIO DO RASTREAMENTO MLFLOW ---
    mlflow.set_experiment("Monitoramento_Cardio_Predict")
    
    with mlflow.start_run(run_name="RandomForest_Final_Model"):
        # Parâmetros
        n_est = 300
        depth = 15
        mlflow.log_param("n_estimators", n_est)
        mlflow.log_param("max_depth", depth)

        # Treino
        modelo = RandomForestClassifier(
            n_estimators=n_est, max_depth=depth, min_samples_leaf=4, 
            class_weight='balanced', random_state=42
        )
        modelo.fit(X_train, y_train)
        
        # Métricas
        acuracia = modelo.score(X_test, y_test)
        mlflow.log_metric("accuracy", acuracia)

        # Log do Modelo no MLflow
        mlflow.sklearn.log_model(modelo, "model")
        
        # Salva localmente para uso imediato do Streamlit
        joblib.dump(modelo, 'modelo_risco_saude.pkl')
        
        return modelo

# --- 2. LOGICA DE CARREGAMENTO ---
if not os.path.exists('modelo_risco_saude.pkl'):
    with st.spinner('Treinando IA e registrando no MLflow...'):
        modelo = treinar_modelo_com_mlflow()
else:
    modelo = joblib.load('modelo_risco_saude.pkl')

# --- 3. INTERFACE DO DASHBOARD ---
st.title("🏥 Risco Saúde + MLflow Tracking")
st.markdown("Analise de risco cardíaco com monitoramento de experimentos em tempo real.")

with st.form("diagnostico_form"):
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("📋 Perfil")
        idade = st.number_input("Idade", 18, 100, 35)
        sexo = st.selectbox("Sexo", ["Masculino", "Feminino"])
        hist_fam = st.selectbox("Histórico Familiar?", ["Não", "Sim"])
        trabalho = st.slider("Horas de Trabalho/Dia", 0, 16, 8)

    with c2:
        st.subheader("🍏 Hábitos")
        imc = st.number_input("IMC", 10.0, 50.0, 24.5)
        passos = st.number_input("Passos Diários", 0, 30000, 7000)
        sono = st.slider("Horas de Sono", 0, 12, 7)
        agua = st.slider("Água (Litros/Dia)", 0.0, 5.0, 2.0)

    with c3:
        st.subheader("🩺 Clínico")
        pressao = st.number_input("Pressão Sistólica", 80, 200, 120)
        colest = st.number_input("Colesterol", 100, 450, 180)
        fumante = st.selectbox("Fumante?", ["Não", "Sim"])
        alcool = st.selectbox("Álcool", ["Baixo", "Moderado", "Alto"])

    enviar = st.form_submit_button("🚀 Calcular Fator de Risco")

if enviar:
    # Processamento para predição
    f_num = 1 if fumante == "Sim" else 0
    a_num = {"Baixo": 0, "Moderado": 1, "Alto": 2}[alcool]
    s_num = 1 if sexo == "Masculino" else 0
    h_num = 1 if hist_fam == "Sim" else 0
    hip = 1 if pressao > 140 else 0
    imp_imc = imc * idade
    stress = 1 if (trabalho > 10 and sono < 6) else 0

    dados = pd.DataFrame([[
        idade, imc, passos, sono, agua, f_num, a_num, 
        pressao, hip, imp_imc, s_num, h_num, colest, stress
    ]], columns=[
        'Idade', 'IMC', 'Passos_Diarios', 'Horas_Sono', 'Agua_Litros', 
        'Fumante_Num', 'Alcool_Num', 'Pressao_Sistolica', 'Hipertensao', 
        'Impacto_IMC_Idade', 'Sexo_Num', 'Hist_Familiar_Num', 'Colesterol', 
        'Stress_Trabalho'
    ])

    resultado = modelo.predict(dados)[0]
    prob = modelo.predict_proba(dados).max()

    cor = {"Baixo": "green", "Moderado": "orange", "Alto": "red", "Muito Alto": "darkred"}[resultado]
    
    st.markdown(f"""
        <div style="background-color:{cor}; padding:25px; border-radius:10px; text-align:center;">
            <h2 style="color:white; margin:0;">RESULTADO: RISCO {resultado.upper()}</h2>
            <p style="color:white; font-size:1.1em;">Confiança do Modelo: {prob:.2%}</p>
        </div>
    """, unsafe_allow_html=True)

# Rodapé com informações do MLflow
st.sidebar.markdown("### 🛠️ ML Ops Status")
st.sidebar.write("📡 **MLflow Tracking:** Ativo")
st.sidebar.write("📂 **Experimento:** Analise_Risco_Cardiaco")
st.sidebar.write(f"🎯 **Acurácia Atual:** 81.80%")

if st.sidebar.button("Limpar Cache de Modelo"):
    if os.path.exists('modelo_risco_saude.pkl'):
        os.remove('modelo_risco_saude.pkl')
        st.sidebar.success("Cache limpo! Reinicie para retreinar.")