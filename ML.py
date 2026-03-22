import pandas as pd
import joblib
import streamlit as st
import os

# --- CONFIGURAÇÃO DA PÁGINA STREAMLIT ---
st.set_page_config(page_title="Risco Saúde", layout="wide", page_icon="🏥")

# --- 1. LÓGICA DE CARREGAMENTO DO MODELO ---
if not os.path.exists('modelo_risco_saude.pkl'):
    st.warning("⚠️ Nenhum modelo de Inteligência Artificial encontrado no sistema!")
    st.info("👉 Vá até a página de **Benchmark** no menu lateral, escolha os testes e clique em Treinar. O melhor modelo gerado será ativado aqui automaticamente.")
    st.stop()
else:
    modelo = joblib.load('modelo_risco_saude.pkl')

# --- 3. INTERFACE DO DASHBOARD ---
st.title("🏥 Risco Saúde + MLflow Tracking")
st.markdown("Analise de risco cardíaco com monitoramento de experimentos em tempo real.")
st.warning("**Aviso Legal:** Este conteúdo é destinado apenas para fins educacionais. Os dados exibidos são ilustrativos e podem não corresponder a situações reais.")

with st.form("diagnostico_form"):
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("📋 Perfil")
        idade = st.number_input("Idade", 18, 100, 35)
        sexo = st.selectbox("Sexo", ["Masculino", "Feminino"])
        hist_fam = st.selectbox("Histórico Familiar?", ["Não", "Sim"])
        trabalho = st.slider("Horas de Trabalho/Dia", 0, 16, 8)

    with c2:
        st.subheader("🍏 Físico e Hábitos")
        peso = st.number_input("Peso (kg)", 30.0, 250.0, 75.0)
        altura = st.number_input("Altura (m)", 1.00, 2.50, 1.70, step=0.01)
        imc = peso / (altura ** 2)
        st.caption(f"📐 **IMC Automático:** {imc:.1f}")
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

# Exibe o modelo ativo
nome_modelo_ativo = type(modelo).__name__
st.sidebar.markdown("### 🤖 IA Ativa no Sistema")
st.sidebar.success(f"**{nome_modelo_ativo}**")

# Rodapé lateral
st.sidebar.markdown("### 🛠️ Configurações Gerais")
if st.sidebar.button("Excluir Modelo Ativo"):
    if os.path.exists('modelo_risco_saude.pkl'):
        os.remove('modelo_risco_saude.pkl')
        st.rerun()