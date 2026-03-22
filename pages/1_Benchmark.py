import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

st.set_page_config(page_title="Benchmark", page_icon="📊", layout="wide")
st.title("📊 Benchmark de Modelos")
st.markdown("Bem-vindo à central de testes. Altere a proporção de treino e teste e avalie a performance de diferentes algoritmos de Machine Learning, com tudo rastreado pelo **MLflow**!")

# Reutilizando a mesma lógica do dataset
caminho_csv = 'dataset_saude_brasil.csv'

@st.cache_data
def carregar_e_preparar_dados():
    if not os.path.exists(caminho_csv):
        st.error(f"Arquivo CSV não encontrado em: {caminho_csv}")
        return None, None

    df = pd.read_csv(caminho_csv)

    # Tratamento de Dados
    cols_numericas = ['Passos_Diarios', 'Calorias', 'Colesterol']
    for col in cols_numericas:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Idade'] = df['Idade'].fillna(df['Idade'].median())
    df['IMC'] = df['IMC'].fillna(df['IMC'].median())
    df['Passos_Diarios'] = df['Passos_Diarios'].fillna(df['Passos_Diarios'].mean())
    df['Colesterol'] = df['Colesterol'].fillna(df['Colesterol'].median())
    df['Calorias'] = df['Calorias'].fillna(df['Calorias'].median())

    # Feature Engineering
    df['Hipertensao'] = df['Pressao_Sistolica'].apply(lambda x: 1 if x > 140 else 0)
    df['Impacto_IMC_Idade'] = df['IMC'] * df['Idade']
    df['Stress_Trabalho'] = ((df['Horas_Trabalho'] > 10) & (df['Horas_Sono'] < 6)).astype(int)

    # Tratamento de Categorias
    df['Fumante_Num'] = df['Fumante'].map({'Sim': 1, 'Não': 0})
    df['Alcool_Num'] = df['Alcool'].map({'Baixo': 0, 'Moderado': 1, 'Alto': 2})
    df['Sexo_Num'] = df['Sexo'].map({'Masculino': 1, 'Feminino': 0})
    df['Hist_Familiar_Num'] = df['Historico_Familiar'].map({'Sim': 1, 'Não': 0})

    # Seleção de Features
    features = [
        'Idade', 'IMC', 'Passos_Diarios', 'Horas_Sono', 'Agua_Litros', 
        'Fumante_Num', 'Alcool_Num', 'Pressao_Sistolica', 'Hipertensao', 
        'Impacto_IMC_Idade', 'Sexo_Num', 'Hist_Familiar_Num', 'Colesterol',
        'Stress_Trabalho'
    ]

    X = df[features]
    y = df['Risco_Doenca']

    return X, y

st.sidebar.header("🛠️ Configurações de Treino")
test_size_percent = st.sidebar.slider("Porcentagem de Teste (%)", min_value=10, max_value=50, value=20, step=5)
test_size = test_size_percent / 100.0

st.sidebar.markdown(f"**Divisão atual:**\n- Treino: {100 - test_size_percent}%\n- Teste: {test_size_percent}%")

X, y = carregar_e_preparar_dados()

if X is not None:
    st.markdown("### Selecione os Modelos para Testar")
    
    modelos_disponiveis = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    
    modelos_selecionados = st.multiselect(
        "Modelos:", 
        list(modelos_disponiveis.keys()), 
        default=["RandomForest", "GradientBoosting"]
    )
    
    if st.button("🚀 Iniciar Benchmark com MLflow", use_container_width=True):
        if len(modelos_selecionados) == 0:
            st.warning("Por favor, selecione pelo menos um modelo!")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("Benchmark_Cardio_Predict")
            
            resultados = []
            
            with st.spinner("Treinando modelos e registrando no MLflow Cloud/Local..."):
                for nome_modelo in modelos_selecionados:
                    with mlflow.start_run(run_name=f"Bench_{nome_modelo}_{test_size_percent}%"):
                        modelo = modelos_disponiveis[nome_modelo]
                        
                        # Registrando Parametros Globais
                        mlflow.log_param("test_size", test_size)
                        mlflow.log_param("modelo_tipo", nome_modelo)
                        
                        # Treinando
                        modelo.fit(X_train, y_train)
                        
                        # Calculando
                        acc = accuracy_score(y_test, modelo.predict(X_test))
                        
                        # Logando MLflow
                        mlflow.log_metric("accuracy", acc)
                        mlflow.sklearn.log_model(modelo, f"model_{nome_modelo}")
                        
                        resultados.append({
                            "Modelo": nome_modelo, 
                            "Acurácia": acc
                        })
            
            st.success("✅ Experimentos concluídos com sucesso e gravados no MLflow!")
            
            # Exibe os resultados tabelados
            df_res = pd.DataFrame(resultados).sort_values(by="Acurácia", ascending=False)
            st.dataframe(df_res.style.format({"Acurácia": "{:.2%}"}), use_container_width=True)
            
            melhor = df_res.iloc[0]['Modelo']
            st.info(f"🏆 O melhor modelo neste split foi o **{melhor}**.")
