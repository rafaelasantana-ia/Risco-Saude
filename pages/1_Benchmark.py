import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import os
import numpy as np
from scipy.stats import mstats

def clip_iqr(s: pd.Series, factor: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - factor * iqr, q3 + factor * iqr
    return s.clip(lo, hi)

def winsorize_series(s: pd.Series, limits=(0.01, 0.01)) -> pd.Series:
    arr = s.astype(float).values
    mask = np.isfinite(arr)
    if mask.sum() == 0:
        return s
    out = arr.copy()
    out[mask] = mstats.winsorize(arr[mask], limits=limits)
    return pd.Series(out, index=s.index)

def cap_zscore(s: pd.Series, z: float = 3.0) -> pd.Series:
    mu, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return s
    lo, hi = mu - z * sd, mu + z * sd
    return s.clip(lo, hi)

st.set_page_config(page_title="Benchmark", page_icon="📊", layout="wide")
st.title("📊 Benchmark Avançado de Modelos")
st.markdown("Altere a proporção de treino e teste, e avalie vários modelos. Tudo com **Métricas Avançadas** e **Matriz de Confusão** salva direto no MLflow!")

# Reutilizando a mesma lógica do dataset
caminho_csv = 'dataset_saude_brasil.csv'

@st.cache_data
def carregar_e_preparar_dados():
    if not os.path.exists(caminho_csv):
        st.error(f"Arquivo CSV não encontrado em: {caminho_csv}")
        return None, None

    df = pd.read_csv(caminho_csv)

    cols_numericas = ['Passos_Diarios', 'Calorias', 'Colesterol']
    for col in cols_numericas:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Idade'] = df['Idade'].fillna(df['Idade'].median())
    df['IMC'] = df['IMC'].fillna(df['IMC'].median())
    df['Passos_Diarios'] = df['Passos_Diarios'].fillna(df['Passos_Diarios'].mean())
    df['Colesterol'] = df['Colesterol'].fillna(df['Colesterol'].median())
    df['Calorias'] = df['Calorias'].fillna(df['Calorias'].median())

    # --- TRATAMENTO DE OUTLIERS ---
    for col in ["Passos_Diarios", "Calorias", "Colesterol", "IMC"]:
        if col in df.columns:
            df[col] = winsorize_series(df[col], limits=(0.01, 0.01))
            df[col] = clip_iqr(df[col])

    for col in ["Pressao_Sistolica"]:
        if col in df.columns:
            df[col] = cap_zscore(df[col], z=3.0)

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
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True, class_weight='balanced', random_state=42),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced')
    }
    
    modelos_selecionados = st.multiselect(
        "Modelos:", 
        list(modelos_disponiveis.keys()), 
        default=["RandomForest", "GradientBoosting", "DecisionTree"]
    )
    
    if st.button("🚀 Iniciar Benchmark com MLflow", use_container_width=True):
        if len(modelos_selecionados) == 0:
            st.warning("Por favor, selecione pelo menos um modelo!")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("Benchmark_Cardio_Predict")
            
            resultados = []
            
            with st.spinner("Treinando modelos e registrando no MLflow (Métricas e Gráficos)..."):
                for nome_modelo in modelos_selecionados:
                    with mlflow.start_run(run_name=f"Bench_{nome_modelo}_{test_size_percent}%"):
                        modelo = modelos_disponiveis[nome_modelo]
                        
                        # Parametros globais
                        mlflow.log_param("test_size", test_size)
                        mlflow.log_param("modelo_tipo", nome_modelo)
                        
                        # Treinando
                        modelo.fit(X_train, y_train)
                        
                        # Predições
                        y_pred = modelo.predict(X_test)
                        
                        # Calculando as métricas avançadas (weighted por conta das classes de Risco)
                        acc = accuracy_score(y_test, y_pred)
                        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        
                        # Logando Métricas no MLflow
                        mlflow.log_metrics({
                            "accuracy": acc,
                            "precision": prec,
                            "recall": rec,
                            "f1_score": f1
                        })
                        
                        # Gerando e Salvando a Matriz de Confusão no MLflow
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                    xticklabels=modelo.classes_, yticklabels=modelo.classes_)
                        plt.title(f"Matriz de Confusão - {nome_modelo}")
                        plt.ylabel('Verdadeiro (Real)')
                        plt.xlabel('Predito (Modelo)')
                        
                        cm_path = f"tmp_cm_{nome_modelo}.png"
                        fig.savefig(cm_path, bbox_inches='tight')
                        mlflow.log_artifact(cm_path, artifact_path="graficos")
                        plt.close(fig)
                        
                        # Apaga o arquivo temporário local
                        if os.path.exists(cm_path):
                            os.remove(cm_path)
                            
                        # Logando o Modelo
                        mlflow.sklearn.log_model(modelo, f"model_{nome_modelo}")
                        
                        resultados.append({
                            "Modelo": nome_modelo, 
                            "Acurácia": acc,
                            "F1-Score": f1,
                            "Precision": prec,
                            "Recall": rec
                        })
            
            st.success("✅ Experimentos concluídos! Acesse o painel do MLflow para ver as matrizes.")
            
            # Exibe os resultados na tela ranqueados por F1-Score (A métrica mais confiável para desbalanceados)
            df_res = pd.DataFrame(resultados).sort_values(by="F1-Score", ascending=False)
            
            # Aplica CSS do Pandas Styler para visual
            tabela_formatada = df_res.style.format({
                "Acurácia": "{:.2%}",
                "F1-Score": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}"
            }).background_gradient(subset=["F1-Score"], cmap="Greens")
            
            st.dataframe(tabela_formatada, use_container_width=True)
            
            melhor = df_res.iloc[0]['Modelo']
            st.info(f"🏆 O melhor modelo neste split (baseado em F1-Score) foi o **{melhor}**.")
