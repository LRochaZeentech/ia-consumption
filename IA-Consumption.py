import streamlit as st
import pandas as pd
from modelo_consumption import carregar_modelo, prever  # Importa as funções do arquivo modelo.pkl
import joblib
import json
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import io
import sqlite3
import os

# Obtém o diretório atual do script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Caminho relativo para a imagem
image_path = os.path.join(base_dir, "static", "logo-zeentech.png")

st.set_page_config(
        page_title="Consumption predict - IA",
        page_icon=image_path,
        layout="wide",
        initial_sidebar_state="expanded"  # Sidebar expandido e fixo
    )

# Definir um scaler global antes de qualquer processamento
scaler = None  

# Função para carregar o scaler antes de qualquer interação
@st.cache_resource
def carregar_scaler():
    return joblib.load("scaler.pkl")  # Substitua pelo caminho correto

scaler = carregar_scaler()

# Caminho relativo para a imagem
image_path_vinho = os.path.join(base_dir, "static", "ZeentechIDT_-_Vinho.png")

# Título da aplicação
st.image(image_path_vinho, width=170)
st.title("Consumption Predict")

def verificar_usuario(username, password):
    conn = sqlite3.connect("usuarios.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM usuarios WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()
    conn.close()
    return user

# Estado de login
if "logado" not in st.session_state:
    st.session_state.logado = False

if "usuario" not in st.session_state:
    st.session_state.usuario = ""

if not st.session_state.logado:
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Usuário")
    password = st.sidebar.text_input("Senha", type="password")

    if st.sidebar.button("Entrar"):
        if verificar_usuario(username, password):
            st.session_state.logado = True
            st.session_state.usuario = username  # Salva o nome do usuário na sessão
            st.rerun()
        else:
            st.sidebar.error("Usuário ou senha inválidos")
else:
    st.sidebar.success(f"Bem-vindo, {st.session_state.usuario}!")
    if st.sidebar.button("Sair"):
        st.session_state.logado = False
        st.session_state.usuario = ""
        st.rerun()

    # Função para salvar o gráfico em buffer
    def salvar_grafico(fig):
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        return buffer

    # Carrega o modelo ao iniciar a aplicação
    modelo = carregar_modelo()

    # Carregar as classes
    with open("Engine_classes.json", "r") as f:
        encoder_classes_engine = json.load(f)
        
    with open("Transmission_classes.json", "r") as f:
        encoder_classes_transmission = json.load(f)
        
    with open("Calibration_classes.json", "r") as f:
        encoder_classes_calibration = json.load(f)

    # Entrada de dados para o modelo
    # Lista com as opções

    options_transmission = list(encoder_classes_transmission)

    options_model_PBT = ["29", "28"]

    options_hp_model = ["440","460","480","520", "530", "540"]

    options_engine_model = list(encoder_classes_engine)


    # Criar cartões no Streamlit
    col1, col2, col3 = st.columns(3)

    # Selectbox para o usuário escolher uma opção
    with col1:
        PBT_model = st.selectbox("Model:", options_model_PBT)
        hp_model = st.selectbox("Power model:", options_hp_model)
    with col2:
        Load = st.number_input("Load condition: ")
        hp_real = st.number_input("Current power: ")
    with col3:
        Transmission = st.selectbox("Transmission Model:", options_transmission)
        Engine_model = st.selectbox("Engine model:", options_engine_model)

    path_excel = os.path.join(os.getcwd(), "Base-Consumo.xlsx")
    df = pd.read_excel(path_excel, engine = 'openpyxl', sheet_name='MN')  
    df = df[['Engine', 'Engine Calibration']]

    options_engine_calibration = df[df['Engine']==Engine_model]["Engine Calibration"].dropna().unique()
    Engine_Calibration = st.selectbox("Engine Calibration:", options_engine_calibration)

    Engine_encoder = LabelEncoder()
    Engine_encoder.classes_ = np.array(encoder_classes_engine)

    Transmission_encoder = LabelEncoder()
    Transmission_encoder.classes_ = np.array(encoder_classes_transmission)

    Calibration_encoder = LabelEncoder()
    Calibration_encoder.classes_ = np.array(encoder_classes_calibration)

    if "predict_earlier" not in st.session_state:
        st.session_state.predict_earlier = []

    if "table_earlier" not in st.session_state:
        st.session_state.table_earlier = []

    # Inicializa o contador no estado da sessão, se não estiver configurado
    if "predict_count" not in st.session_state:
        st.session_state.predict_count = 0

    if "predict" not in st.session_state:
        st.session_state.predict = []
        
    # Botão para realizar a previsão
    if st.button("Predict"):
        # Incrementa o contador sempre que o botão é pressionado
        st.session_state.predict_count += 1
        # Armazenar os valores no session state temporário
        st.session_state.PBT_model = PBT_model
        st.session_state.hp_model = hp_model
        st.session_state.Load = Load
        st.session_state.hp_real = hp_real
        st.session_state.Transmission = Transmission
        st.session_state.Engine_model = Engine_model
        st.session_state.Engine_Calibration = Engine_Calibration
        
        # Cria um DataFrame com os dados de entrada
        dados = pd.DataFrame([[st.session_state.PBT_model, st.session_state.hp_model, st.session_state.Load, st.session_state.hp_real, st.session_state.Transmission, 
                            st.session_state.Engine_model, st.session_state.Engine_Calibration]])  # Adicione mais colunas conforme necessário
        dados.columns = ['PBT', 'hp', 'Weight', 'HP Real','Transmission model', 'Engine', 'Engine Calibration']
        dados['PBT'] = dados['PBT'].astype(int)
        dados['hp'] = dados['hp'].astype(int)
        dados['Weight'] = dados['Weight'].astype(int)
        dados['HP Real'] = dados['HP Real'].astype(int)
        dados['hp'] = dados['hp'].astype(int)
        dados['Weight'] = dados['Weight'].astype(int)
        st.session_state.predict_earlier.extend(dados.to_numpy().tolist())
        
        dados['Engine'] = Engine_encoder.transform(dados["Engine"])
        dados['Transmission model'] = Transmission_encoder.transform(dados["Transmission model"])
        dados['Engine Calibration'] = Calibration_encoder.transform(dados["Engine Calibration"])

        for column in ['Engine', 'Engine Calibration', 'Transmission model']:
            dados[column] = dados[column].astype(int)  # Convertendo os valores para int
        
        with open("scaler.pkl", "rb") as scaler_file:
            scaler = joblib.load(scaler_file)
        
        dados = scaler.transform(dados)
        st.session_state.table_earlier.extend(dados.tolist())
        
        # Usa a função prever() do modelo carregado
        previsao = modelo.predict(dados)
        st.session_state.predict.append(previsao[0].round(2))

    # Gráfico
    for index, item in enumerate(st.session_state.predict_earlier):
        st.write(f"### {index + 1}ª Iteration:")
        PBT_model = int(item[0])
        hp_model = int(item[1])
        Load = int(item[2])
        hp_real = int(item[3])
        Transmission = item[4]
        Engine_model = item[5]
        Engine_Calibration = item[6]
        Weight_range = np.arange(Load - 20000, Load + 20000 + 5000, 5000)
        # Gerar previsões com base nos pesos
        data_list = []
        for weight in Weight_range:
            data_list.append([PBT_model, hp_model, weight, hp_real, Transmission, Engine_model, Engine_Calibration])
        dados = pd.DataFrame(data_list, columns=['PBT', 'hp', 'Weight', 'HP Real', 'Transmission model', 'Engine', 'Engine Calibration'])
        dados['Engine'] = Engine_encoder.transform(dados["Engine"])
        dados['Transmission model'] = Transmission_encoder.transform(dados["Transmission model"])
        dados['Engine Calibration'] = Calibration_encoder.transform(dados["Engine Calibration"])
        dados = scaler.transform(dados)
        # Fazer previsões
        previsoes = modelo.predict(dados).round(3)
        fig, ax = plt.subplots(figsize=(6, 4), dpi=500)
        ax.plot((Weight_range / 1000), previsoes, marker='o', linestyle='-', color='#1D3E73')
        ax.set_xlabel('Weight (ton)', fontsize=12)
        ax.set_ylabel('Consumption (km/L)', fontsize=12)
        ax.grid(True, color='lightgray', linestyle='--')
        # Remover o contorno do gráfico
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        # Adicionando labels aos dados
        for i, txt in enumerate(previsoes):
            ax.annotate(txt, ((Weight_range[i]/1000), previsoes[i]), fontsize=9)
        fig.patch.set_facecolor('white')  # Define a cor de fundo
        fig.patch.set_alpha(0.7)
        
        buffer = salvar_grafico(fig)
        # Exibir o gráfico e a tabela
        cols = st.columns([2, 1])
        with cols[0]:
            st.image(buffer, caption="Consumption Predict", use_column_width=False, width=600)
        with cols[1]:
            data_list = pd.DataFrame(st.session_state.predict_earlier[index], columns=['Values'])
            data_list.loc[len(data_list)] = [st.session_state.predict[index]]
            labels = ['PBT', 'hp', 'Weight', 'HP Real', 'Transmission model', 'Engine', 'Engine Calibration', 'Consumption']
            data_list.index = labels
            st.dataframe(data_list)


