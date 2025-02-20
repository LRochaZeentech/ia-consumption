import joblib  # Biblioteca para carregar o modelo salvo

# Função para carregar o modelo
def carregar_modelo():
    # Carrega o modelo do arquivo .pkl
    modelo = joblib.load('modelo.pkl')
    return modelo

# Função para realizar previsões
def prever(modelo, dados):
    return modelo.predict(dados)