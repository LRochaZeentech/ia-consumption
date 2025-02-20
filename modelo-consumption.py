import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.layers import Dropout
import pickle

df = pd.read_excel(r'C:\Users\LUCASDOSSANTOSROCHA\OneDrive - Zeentech Brasil\Área de Trabalho\Consumo IA\Base-Consumo.xlsx', engine = 'openpyxl', sheet_name='MN')
df = df.drop(columns='Vehicle')
df = df.drop(columns='Torque')
df = df.drop(columns='Tração')
df = df.drop(columns='Route')
df = df.drop(columns='RPM')
df = df.drop(columns='Rear Axle')
df = df.drop(columns='Max Torque')
df = df.drop(columns='Min Torque')
df = df.drop(columns='Engine Model')
df = df.drop(columns='Transmission Calibration')
df['PBT'] = df['PBT'].astype(int)
df['hp'] = df['hp'].astype(int)
df['Weight'] = df['Weight'].astype(int)
df['HP Real'] = df['HP Real'].astype(int)

# Convertendo a coluna 'Consumption' para float
df['Consumption'] = df['Consumption'].astype(float)

Engine_encoder = LabelEncoder()
Transmission_encoder = LabelEncoder()
Calibration_encoder = LabelEncoder()

df['Engine'] = Engine_encoder.fit_transform(df['Engine'])
df['Engine Calibration'] = Calibration_encoder.fit_transform(df['Engine Calibration'])
df['Transmission model'] = Transmission_encoder.fit_transform(df['Transmission model'])

# Aplicando LabelEncoder para cada coluna categórica e convertendo para float
for column in ['Engine', 'Engine Calibration', 'Transmission model']:
    df[column] = df[column].astype(int)  # Convertendo os valores para float
    
y = df.iloc[:,7].values
X = df.iloc[:,[0,1,2,3,4,5,6]].values

X_train, X_test, y_treinamento, y_test = train_test_split(X,y,test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Agora vou construir a rede neural
modelo = Sequential()

# modelo validado
modelo.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))  # 1ª camada
modelo.add(Dropout(0.2))
modelo.add(Dense(units=64, activation='swish'))  # 2ª camada
modelo.add(Dropout(0.2))
modelo.add(Dense(units=32, activation='swish'))  # 3ª camada 
modelo.add(Dropout(0.2))
modelo.add(Dense(units=32, activation='swish'))  # 4ª camada 
modelo.add(Dropout(0.2))
modelo.add(Dense(units=16, activation='swish'))  # 5ª camada
modelo.add(Dropout(0.2))
modelo.add(Dense(units=16, activation='swish'))  # 6ª camada
modelo.add(Dropout(0.2))
modelo.add(Dense(units=4, activation='swish'))   # 7ª camada
modelo.add(Dropout(0.2))
modelo.add(Dense(units=2, activation='swish'))   # 8ª camada 
modelo.add(Dropout(0.2))
# Adicionando a camada de saída
modelo.add(Dense(units=1, activation='linear'))  # Regressão linear na saída

# Compilando o modelo
modelo.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Treinando o modelo
modelo.fit(X_train, y_treinamento, epochs=500, batch_size=6) #batch=6

# Fazendo previsões no conjunto de teste
previsoes = modelo.predict(X_test)

# Salva o modelo em um arquivo .pkl
with open("modelo.pkl", "wb") as arquivo:
    pickle.dump(modelo, arquivo)

import json
    
# Salvar as classes de cada encoder em arquivos JSON
Engine_classes = Engine_encoder.classes_.tolist()
with open("Engine_classes.json", "w") as f:
    json.dump(Engine_encoder.classes_.tolist(), f)
    
Transmission_classes = Transmission_encoder.classes_.tolist()
with open("Transmission_classes.json", "w") as f:
    json.dump(Transmission_encoder.classes_.tolist(), f)

Calibration_classes = Calibration_encoder.classes_.tolist()
with open("Calibration_classes.json", "w") as f:
    json.dump(Calibration_encoder.classes_.tolist(), f)

# Salvar o scaler
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)




