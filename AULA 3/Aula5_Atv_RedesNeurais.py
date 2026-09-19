import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# dados simples
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])

# criação do modelo
model = keras.Sequential([
    layers.Dense(4, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# compilação do modelo
model.compile(optimizer='adam', loss='binary_crossentropy')

# treinamento
model.fit(X, y, epochs=200, verbose=0)

# previsão
resultado = model.predict([[4]])
print(resultado)


'''
Atividade 2
'''
import pandas as pd
import numpy as np
import tensorflow as tf

# ==========================================
# 1. PREPARAÇÃO DOS DADOS
# ==========================================
# Criamos o DataFrame com a relação entre horas jogadas e nível de cansaço
gamer = pd.DataFrame({
    'horas_jogo': [1, 2, 4, 6, 8, 10],
    'cansaco':    [1, 2, 3, 5, 8, 10]
})

# O TensorFlow trabalha de forma otimizada com arrays NumPy do tipo float32
X_train = np.array(gamer['horas_jogo'], dtype=np.float32)
y_train = np.array(gamer['cansaco'], dtype=np.float32)

# ==========================================
# 2. DEFINIÇÃO DA ARQUITETURA DO MODELO
# ==========================================
# Criamos um modelo sequencial com apenas 1 camada densa de 1 neurônio.
# Essa estrutura equivale matematicamente a uma Regressão Linear: y = (w * x) + b
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# ==========================================
# 3. COMPILAÇÃO DO MODELO
# ==========================================
# Definimos o otimizador (Adam ou SGD) e a função de perda (Erro Quadrático Médio)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
    loss='mean_squared_error'
)

# ==========================================
# 4. TREINAMENTO DO MODELO
# ==========================================
# Ajustamos o modelo aos dados ao longo de 600 épocas (iterações completas)
print("Iniciando o treinamento da rede neural...")
history = model.fit(X_train, y_train, epochs=600, verbose=0)
print("Treinamento concluído com sucesso!")

# ==========================================
# 5. INFERÊNCIA / PREVISÃO DE TESTE
# ==========================================
# Vamos prever o nível de cansaço para alguém que jogou por 5 horas seguidas
horas_teste = np.array([[5.0]], dtype=np.float32)
cansaco_previsto = model.predict(horas_teste)

print(f"\n--- Resultado da Previsão ---")
print(f"Horas de jogo: {horas_teste[0][0]}h")
print(f"Nível de cansaço estimado: {cansaco_previsto[0][0]:.2f} / 10")

'''
Atividade 3
'''
import pandas as pd
import numpy as np
import tensorflow as tf

# ==========================================
# 1. PREPARAÇÃO DOS DADOS
# ==========================================
# Criando o DataFrame com os dados de temperatura e vendas
sorvete = pd.DataFrame({
    'temperatura': [18, 20, 24, 27, 30, 35],
    'vendas': [20, 25, 40, 55, 70, 100]
})

# Convertendo as colunas para arrays NumPy do tipo float32 (formato ideal para o TensorFlow)
X_train = np.array(sorvete['temperatura'], dtype=np.float32)
y_train = np.array(sorvete['vendas'], dtype=np.float32)

# ==========================================
# 2. DEFINIÇÃO DO MODELO (Regressão Linear)
# ==========================================
# Criamos um modelo sequencial com 1 camada densa de 1 neurônio,
# o equivalente a uma Regressão Linear: y = (w * x) + b
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# ==========================================
# 3. COMPILAÇÃO DO MODELO
# ==========================================
# Configuramos o otimizador Adam e a função de perda MSE (Erro Quadrático Médio)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error'
)

# ==========================================
# 4. TREINAMENTO DO MODELO
# ==========================================
# Treinamos a rede ao longo de 1000 épocas para que aprenda a relação dos dados
print("Iniciando o treinamento do modelo...")
history = model.fit(X_train, y_train, epochs=1000, verbose=0)
print("Treinamento concluído com sucesso!")

# ==========================================
# 5. INFERÊNCIA / PREVISÃO DE TESTE
# ==========================================
# Testando o modelo para prever a quantidade de vendas em um dia de 32°C
temp_teste = np.array([[32.0]], dtype=np.float32)
vendas_previstas = model.predict(temp_teste)

print(f"\n--- Resultado da Previsão ---")
print(f"Temperatura informada: {temp_teste[0][0]}°C")
print(f"Estimativa de sorvetes vendidos: {vendas_previstas[0][0]:.0f} unidades")

'''
Atividade 4
'''
import pandas as pd
import numpy as np
import tensorflow as tf

# ==========================================
# 1. PREPARAÇÃO DOS DADOS
# ==========================================
# Criando o DataFrame com os dados de faltas e resultado (1 = Aprovado, 0 = Reprovado)
alunos = pd.DataFrame({
    'faltas': [0, 1, 2, 5, 7, 10],
    'resultado': [1, 1, 1, 0, 0, 0]
})

# Convertendo as colunas para arrays NumPy do tipo float32
X_train = np.array(alunos['faltas'], dtype=np.float32)
y_train = np.array(alunos['resultado'], dtype=np.float32)

# ==========================================
# 2. DEFINIÇÃO DO MODELO (Regressão Logística / Classificação Binária)
# ==========================================
# Usamos 1 neurônio com a função de ativação 'sigmoid' para transformar 
# o resultado em uma probabilidade entre 0 e 1.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1], activation='sigmoid')
])

# ==========================================
# 3. COMPILAÇÃO DO MODELO
# ==========================================
# Para classificação binária, utilizamos a perda 'binary_crossentropy'
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# 4. TREINAMENTO DO MODELO
# ==========================================
# Treinamos o modelo ao longo de 500 épocas
print("Iniciando o treinamento do modelo de classificação...")
history = model.fit(X_train, y_train, epochs=500, verbose=0)
print("Treinamento concluído com sucesso!")

# ==========================================
# 5. INFERÊNCIA / PREVISÃO DE TESTE
# ==========================================
# Vamos prever o resultado para um aluno com 3 faltas
faltas_teste = np.array([[3.0]], dtype=np.float32)
probabilidade = model.predict(faltas_teste)[0][0]

# Definindo a classe final com base no limiar (threshold) de 0.5
classe_prevista = "Aprovado" if probabilidade >= 0.5 else "Reprovado"

print(f"\n--- Resultado da Previsão ---")
print(f"Número de faltas: {faltas_teste[0][0]}")
print(f"Probabilidade calculada de aprovação: {probabilidade * 100:.2f}%")
print(f"Resultado final: {classe_prevista}")

'''
Atividade 5
'''
import pandas as pd
import numpy as np
import tensorflow as tf

# ==========================================
# 1. PREPARAÇÃO DOS DADOS
# ==========================================
# Criando o DataFrame com os dados de passeios e felicidade
pets = pd.DataFrame({
    'passeios': [1, 2, 3, 4, 5],
    'felicidade': [2, 4, 5, 8, 10]
})

# Convertendo as colunas para arrays NumPy do tipo float32
X_train = np.array(pets['passeios'], dtype=np.float32)
y_train = np.array(pets['felicidade'], dtype=np.float32)

# ==========================================
# 2. DEFINIÇÃO DO MODELO (Regressão Linear)
# ==========================================
# Criamos um modelo sequencial contendo 1 camada densa de 1 neurônio,
# o equivalente a uma Regressão Linear: y = (w * x) + b
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# ==========================================
# 3. COMPILAÇÃO DO MODELO
# ==========================================
# Configuramos o otimizador Adam e a função de perda MSE (Erro Quadrático Médio)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error'
)

# ==========================================
# 4. TREINAMENTO DO MODELO
# ==========================================
# Treinamos a rede ao longo de 800 épocas
print("Iniciando o treinamento do modelo...")
history = model.fit(X_train, y_train, epochs=800, verbose=0)
print("Treinamento concluído com sucesso!")

# ==========================================
# 5. INFERÊNCIA / PREVISÃO DE TESTE
# ==========================================
# Testando o modelo para prever a felicidade de um pet com 3.5 passeios no dia
passeios_teste = np.array([[3.5]], dtype=np.float32)
felicidade_prevista = model.predict(passeios_teste)

print(f"\n--- Resultado da Previsão ---")
print(f"Passeios por dia: {passeios_teste[0][0]}")
print(f"Estimativa do nível de felicidade: {felicidade_prevista[0][0]:.2f} / 10")

'''
Atividade 6
'''

import pandas as pd
import numpy as np
import tensorflow as tf

# ==========================================
# 1. PREPARAÇÃO DOS DADOS
# ==========================================
# Criando o DataFrame com os dados de duração e nota
filmes = pd.DataFrame({
    'duracao': [80, 90, 100, 110, 120],
    'nota': [4, 5, 7, 8, 9]
})

# Convertendo as colunas para arrays NumPy do tipo float32
X_train = np.array(filmes['duracao'], dtype=np.float32)
y_train = np.array(filmes['nota'], dtype=np.float32)

# ==========================================
# 2. DEFINIÇÃO DO MODELO (Regressão Linear)
# ==========================================
# Modelo sequencial com 1 camada densa de 1 neurônio,
# equivalente a uma Regressão Linear: nota = (w * duracao) + b
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# ==========================================
# 3. COMPILAÇÃO DO MODELO
# ==========================================
# Configuramos o otimizador Adam e a função de perda MSE (Erro Quadrático Médio)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
    loss='mean_squared_error'
)

# ==========================================
# 4. TREINAMENTO DO MODELO
# ==========================================
# Treinamos a rede ao longo de 800 épocas
print("Iniciando o treinamento do modelo...")
history = model.fit(X_train, y_train, epochs=800, verbose=0)
print("Treinamento concluído com sucesso!")

# ==========================================
# 5. INFERÊNCIA / PREVISÃO DE TESTE
# ==========================================
# Prevendo a nota de um filme com 105 minutos de duração
duracao_teste = np.array([[105.0]], dtype=np.float32)
nota_prevista = model.predict(duracao_teste)

print(f"\n--- Resultado da Previsão ---")
print(f"Duração do filme: {duracao_teste[0][0]} minutos")
print(f"Nota estimada: {nota_prevista[0][0]:.1f} / 10")

'''
Atividade 7
'''
import pandas as pd
import numpy as np
import tensorflow as tf

# ==========================================
# 1. PREPARAÇÃO DOS DADOS
# ==========================================
# Criando o DataFrame com os dados de tamanho e preço
pizza = pd.DataFrame({
    'tamanho': [20, 25, 30, 35, 40],
    'preco': [20, 30, 40, 50, 60]
})

# Convertendo as colunas para arrays NumPy do tipo float32
X_train = np.array(pizza['tamanho'], dtype=np.float32)
y_train = np.array(pizza['preco'], dtype=np.float32)

# ==========================================
# 2. DEFINIÇÃO DO MODELO (Regressão Linear)
# ==========================================
# Criamos um modelo sequencial contendo 1 camada densa de 1 neurônio,
# equivalente a uma Regressão Linear: preco = (w * tamanho) + b
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# ==========================================
# 3. COMPILAÇÃO DO MODELO
# ==========================================
# Configuramos o otimizador Adam e a função de perda MSE (Erro Quadrático Médio)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error'
)

# ==========================================
# 4. TREINAMENTO DO MODELO
# ==========================================
# Treinamos a rede ao longo de 800 épocas para encontrar os pesos ideais
print("Iniciando o treinamento do modelo...")
history = model.fit(X_train, y_train, epochs=800, verbose=0)
print("Treinamento concluído com sucesso!")

# ==========================================
# 5. INFERÊNCIA / PREVISÃO DE TESTE
# ==========================================
# Testando o modelo para prever o preço de uma pizza de 32 cm
tamanho_teste = np.array([[32.0]], dtype=np.float32)
preco_previsto = model.predict(tamanho_teste)

print(f"\n--- Resultado da Previsão ---")
print(f"Tamanho da pizza: {tamanho_teste[0][0]} cm")
print(f"Preço estimado: R$ {preco_previsto[0][0]:.2f}")

'''
Atividade 8
'''
import pandas as pd
import numpy as np
import tensorflow as tf

# ==========================================
# 1. PREPARAÇÃO DOS DADOS
# ==========================================
# Criando o DataFrame com os dados de BPM e pontuação de viralização
musica = pd.DataFrame({
    'bpm': [80, 90, 100, 120, 140],
    'viral': [1, 2, 4, 7, 10]
})

# Convertendo as colunas para arrays NumPy do tipo float32 (formato ideal para o TensorFlow)
X_train = np.array(musica['bpm'], dtype=np.float32)
y_train = np.array(musica['viral'], dtype=np.float32)

# ==========================================
# 2. DEFINIÇÃO DO MODELO (Regressão Linear)
# ==========================================
# Criamos um modelo sequencial com 1 camada densa de 1 neurônio,
# equivalente a uma Regressão Linear: viral = (w * bpm) + b
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# ==========================================
# 3. COMPILAÇÃO DO MODELO
# ==========================================
# Configuramos o otimizador Adam e a função de perda MSE (Erro Quadrático Médio)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
    loss='mean_squared_error'
)

# ==========================================
# 4. TREINAMENTO DO MODELO
# ==========================================
# Treinamos a rede ao longo de 800 épocas para encontrar a relação entre os dados
print("Iniciando o treinamento do modelo...")
history = model.fit(X_train, y_train, epochs=800, verbose=0)
print("Treinamento concluído com sucesso!")

# ==========================================
# 5. INFERÊNCIA / PREVISÃO DE TESTE
# ==========================================
# Testando o modelo para prever a chance de uma música de 110 BPM viralizar
bpm_teste = np.array([[110.0]], dtype=np.float32)
viral_previsto = model.predict(bpm_teste)

print(f"\n--- Resultado da Previsão ---")
print(f"BPM da música: {bpm_teste[0][0]}")
print(f"Índice estimado de chance de viralizar: {viral_previsto[0][0]:.2f} / 10")

'''
Atividade 9
'''
import pandas as pd
import numpy as np
import tensorflow as tf

# ==========================================
# 1. PREPARAÇÃO DOS DADOS
# ==========================================
# Criando o DataFrame com os dados de xícaras de café e energia
cafe = pd.DataFrame({
    'xicaras': [1, 2, 3, 4, 5],
    'energia': [2, 4, 6, 8, 10]
})

# Convertendo as colunas para arrays NumPy do tipo float32
X_train = np.array(cafe['xicaras'], dtype=np.float32)
y_train = np.array(cafe['energia'], dtype=np.float32)

# ==========================================
# 2. DEFINIÇÃO DO MODELO (Regressão Linear)
# ==========================================
# Criamos um modelo sequencial contendo 1 camada densa de 1 neurônio,
# equivalente a uma Regressão Linear: energia = (w * xicaras) + b
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# ==========================================
# 3. COMPILAÇÃO DO MODELO
# ==========================================
# Configuramos o otimizador Adam e a função de perda MSE (Erro Quadrático Médio)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error'
)

# ==========================================
# 4. TREINAMENTO DO MODELO
# ==========================================
# Treinamos a rede ao longo de 800 épocas para encontrar os pesos ideais
print("Iniciando o treinamento do modelo...")
history = model.fit(X_train, y_train, epochs=800, verbose=0)
print("Treinamento concluído com sucesso!")

# ==========================================
# 5. INFERÊNCIA / PREVISÃO DE TESTE
# ==========================================
# Testando o modelo para prever a energia após tomar 3.5 xícaras de café
xicaras_teste = np.array([[3.5]], dtype=np.float32)
energia_prevista = model.predict(xicaras_teste)

print(f"\n--- Resultado da Previsão ---")
print(f"Xícaras de café: {xicaras_teste[0][0]}")
print(f"Nível de energia estimado: {energia_prevista[0][0]:.2f} / 10")

'''
Atividade 10
'''

import pandas as pd
import numpy as np
import tensorflow as tf

# Fixando sementes para garantir reprodutibilidade
tf.random.set_seed(42)
np.random.seed(42)

# ==========================================
# 1. PREPARAÇÃO E TRATAMENTO DOS DADOS
# ==========================================
herois = pd.DataFrame({
    'forca': [1, 2, 3, 7, 8, 10],
    'heroi': [0, 0, 0, 1, 1, 1]  # 0 = Fraco, 1 = Forte
})

# Garantindo dados no formato float32 otimizado para o TensorFlow
X_train = np.array(herois[['forca']], dtype=np.float32)
y_train = np.array(herois['heroi'], dtype=np.float32)

# ==========================================
# 2. ARQUITETURA DO MODELO (Classificação Binária)
# ==========================================
# A função 'sigmoid' comprime o resultado final em um intervalo de 0 a 1 (probabilidade)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1], activation='sigmoid')
])

# ==========================================
# 3. COMPILAÇÃO DO MODELO
# ==========================================
# 'binary_crossentropy' é a perda ideal para classificação 0 ou 1
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# 4. CALLBACKS E TREINAMENTO ROBUSTO
# ==========================================
# EarlyStopping interrompe o treino quando o modelo atinge convergência
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=20,
    restore_best_weights=True
)

print("Iniciando o treinamento da Rede Neural dos Super-Heróis...")
history = model.fit(
    X_train, 
    y_train, 
    epochs=1000, 
    callbacks=[early_stop], 
    verbose=0
)
print(f"Treinamento finalizado com sucesso na época {len(history.history['loss'])}!")

# ==========================================
# 5. AVALIAÇÃO E INFERÊNCIA
# ==========================================
# Testando para um novo herói com força igual a 5
forca_teste = np.array([[5.0]], dtype=np.float32)
probabilidade = model.predict(forca_teste, verbose=0)[0][0]

# Convertendo a probabilidade em classe (threshold de 0.5)
rotulo = "Forte 🦸‍♂️" if probabilidade >= 0.5 else "Fraco 🧙‍♂️"

print("\n--- Resultado da Avaliação ---")
print(f"Força informada: {forca_teste[0][0]}")
print(f"Probabilidade calculada de ser forte: {probabilidade * 100:.2f}%")
print(f"Classificação final: {rotulo}")