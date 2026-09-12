"""
import streamlit as st                              #interface grafica
import pandas as pd                                 #tratamento de dados
from sklearn.linear_model import LinearRegression   #Tipo de treinamento

st.header ('PREVISÃO DE VENDAS')                    #Cabeçalo/titulo


dados = pd.read_csv('vendas.csv')
df = pd.DataFrame(dados)

X = df[['mes']]
y = df['vendas']

model = LinearRegression().fit(X, y)

venda = st.number_input('Digite o mês: ', value = 0)                   

meses = {
   1: "Janeiro",
   2: "Fevereiro",
   3: "Março",
   4: "Abril",
   5: "Maio",
   6: "Junho",
   7: "Julho",
   8: "Agosto",
   9: "Setembro",
   10: "Outubro",
   11: "Novembro",
   12: "Dezembro"
}

if venda:
    if st.button('Analisar: '):

        #Previsão 

        previsao_venda = model.predict([[venda]])[0]
                                           
        st.write(f'O faturamento previsto para o mês de {meses[venda]} é: R${previsao_venda:.2f}')
"""

#++++++++++++++++++++++++++++++++++++++++++AULA 4+++++++++++++++++++++++++++++++++++++++++++++
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import numpy as np
import streamlit as st  
import pandas as pd                                


#Tempo de uso de produto X número de reclamações

X = np.array([
    [1,1],
    [6,5],
    [3,8],
    [5,2],
    [8,6],
    [6,0]
])

# 0 =  Fica/ 1 = Cancela

y = np.array([0,1,1,0,1,0])

modelo = DecisionTreeClassifier()
modelo.fit(X,y)

uso = st.number_input('Quantidade de vez que o produto foi utilizado: ') 
reclamacoes = st.number_input('Reclamações: ')

if st.button('ANALÍSAR CLIENTES'):
    if modelo.predict([[uso, reclamacoes]]) == 0:
        st.write('CLIENTE CONSOLIDADO')
    else:
        st.write('POSSÍVEL CANCELAMENTO')

print(modelo.predict([[5,3]]))
print(modelo.predict([[8,3]]))
print(modelo.predict([[2,3]]))
print(modelo.predict([[7,2]]))
print(modelo.predict([[56,36]]))
print(modelo.predict([[50,3]]))
print(modelo.predict([[45,23]]))
print(modelo.predict([[15,3]]))
