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