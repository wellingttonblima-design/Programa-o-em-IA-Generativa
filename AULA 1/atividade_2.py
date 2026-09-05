import streamlit as st
 
 #streamlit run atividade_2.py

st.header("Formulário de cadastro")
st.write("Para iniciar o atendimento, digite os dados solicitados abaixo")
st.write("Ao final, leia os termos de serviço e clique em 'Aceitar'")

st.text_input("Nome: ")
st.number_input("Idade: ", value = None)
st.text_input("Gênero: ")
st.text_input("Data de Nascimento: ")
st.text_input("Cidade: ")
st.text_input("Nacionalidade: ")

st.checkbox("Aceito os termos de serviço")

st.button("Salvar dados")