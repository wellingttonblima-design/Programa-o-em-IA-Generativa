#streamlit run atividade_3.py

import streamlit as st

curso = st.selectbox('Selecione o curso de interesse (OBS: Só é possível escolher um): ',['Python','SQL','Azure','AWS','Power BI'])
temas =  st.multiselect('Selecione os temas de interesse: ', ['HTML', 'CSS', 'SQL', 'Git'])
