import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import warnings
from PIL import Image

st.set_page_config(page_title="Crop Recommender", page_icon="🌿", layout='centered', initial_sidebar_state="collapsed")

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def main():
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;">Рекомендация выбора сельскохозяйственной культуры 👨‍🌾</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    N = st.number_input("Азот", 1,10000)
    P = st.number_input("Фосфор", 1,10000)
    K = st.number_input("Калий", 1,10000)
    temp = st.number_input("Температура",0.0,100000.0)
    humidity = st.number_input("Влажность в %", 0.0,100000.0)
    rainfall = st.number_input("Осадки в мм",0.0,100000.0)
    ph = st.number_input("Ph", 0.0,100000.0)
    #ph = st.slider('Ph',0.0,10.0,6.7)
    norm = st.checkbox("Нормализация данных", value=True, disabled=True)
    algo = st.selectbox('Алгоритм для классификации',('GaussianNB', 'RandomForest'))

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1,-1)

    if algo == 'GaussianNB':
        filename = 'gauss.pkl'
    elif algo == 'RandomForest':
        filename = 'rf.pkl'

    if st.button('Прогноз'):
        loaded_model = load_model(filename)
        prediction = loaded_model.predict(single_pred)
        st.write('''
		## Результат 🔍
		''')
        st.success(f"{prediction.item().title()} рекомендуется искусственным интеллектом для вашего поля.")
      #code for html ☘️ 🌾 🌳 👨‍🌾  🍃

        html_temp2 = """
        <div>
        <h2 style="color:MEDIUMSEAGREEN;text-align:left;">Параметры модели искусственного интеллекта</h2>
        </div>
        """
        st.markdown(html_temp2, unsafe_allow_html=True)

        image1 = Image.open('pairplot.png')
        st.image(image1, caption='Парные отношения между параметрами модели')
        image2 = Image.open('corr_matrix.png')
        st.image(image2, caption='Корреляционная матрица')
        if algo == 'GaussianNB':
    	    image3 = Image.open('gauss_conf_matrix.png')
        elif algo == 'RandomForest':
    	    image3 = Image.open('rf_conf_matrix.png')
        st.image(image3, caption='Матрица ошибок')
        if algo == 'RandomForest':
            image4 = Image.open('rf_feature_importance.png')
            st.image(image4, caption='Важность признаков')


    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()
