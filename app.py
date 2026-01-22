import streamlit as st
import pandas as pd

@st.cache_data()
def load_data():
    game_df = pd.read_csv("output_game_data.csv")
    return game_df

st.title("Analiza Meczu Piłkarskiego")

df = load_data()
st.dataframe(df)

