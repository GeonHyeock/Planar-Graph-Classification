import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import os
from matplotlib import pyplot as plt
from glob import glob


def graph_visual(data):
    G = nx.from_edgelist(pd.read_csv(data.data_path).values.astype(int))
    fig = plt.figure(figsize=(5, 5))
    nx.draw(G)
    st.pyplot(fig)
    st.dataframe(data.astype(str))


def gh_plot(df):
    sns.set(font_scale=3)
    for col in ["colors", "node_number", "edge_number"]:
        fig = plt.figure(figsize=(30, 10))
        sns.kdeplot(data=df, x=col)
        plt.xlim(0, max(df[col]) * 1.1)
        st.pyplot(fig)

    sns.set(font_scale=1)
    fig = plt.figure(figsize=(10, 10))
    sns.scatterplot(data=df, x="edge_number", y="colors")
    st.pyplot(fig)


def main():
    st.title("Chromatic Number of Graphs")
    with st.sidebar:
        data_path = st.selectbox("data version을 선택해주세요.", glob("data/version*"))
        df = pd.read_csv(os.path.join(data_path, "label.csv"))
        data_type = st.radio("데이터 타입을 선택해주세요", ["all", "train", "valid", "test"])
        if data_type != "all":
            df = df[df["type"] == data_type]

        idx = st.slider("data를 선택해주세요.", 0, len(df) - 1, 0, 1)

    with st.expander("Graph"):
        graph_visual(df.iloc[idx])

    with st.expander("DATA EDA"):
        gh_plot(df)


if __name__ == "__main__":
    main()
