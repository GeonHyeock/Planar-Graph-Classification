import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import os
from matplotlib import pyplot as plt
from glob import glob


def my_tab1(data_type, df, idx):
    col1, col2 = st.columns(2)
    with col1:
        graph_visual(df.iloc[idx])
    with col2:
        st.dataframe(df.iloc[idx].astype(str), width=int(1e3))
        if data_type == "test":
            PF = df.iloc[idx]["colors"] == (df.iloc[idx]["predict"] + 1)
            if PF == True:
                st.write("예측 성공")
            else:
                st.write("예측 실패")


def pred_df(data_path, df):
    a, b = data_path.split("/")
    folder_path = os.path.join(a, "result", b)
    paths = glob(folder_path + "/*")
    predict_path = st.selectbox("result를 선택해주세요.", [p.split("/")[-1] for p in paths])
    predict_df = pd.read_csv(os.path.join(folder_path, predict_path))
    predict_df = pd.merge(df, predict_df, how="inner").sort_values(
        "loss", ascending=False
    )
    return predict_df


def graph_visual(data):
    G = nx.from_edgelist(pd.read_csv(data.data_path).values.astype(int))
    fig = plt.figure(figsize=(5, 5))
    nx.draw(G)
    st.pyplot(fig)


def gh_plot(df):
    sns.set(font_scale=3)
    for col in ["colors", "node_number", "edge_number"]:
        fig = plt.figure(figsize=(30, 10))
        sns.kdeplot(data=df, x=col)
        plt.xlim(0, max(df[col]) * 1.1)
        st.pyplot(fig)


def slidebar():
    with st.sidebar:
        data_path = st.selectbox("data version을 선택해주세요.", glob("data/version*"))
        df = pd.read_csv(os.path.join(data_path, "label.csv"))
        data_type = st.radio("데이터 타입을 선택해주세요", ["all", "train", "valid", "test"])
        if data_type != "all":
            df = df[df["type"] == data_type]

        idx = st.slider("data를 선택해주세요.", 0, len(df) - 1, 0, 1)
        return df, idx, data_path, data_type


def main():
    st.title("Chromatic Number of Graphs")

    tab1, tab2 = st.tabs(["Data_View", "EDA"])
    df, idx, data_path, data_type = slidebar()
    if data_type == "test":
        df = pred_df(data_path, df)

    with tab1:
        my_tab1(data_type, df, idx)
    with tab2:
        gh_plot(df)


if __name__ == "__main__":
    main()
