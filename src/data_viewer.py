import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import torch
import os, sys
from matplotlib import pyplot as plt
from glob import glob


def my_tab1(data_type, df, idx, on_model):
    col1, col2 = st.columns(2)
    with col1:
        G, pos = graph_visual(df.iloc[idx])

    with col2:
        st.dataframe(df.iloc[idx].astype(str), width=int(1e3))

    if on_model:
        number = st.multiselect("node를 선택해주세요.", G.nodes)
        number = [int(num) for num in number]

        G = torch.tensor(nx.to_numpy_array(G)[None], dtype=torch.float32).to("cuda" if torch.cuda.is_available else "cpu")
        pos = {int(k): v for k, v in pos.items()}
        scores = st.session_state.net(G, latent=True)["scores"]
        scores = [s.squeeze().mean(axis=0).cpu().detach().numpy() for s in scores]

        st.write("attention")
        col1, col2 = st.columns(2)
        for idx, score in enumerate(scores):
            my_col = [col1, col2][idx % 2]
            with my_col:
                att_G = nx.from_numpy_array(score)
                fig = plt.figure(figsize=(5, 5))
                plt.title(f"layer : {idx+1}")
                nx.draw_networkx_nodes(att_G, pos)
                nx.draw_networkx_labels(att_G, pos, font_family="sans-serif")
                for a, b in att_G.edges():
                    row_pivot = score[a][:][np.nonzero(score[a][:])[0]].mean()
                    col_pivot = score[:][b][np.nonzero(score[:][b])[0]].mean()

                    w1 = score[a][b] if score[a][b] > row_pivot else 0
                    w2 = score[b][a] if score[b][a] > col_pivot else 0
                    if number and (a in number or b in number and w1 * w2 > 0):
                        nx.draw_networkx_edges(att_G, pos, edgelist=[(a, b)], alpha=(w1 + w2) / 2)
                    elif not number and w1 * w2 > 0:
                        nx.draw_networkx_edges(att_G, pos, edgelist=[(a, b)], alpha=(w1 + w2) / 2)
                st.pyplot(fig)


def graph_visual(data):
    G = nx.from_numpy_array(nx.to_numpy_array(nx.read_adjlist(data.data_path)))
    pos = nx.spring_layout(G, seed=42)
    fig = plt.figure(figsize=(5, 5))
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos, font_family="sans-serif")
    nx.draw_networkx_edges(G, pos, edgelist=G.edges())
    st.pyplot(fig)
    return G, pos


def gh_plot(df):
    for col in ["node_number", "edge_number"]:
        fig = plt.figure(figsize=(30, 10))
        sns.kdeplot(data=df, x=col, hue="label_name")
        plt.xlim(0, max(df[col]) * 1.1)
        st.pyplot(fig)


def slidebar():
    with st.sidebar:
        data_path = st.selectbox("data version을 선택해주세요.", sorted(glob("data/version*"), reverse=True))
        df = pd.read_csv(os.path.join(data_path, "label.csv"))
        label_dict = pd.read_csv(os.path.join(data_path, "label_dict.csv")).to_dict()["label_name"]

        data_type = st.radio("데이터 타입을 선택해주세요", ["all", "train", "valid", "test"])
        if data_type != "all":
            df = df[df["type"] == data_type]

        df.label_name = df.label_name.apply(lambda x: label_dict[x])
        label_name = st.radio("데이터 타입을 선택해주세요", ["all"] + list(df.label_name.unique()))
        if label_name != "all":
            df = df[df["label_name"] == label_name]

        idx = st.slider("data를 선택해주세요.", 0, len(df) - 1, 0, 1)

        on_model = st.checkbox("Model On")

        return df.reset_index(drop=True), idx, data_path, data_type, on_model


def main():
    st.title("Chromatic Number of Graphs")
    sns.set(font_scale=3)

    tab1, tab2 = st.tabs(["Data_View", "EDA"])
    df, idx, data_path, data_type, on_model = slidebar()
    df.sort_values(["node_number", "edge_number"], inplace=True)
    df.reset_index(inplace=True, drop=True)

    if on_model and not "net" in st.session_state:
        sys.path.append("lightning-hydra-template")
        from GCNET.model import GCnet

        ckpt = torch.load("lightning-hydra-template/logs/train/multiruns/2024-03-25_23-40-42/18/checkpoints/epoch_081.ckpt")
        net = GCnet(50, 128, 0, 4, 64, 8, 0.1, [64, 2], True)
        net.load_state_dict({k[4:]: v for k, v in ckpt["state_dict"].items()})
        st.session_state.net = net.to("cuda" if torch.cuda.is_available() else "cpu")

    with tab1:
        my_tab1(data_type, df, idx, on_model)
    with tab2:
        gh_plot(df)


if __name__ == "__main__":
    main()
