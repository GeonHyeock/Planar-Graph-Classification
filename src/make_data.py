import numpy as np
import pandas as pd
import networkx as nx
import random
import argparse
import os, sys, shutil
from collections import defaultdict
from tqdm import tqdm


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        msg = ""
        while msg not in ["Y", "N"]:
            msg = input(
                f"{folder_path}가 존재합니다 \n Y : 기존의 폴더를 삭제 -> 새로 데이터 생성합니다 \n N : 실행 중단 \n"
            )

        if msg == "Y":
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
        else:
            sys.exit()


def make_graph(args):
    df, idx = defaultdict(list), 0
    with tqdm(total=args.N) as pbar:
        while idx < args.N:
            # 1 random Graph
            if idx < args.N // 2:
                n = random.randint(5, args.node)
                e = random.randint(n, (n * (n - 1)) // 2)
                G = nx.gnm_random_graph(n, e)
            # 2 bipartite Graph
            else:
                n1 = random.randint(5, args.node)
                n2 = random.randint(min(args.node - n1, 5), args.node - n1)
                G = nx.bipartite.random_graph(n1, n2, 0.5)

            if nx.is_connected(G):
                colors = len(
                    set(nx.greedy_color(G, "connected_sequential_bfs").values())
                )
                data_path = os.path.join(args.folder_path, str(idx).zfill(8) + ".csv")
                graph = pd.DataFrame(G.edges, columns=("n1", "n2")) + 1
                graph.to_csv(data_path, index=False)
                df["data_path"].append(data_path)
                df["colors"].append(colors)
                df["node_number"].append(len(G.nodes))
                df["edge_number"].append(len(G.edges))
                idx += 1
                pbar.update(1)
    df = pd.DataFrame(df)
    data_type = ["train"] * 8 + ["valid"] * 1 + ["test"] * 1
    data_type = np.random.permutation(
        np.array(data_type * (len(df) // 10 + 1))[: len(df)]
    )
    df["type"] = data_type
    df.to_csv(os.path.join(args.folder_path, "label.csv"), index=False)


def main(args):
    create_folder(args.folder_path)
    make_graph(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=1, help="data_version")
    parser.add_argument("--node", default=50, help="그래프 노드의 개수")
    parser.add_argument("--N", default=10000, help="생성할 그래프의 개수")
    args = parser.parse_args()
    args.folder_path = "./data/version_" + str(args.version).zfill(3)

    random.seed(args.version)
    np.random.seed(args.version)
    main(args)
