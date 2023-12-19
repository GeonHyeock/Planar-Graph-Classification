import numpy as np
import pandas as pd
import networkx as nx
import random
import argparse
import os, sys, shutil
from collections import defaultdict
from tqdm import tqdm


def random_partition(n, k):
    result = [1 for _ in range(k)]
    for _ in range(n - k):
        idx = random.randint(0, k - 1)
        result[idx] += 1
    return result


def k_color_graph(n, k):
    # 연결된 2분 그래프 생성
    is_conected = False
    while not is_conected:
        partition = random_partition(n, k)
        n1, n2 = partition[0], partition[1]
        G = nx.bipartite.random_graph(n1, n2, p=random.choice([0.3, 0.5, 0.7]))
        is_conected = nx.is_connected(G)

    # k_color 그래프 생성
    node_dict = nx.algorithms.bipartite.color(G)
    nodes = [[i for i in node_dict if node_dict[i] == j] for j in [0, 1]]
    v = sum(nodes, [])
    for n in range(2, k):
        new_node = range(sum(partition[:n]), sum(partition[: n + 1]))
        for idx, node in enumerate(new_node):
            G.add_node(node)
            edges = set([e for e in random.sample(v, random.randint(1, len(v)))])
            if idx == 0:
                edges = set(edges) | set([p[0] for p in nodes])
            G.add_edges_from([(n, node) for n in edges])
        nodes.append(list(new_node))
        v += list(new_node)
    return G, nodes


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
    with tqdm(total=args.N * (args.k - 1)) as pbar:
        for k in range(2, args.k + 1):
            for _ in range(args.N):
                n = random.randint(5, args.node)
                G, nodes = k_color_graph(n, k)

                # pd.DataFrame(,columns=["color_idx","node"])
                data_name = str(idx).zfill(8) + ".csv"
                data_path = os.path.join(args.folder_path, data_name)
                data = pd.DataFrame(G.edges, columns=("n1", "n2"))
                data.to_csv(data_path, index=False)

                node_info_path = os.path.join(args.folder_path, "NodeInfo_" + data_name)
                node_info = [(idx, n) for idx, node in enumerate(nodes) for n in node]
                node_info = pd.DataFrame(node_info, columns=["color_idx", "node"])
                node_info.to_csv(node_info_path, index=False)

                df["data_path"].append(data_path)
                df["node_info_path"].append(node_info_path)
                df["colors"].append(k)
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


def graph_dict(args):
    df = pd.DataFrame(
        {
            "color": list(range(2, args.k + 1)),
            "label": list(range(args.k - 1)),
        }
    )
    df.to_csv(os.path.join(args.folder_path, "graph_dict.csv"), index=False)


def main(args):
    create_folder(args.folder_path)
    graph_dict(args)
    make_graph(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=2, help="data_version")
    parser.add_argument("--node", default=50, help="그래프 노드의 개수")
    parser.add_argument("--k", default=10, help="생성할 그래프의 최대 채색수")
    parser.add_argument("--N", default=20, help="채색수 별 생성할 그래프")
    args = parser.parse_args()
    args.folder_path = "./data/version_" + str(args.version).zfill(3)

    random.seed(args.version)
    np.random.seed(args.version)
    main(args)
