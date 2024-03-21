import numpy as np
import pandas as pd
import networkx as nx
import random
import argparse
import os, sys, shutil
from collections import defaultdict
from tqdm import tqdm


def random_seed(args):
    random.seed(args.version)
    np.random.seed(args.version)
    os.environ["PYTHONHASHSEED"] = str(args.version)


def create_folder(args):
    if not os.path.exists(args.folder_path):
        os.makedirs(args.folder_path)
    else:
        msg = ""
        while msg not in ["Y", "N"]:
            msg = input(
                f"{args.folder_path}가 존재합니다 \n Y : 기존의 폴더를 삭제 -> 새로 데이터 생성합니다 \n N : 실행 중단 \n"
            )

        if msg == "Y":
            shutil.rmtree(args.folder_path)
            os.makedirs(args.folder_path)
        else:
            sys.exit()


def make_label_graph(args, weight_funtion=lambda x: x * (x - 1)):
    n_range = range(args.min_node, args.max_node + 1)

    is_conected, count = False, 0
    while not is_conected:
        if count // 1000 == 0:
            n = random.choices(n_range, weights=map(weight_funtion, n_range))[0]
            m_range = range(n - 1, n * (n - 1) // 2)
            m = random.choices(m_range)[0]
        G = nx.gnm_random_graph(n, m)
        is_conected = nx.is_connected(G)
        count += 1

    label, r = divmod(sum(nx.triangles(G).values()), 3)
    assert r == 0, "코드확인."
    return G, label


def make_graph(args):
    df, idx = defaultdict(list), 0
    with tqdm(total=args.N) as pbar:
        for _ in range(args.N):
            G, label = make_label_graph(args)

            # 그래프 정보 csv 저장
            data_name = str(idx).zfill(8) + ".csv"
            data_path = os.path.join(args.folder_path, data_name)
            data = pd.DataFrame(G.edges, columns=("n1", "n2"))
            data.to_csv(data_path, index=False)

            df["data_path"].append(data_path)
            df["label_name"].append(label)
            df["node_number"].append(len(G.nodes))
            df["edge_number"].append(len(G.edges))
            idx += 1
            pbar.update(1)

    df = pd.DataFrame(df)
    data_type = ["train"] * 8 + ["valid"] * 1 + ["test"] * 1
    data_type = np.random.permutation(np.array(data_type * (len(df) // 10 + 1))[: len(df)])
    df["type"] = data_type
    df.to_csv(os.path.join(args.folder_path, "label.csv"), index=False)


def label_dict(args):
    if args.label_size:
        df = pd.DataFrame(
            {
                "label_name": args.label_name,
                "label": list(range(args.label_size)),
            }
        )
        df.to_csv(os.path.join(args.folder_path, "label_dict.csv"), index=False)


def main(args):
    random_seed(args)
    create_folder(args)
    label_dict(args)
    make_graph(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=5, help="data_version")
    parser.add_argument("--min_node", default=10, help="그래프 노드의 최소 개수")
    parser.add_argument("--max_node", default=100, help="그래프 노드의 최대 개수")
    parser.add_argument("--N", default=100000, help="Sample_size")
    parser.add_argument("--label_name", nargs="+", default=[], help="데이터 라벨")
    args = parser.parse_args()
    args.folder_path = "./data/version_" + str(args.version).zfill(3)
    args.label_size = len(args.label_name)
    main(args)
