import json
import umap
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import logging

# ===== START LOGGER =====
logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
sh.setFormatter(formatter)
root_logger.addHandler(sh)


network_df = pd.read_csv("outputs/network_df.csv")

# Prep data
network_df["target_title"] = network_df["target_title"].fillna("")
network_df["topic_id"] = network_df["topic_id"].astype(str)
topic_ids = [str(i) for i in range(len(network_df["topic_id"].unique()))]


def tsne_to_cyto(tsne_val, scale_factor=40):
    return int(scale_factor * (float(tsne_val)))


def get_node_list(in_df):  # Convert DF data to node list for cytoscape
    return [
        {
            "data": {
                "id": str(i),
                "label": str(i),
                "title": row["title"],
                "pub_date": row["date"],
                "authors": row["author"],
                "target_txt": row["target_title"],
                "n_target_txt": row["n_target_title"],
                "node_size": int(np.sqrt(1 + row["n_target_title"]) * 2),
            },
            "position": {"x": tsne_to_cyto(row["x"]), "y": tsne_to_cyto(row["y"])},
            "classes": row["topic_id"],
            "selectable": True,
            "grabbable": False,
        }
        for i, row in in_df.iterrows()
    ]


def get_node_locs(in_df, dim_red_algo="tsne", tsne_perp=40):
    logger.info(
        f"Starting dimensionality reduction on {len(in_df)} nodes, with {dim_red_algo}"
    )

    if dim_red_algo == "tsne":
        node_locs = TSNE(
            n_components=2,
            perplexity=tsne_perp,
            n_iter=300,
            n_iter_without_progress=100,
            learning_rate=150,
            random_state=23,
        ).fit_transform(in_df[topic_ids].values)
    elif dim_red_algo == "umap":
        reducer = umap.UMAP(n_components=2)
        node_locs = reducer.fit_transform(in_df[topic_ids].values)
    else:
        logger.error(
            f"Dimensionality reduction algorithm {dim_red_algo} is not a valid choice! Something went wrong"
        )
        node_locs = np.zeros([len(in_df), 2])

    logger.info("Finished dimensionality reduction")

    x_list = node_locs[:, 0]
    y_list = node_locs[:, 1]

    return x_list, y_list


default_tsne = 40


def update_node_data(dim_red_algo, tsne_perp, in_df):
    (x_list, y_list) = get_node_locs(in_df, dim_red_algo, tsne_perp=tsne_perp)

    x_range = max(x_list) - min(x_list)
    y_range = max(y_list) - min(y_list)
    # print("Ranges: ", x_range, y_range)

    scale_factor = int(4000 / (x_range + y_range))
    in_df["x"] = x_list
    in_df["y"] = y_list

    node_list_in = get_node_list(in_df)
    for i in range(len(in_df)):
        node_list_in[i]["position"]["x"] = tsne_to_cyto(
            x_list[i], scale_factor)
        node_list_in[i]["position"]["y"] = tsne_to_cyto(
            y_list[i], scale_factor)

    return node_list_in


def draw_edges(in_df=network_df):
    conn_list_out = list()

    for i, row in in_df.iterrows():
        tgts = row["target_title"]

        if len(tgts) == 0:
            tgt_list = []
        else:
            tgt_list = tgts.split("/")

        for tgt in tgt_list:
            if tgt in in_df.title.values.tolist():
                tgt_topic = row["topic_id"]
                src_ = str(int(in_df.title.values.tolist().index(tgt)))
                temp_dict = {
                    "data": {"source": src_, "target": str(i)},
                    "classes": tgt_topic,
                    "tgt_topic": tgt_topic,
                    "src_topic": in_df[in_df['title'] == tgt]["topic_id"].values[0],
                    "locked": True,
                }
                conn_list_out.append(temp_dict)

    return conn_list_out


min_n_targets = 3
src = ['medium.com', 'confluence']

startup_elms = dict(n_tgt=min_n_targets, doc_src=src)

filt_df = network_df[
    (network_df.n_target_title >= min_n_targets) & (
        network_df.src.isin(src))
]

node_list = get_node_list(filt_df)
edge_list = draw_edges(filt_df)
startup_elms["elm_list"] = node_list + edge_list

'''
nn = dict(n_tgt=min_n_targets, doc_src=src)
nn['elms'] = node_list
with open("outputs/node_list.json", "w") as f:
    json.dump(nn, f)
ee = dict(n_tgt=min_n_targets, doc_src=src)
ee['elms'] = edge_list
with open("outputs/edge_list.json", "w") as f:
    json.dump(ee, f)
'''
with open("outputs/startup_elms.json", "w") as f:
    json.dump(startup_elms, f)
