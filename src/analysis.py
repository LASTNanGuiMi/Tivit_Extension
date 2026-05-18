import os

import numpy as np
import pandas as pd
import torch
from dadapy.data import Data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.mutual_knn import mutual_knn


def get_intrinsic_dimension(embedding, dataset, result_dir):
    results = []

    for i in range(len(embedding)):
        data = Data(embedding[i])
        data.remove_identical_points()

        intrinsic_dimension, _, _ = data.return_id_scaling_2NN()

        results.append(
            {
                "dataset": dataset,
                "layer": i,
                "intrinsic_dimension": intrinsic_dimension,
            }
        )

    df_results = pd.DataFrame(results)

    filename = f"{result_dir}/intrinsic_dimensionality.csv"
    file_exists = os.path.isfile(filename)
    df_results.to_csv(filename, mode="a", index=False, header=not file_exists)

    return


def get_principal_components(embedding, dataset, result_dir):
    results = []

    for i in range(len(embedding)):
        X_std = StandardScaler().fit_transform(embedding[i])
        pca = PCA().fit(X_std)
        cum_var = np.cumsum(pca.explained_variance_ratio_)

        max_var = 0.95
        n_components = np.searchsorted(cum_var, max_var) + 1

        results.append(
            {
                "dataset": dataset,
                "layer": i,
                "num_principal_components": n_components,
            }
        )

    df_results = pd.DataFrame(results)

    filename = f"{result_dir}/principal_components.csv"
    file_exists = os.path.isfile(filename)
    df_results.to_csv(filename, mode="a", index=False, header=not file_exists)

    return


def measure_alignment(
    mantis_embedding,
    moment_embedding,
    vision_embedding_1,
    vision_embedding_2,
    dataset,
    result_dir,
):
    embeddings = [
        torch.cat([torch.from_numpy(v[0]), torch.from_numpy(v[1])], dim=0)
        for v in [
            mantis_embedding,
            moment_embedding,
            vision_embedding_1,
            vision_embedding_2,
        ]
        if v is not None
    ]

    assert len(embeddings) == 2, "Alignment can only be measured between two models."

    alignment_score = mutual_knn(embeddings[0], embeddings[1], topk=10)

    df_results = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "alignment_score": alignment_score,
            }
        ]
    )

    filename = f"{result_dir}/alignment_score.csv"
    file_exists = os.path.isfile(filename)
    df_results.to_csv(filename, mode="a", index=False, header=not file_exists)

    return
