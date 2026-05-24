import json
import os
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import numpy as np
import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from aeon.datasets.tsc_datasets import multivariate, univariate
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
from momentfm import MOMENTPipeline
from tqdm import tqdm

from src.analysis import (
    get_intrinsic_dimension,
    get_principal_components,
    measure_alignment,
)
from src.arguments import parse_args
from src.classifier import train_classifier
from src.datautils import get_dataloader
from src.embedding import concat_embeddings, embed
from src.tivit import get_tivit
from src.utils import (
    get_patch_size,
    set_random_seed,
    write_result_table,
    write_split_indices,
)


def model_slug(model_name):
    return os.path.basename(os.path.normpath(model_name)).replace("-", "_")


if __name__ == "__main__":
    args = parse_args()

    set_random_seed(args.random_seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    available_models = []

    if args.vit_1_name:
        available_models.append(model_slug(args.vit_1_name))
    if args.vit_2_name:
        available_models.append(model_slug(args.vit_2_name))
    if args.mantis:
        available_models.append("mantis")
    if args.moment:
        available_models.append(f"moment_{args.moment}")

    has_vision_modality = bool(args.vit_1_name or args.vit_2_name)
    has_timeseries_modality = bool(args.mantis or args.moment)

    if not has_vision_modality and not has_timeseries_modality:
        raise ValueError(
            "At least one modality must be enabled: use a ViT model for line-plot "
            "vision, or enable --mantis/--moment for raw time-series embeddings."
        )

    enabled_modalities = []
    if has_vision_modality:
        enabled_modalities.append(f"vision:{args.image_mode}")
    if has_timeseries_modality:
        ts_models = []
        if args.mantis:
            ts_models.append("mantis")
        if args.moment:
            ts_models.append(f"moment-{args.moment}")
        enabled_modalities.append(f"time_series:{'+'.join(ts_models)}")
    print("Enabled modalities:", ", ".join(enabled_modalities))

    available_models = "_".join(available_models)

    result_dir = f"{args.result_dir}/{timestamp}_{args.datasets}_{available_models}_{args.classifier_type}"
    os.makedirs(result_dir, exist_ok=False)

    # Save parsed arguments as json dictionary
    args_dict = vars(args)
    with open(f"{result_dir}/args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.datasets == "ucr":
        datasets = univariate
    elif args.datasets == "uea":
        datasets = multivariate
    else:
        raise ValueError(f"Only UCR and UEA benchmark available.")

    if args.dataset_names:
        unavailable = sorted(set(args.dataset_names) - set(datasets))
        if unavailable:
            raise ValueError(
                f"Dataset(s) not found in {args.datasets.upper()}: {unavailable}"
            )
        datasets = args.dataset_names

    for dataset in tqdm(datasets):
        print(dataset)

        train_loader, train_labels, test_loader, test_labels = get_dataloader(
            dataset, args
        )

        print("Samples: ", len(train_loader.dataset) + len(test_loader.dataset))
        channels, T = train_loader.dataset[0][0].shape

        mantis_embedding = None
        moment_embedding = None
        vision_embedding_1 = None
        vision_embedding_2 = None

        # Embedding with Mantis TSFM
        if args.mantis:
            network = Mantis8M(device=device)
            network = network.from_pretrained(args.mantis_name)
            network = network.to(device)
            mantis_trainer = MantisTrainer(device=device, network=network)

            mantis_embedding = (
                embed(mantis_trainer, train_loader, "mantis", channels, device),
                embed(mantis_trainer, test_loader, "mantis", channels, device),
            )

        # Embedding with MOMENT TSFM
        if args.moment:
            moment = MOMENTPipeline.from_pretrained(
                f"AutonLab/MOMENT-1-{args.moment}",
                model_kwargs={"task_name": "embedding"},
            )
            moment.init()
            moment.to(device).float()
            moment.eval()

            moment_embedding = (
                embed(moment, train_loader, "moment", channels, device),
                embed(moment, test_loader, "moment", channels, device),
            )

        patch_sizes = get_patch_size(patch_size=args.patch_size, T=T)
        if args.image_mode == "line_plot":
            patch_sizes = [None]

        for p in patch_sizes:

            if p:
                print(f"Patch size: {p}")

            # Embedding with TiViT (1st ViT configuration)
            if args.vit_1_name:
                tivit = get_tivit(
                    model_name=args.vit_1_name,
                    model_layer=args.vit_1_layer,
                    aggregation=args.aggregation,
                    stride=args.stride,
                    patch_size=p,
                    image_mode=args.image_mode,
                )
                tivit = tivit.to(device=device)
                tivit.eval()

                vision_embedding_1 = (
                    embed(tivit, train_loader, "tivit", channels, device),
                    embed(tivit, test_loader, "tivit", channels, device),
                )

            # Embedding with TiViT (2nd ViT configuration)
            if args.vit_2_name:
                tivit = get_tivit(
                    model_name=args.vit_2_name,
                    model_layer=args.vit_2_layer,
                    aggregation=args.aggregation,
                    stride=args.stride,
                    patch_size=p,
                    image_mode=args.image_mode,
                )
                tivit = tivit.to(device=device)
                tivit.eval()

                vision_embedding_2 = (
                    embed(tivit, train_loader, "tivit", channels, device),
                    embed(tivit, test_loader, "tivit", channels, device),
                )

            # Linear classification
            if args.classifier_type:
                train_embeds, test_embeds = concat_embeddings(
                    vision_embedding_1,
                    vision_embedding_2,
                    mantis_embedding,
                    moment_embedding,
                )

                acc_val, acc_test, train_indices, val_indices = train_classifier(
                    train_embeds,
                    train_labels,
                    test_embeds,
                    test_labels,
                    args.classifier_type,
                    args.random_seed,
                    args.val_ratio,
                )
                print(f"Test accuracy: {acc_test:.2f}")

                write_split_indices(
                    result_dir=result_dir,
                    dataset=dataset,
                    train_indices=train_indices,
                    val_indices=val_indices,
                    random_seed=args.random_seed,
                    val_ratio=args.val_ratio,
                )

                write_result_table(
                    result_dir=result_dir,
                    dataset=dataset,
                    acc_val=acc_val,
                    acc_test=acc_test,
                    patch_size=p,
                    image_mode=args.image_mode,
                )

            # Measure alignment of representation spaces using mutual kNN
            elif args.measure_alignment:
                measure_alignment(
                    mantis_embedding,
                    moment_embedding,
                    vision_embedding_1,
                    vision_embedding_2,
                    dataset,
                    result_dir,
                )

            # Compute intrinsic dimension or number of principal components
            elif args.get_intrinsic_dimension or args.get_principal_components:
                embeddings = [
                    e
                    for e in [
                        vision_embedding_1,
                        vision_embedding_2,
                        mantis_embedding,
                        moment_embedding,
                    ]
                    if e is not None
                ]

                assert (
                    len(embeddings) == 1
                ), "Compute intrinsic dimensionality only for one model."

                embedding = np.concatenate(embeddings[0], axis=0).transpose(2, 0, 1)

                if args.get_intrinsic_dimension:
                    get_intrinsic_dimension(embedding, dataset, result_dir)
                if args.get_principal_components:
                    get_principal_components(embedding, dataset, result_dir)

            else:
                raise ValueError(
                    "Please choose: linear probing, intrinsic dimension, principal components, alignment."
                )

            torch.cuda.empty_cache()
