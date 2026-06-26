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
from src.mlp_classifier import train_mlp_classifier
from src.tivit import get_tivit
from src.utils import (
    get_patch_size,
    save_activity_lineplot_samples,
    save_activity_graph_samples,
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
            "At least one modality must be enabled: use a ViT model for vision, "
            "or enable --mantis/--moment for raw time-series embeddings."
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
    elif args.datasets == "uci":
        datasets = ["UCIHAR"]
    elif args.datasets == "falltl":
        datasets = ["FallTL"]
    elif args.datasets == "feng":
        datasets = ["Feng"]
    else:
        raise ValueError("Only UCR, UEA, UCI, FallTL, and Feng benchmark available.")

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
        if args.image_mode == "activity_graph":
            save_activity_graph_samples(
                result_dir=result_dir,
                dataset=dataset,
                dataloader=train_loader,
                num_samples=args.save_activity_graph_samples,
            )
            save_activity_lineplot_samples(
                result_dir=result_dir,
                dataset=dataset,
                dataloader=train_loader,
                num_samples=args.save_activity_lineplot_samples,
            )
        channels, T = train_loader.dataset[0][0].shape

        mantis_embedding = None
        moment_embedding = None
        vision_embedding_1 = None
        vision_embedding_2 = None
        mantis_model = None
        moment_model = None

        # Embedding with Mantis TSFM
        if args.mantis:
            network = Mantis8M(device=device)
            network = network.from_pretrained(args.mantis_name)
            network = network.to(device)

            if args.classifier_type == "mlp":
                mantis_model = network
            else:
                mantis_model = MantisTrainer(device=device, network=network)
                mantis_embedding = (
                    embed(mantis_model, train_loader, "mantis", channels, device),
                    embed(mantis_model, test_loader, "mantis", channels, device),
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
            moment_model = moment

            if args.classifier_type != "mlp":
                moment_embedding = (
                    embed(moment, train_loader, "moment", channels, device),
                    embed(moment, test_loader, "moment", channels, device),
                )

        patch_sizes = get_patch_size(patch_size=args.patch_size, T=T)
        if args.image_mode in {"line_plot", "activity_graph", "activity_matrix"}:
            patch_sizes = [None]

        for p in patch_sizes:
            tivit_1 = None
            tivit_2 = None

            if p:
                print(f"Patch size: {p}")

            # Embedding with TiViT (1st ViT configuration)
            if args.vit_1_name:
                tivit_1 = get_tivit(
                    model_name=args.vit_1_name,
                    model_layer=args.vit_1_layer,
                    aggregation=args.aggregation,
                    stride=args.stride,
                    patch_size=p,
                    image_mode=args.image_mode,
                )
                tivit_1 = tivit_1.to(device=device)
                tivit_1.eval()

                if args.classifier_type != "mlp":
                    vision_embedding_1 = (
                        embed(tivit_1, train_loader, "tivit", channels, device),
                        embed(tivit_1, test_loader, "tivit", channels, device),
                    )

            # Embedding with TiViT (2nd ViT configuration)
            if args.vit_2_name:
                tivit_2 = get_tivit(
                    model_name=args.vit_2_name,
                    model_layer=args.vit_2_layer,
                    aggregation=args.aggregation,
                    stride=args.stride,
                    patch_size=p,
                    image_mode=args.image_mode,
                )
                tivit_2 = tivit_2.to(device=device)
                tivit_2.eval()

                if args.classifier_type != "mlp":
                    vision_embedding_2 = (
                        embed(tivit_2, train_loader, "tivit", channels, device),
                        embed(tivit_2, test_loader, "tivit", channels, device),
                    )

            # Linear classification
            if args.classifier_type:
                if args.classifier_type == "mlp":
                    val_metrics, test_metrics, train_indices, val_indices = (
                        train_mlp_classifier(
                            train_loader=train_loader,
                            train_labels=train_labels,
                            test_loader=test_loader,
                            test_labels=test_labels,
                            channels=channels,
                            device=device,
                            batch_size=args.batch_size,
                            random_seed=args.random_seed,
                            val_ratio=args.val_ratio,
                            hidden_dim=args.mlp_hidden_dim,
                            num_layers=args.mlp_num_layers,
                            dropout=args.mlp_dropout,
                            lr=args.mlp_lr,
                            weight_decay=args.mlp_weight_decay,
                            epochs=args.mlp_epochs,
                            early_stop_patience=args.mlp_early_stop_patience,
                            modal_interaction=args.modal_interaction,
                            fusion_dim=args.fusion_dim,
                            fusion_heads=args.fusion_heads,
                            cross_attn_query=args.cross_attn_query,
                            mask_prob=args.mask_prob,
                            pretrain_epochs=args.pretrain_epochs,
                            vision_model_1=tivit_1 if args.vit_1_name else None,
                            vision_model_2=tivit_2 if args.vit_2_name else None,
                            mantis_model=mantis_model,
                            moment_model=moment_model,
                        )
                    )
                else:
                    train_embeds, test_embeds = concat_embeddings(
                        vision_embedding_1,
                        vision_embedding_2,
                        mantis_embedding,
                        moment_embedding,
                    )

                    val_metrics, test_metrics, train_indices, val_indices = train_classifier(
                        train_embeds,
                        train_labels,
                        test_embeds,
                        test_labels,
                        args.classifier_type,
                        args.random_seed,
                        args.val_ratio,
                    )
                print(
                    "Val metrics: "
                    + ", ".join(
                        f"{metric}={value:.4f}"
                        for metric, value in val_metrics.items()
                    )
                )
                print(
                    "Test metrics: "
                    + ", ".join(
                        f"{metric}={value:.4f}"
                        for metric, value in test_metrics.items()
                    )
                )

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
                    val_metrics=val_metrics,
                    test_metrics=test_metrics,
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
