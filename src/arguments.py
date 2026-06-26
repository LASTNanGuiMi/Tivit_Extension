import argparse

VIT_NAME = [
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "facebook/dinov2-small",
    "facebook/dinov2-base",
    "facebook/dinov2-large",
    "google/siglip2-so400m-patch14-224",
    "facebook/vit-mae-base",
    "facebook/vit-mae-large",
    "facebook/vit-mae-huge",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Time Vision Transformer.")

    parser.add_argument(
        "--vit_1_name",
        type=str,
        help="Pretrained weights of vision backbone, either a HuggingFace id or a local path",
    )

    parser.add_argument(
        "--vit_2_name",
        type=str,
        help="Pretrained weights of vision backbone, either a HuggingFace id or a local path",
    )

    parser.add_argument(
        "--vit_1_layer",
        type=int,
        help="Layer of vision backbone from which we extract representations",
    )

    parser.add_argument(
        "--vit_2_layer",
        type=int,
        help="Layer of vision backbone from which we extract representations",
    )

    parser.add_argument(
        "--mantis",
        action="store_true",
        help="Use time series foundation model Mantis",
    )

    parser.add_argument(
        "--mantis_name",
        type=str,
        default="paris-noah/Mantis-8M",
        help="Pretrained weights of Mantis, either a HuggingFace id or a local path",
    )

    parser.add_argument(
        "--moment",
        type=str,
        choices=["small", "base", "large"],
        help="Use time series foundation model MOMENT",
    )

    parser.add_argument(
        "--aggregation",
        type=str,
        help="Aggregation of hidden representations",
    )

    parser.add_argument(
        "--image_mode",
        type=str,
        choices=["line_plot", "activity_graph", "activity_matrix", "segment"],
        default="line_plot",
        help="How to convert each time series into an image for the vision backbone",
    )

    parser.add_argument(
        "--patch_size",
        type=str,
        choices=["sqrt", "linspace"],
        help="How to find the patch size for 2D segmentation",
    )

    parser.add_argument(
        "--stride",
        type=float,
        help="Stride as a fraction of patch size",
    )

    parser.add_argument(
        "--classifier_type",
        type=str,
        choices=[
            "logistic_regression",
            "nearest_centroid",
            "random_forest",
            "mlp",
        ],
        help="Classifier type",
    )

    parser.add_argument(
        "--mlp_hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension of the MLP classifier",
    )

    parser.add_argument(
        "--mlp_num_layers",
        type=int,
        default=2,
        help="Number of linear layers in the MLP classifier",
    )

    parser.add_argument(
        "--mlp_dropout",
        type=float,
        default=0.1,
        help="Dropout probability of the MLP classifier",
    )

    parser.add_argument(
        "--mlp_lr",
        type=float,
        default=1e-4,
        help="Learning rate for MLP/fusion-head training",
    )

    parser.add_argument(
        "--mlp_weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for MLP/fusion-head training",
    )

    parser.add_argument(
        "--mlp_epochs",
        type=int,
        default=20,
        help="Number of training epochs for the MLP classifier",
    )

    parser.add_argument(
        "--mlp_early_stop_patience",
        type=int,
        default=0,
        help="Stop MLP training after this many epochs without val macro F1 improvement",
    )

    parser.add_argument(
        "--modal_interaction",
        type=str,
        choices=["concat", "concat_attn", "cross_attn_gate", "masked_pretrain"],
        default="concat",
        help="How to fuse branch embeddings in the MLP path",
    )

    parser.add_argument(
        "--fusion_dim",
        type=int,
        default=512,
        help="Shared fusion dimension for non-concat interaction modes",
    )

    parser.add_argument(
        "--fusion_heads",
        type=int,
        default=4,
        help="Number of attention heads used by fusion interaction modes",
    )

    parser.add_argument(
        "--cross_attn_query",
        type=str,
        choices=["ts", "visual"],
        default="ts",
        help="Which branch acts as the query in cross_attn_gate",
    )

    parser.add_argument(
        "--mask_prob",
        type=float,
        default=0.3,
        help="Mask probability for masked_pretrain branch reconstruction",
    )

    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=10,
        help="Number of masked_pretrain self-supervised pretraining epochs",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        choices=["ucr", "uea", "uci", "falltl", "feng"],
        help="Time series classification benchmark",
    )

    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        help="Optional dataset names to run within the selected benchmark",
    )

    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for dataloader"
    )

    parser.add_argument(
        "--aeon",
        action="store_true",
        help="Activate aeon data preprocessing",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory where datasets are stored",
    )

    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Path to the directory where results are stored",
    )

    parser.add_argument(
        "--measure_alignment",
        action="store_true",
        help="Measure the alignment of model representations",
    )

    parser.add_argument(
        "--get_intrinsic_dimension",
        action="store_true",
        help="Compute the intrinsic dimension of representations",
    )

    parser.add_argument(
        "--get_principal_components",
        action="store_true",
        help="Compute the number of principal components required to cover 95 percent of the representation variance",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Change random seed for experiments",
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation ratio sampled from the official training split",
    )

    parser.add_argument(
        "--custom_test_ratio",
        type=float,
        default=0.2,
        help="Test ratio for custom CSV datasets without an official split",
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default=200,
        help="Sliding-window length for custom CSV datasets",
    )

    parser.add_argument(
        "--window_stride",
        type=int,
        default=100,
        help="Sliding-window stride for custom CSV datasets",
    )

    parser.add_argument(
        "--max_windows_per_file",
        type=int,
        help="Optional maximum number of windows to keep from each Feng CSV file",
    )

    parser.add_argument(
        "--save_activity_graph_samples",
        type=int,
        default=0,
        help="Number of activity graph sample images to save per dataset",
    )

    parser.add_argument(
        "--save_activity_lineplot_samples",
        type=int,
        default=0,
        help="Number of activity line plot sample images to save per dataset",
    )

    return parser.parse_args()
