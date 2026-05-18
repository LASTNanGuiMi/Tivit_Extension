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
        choices=["line_plot", "segment"],
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
        ],
        help="Classifier type",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        choices=["ucr", "uea"],
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

    return parser.parse_args()
