import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

from src.classifier import compute_metrics_from_predictions
from src.utils import (
    get_split,
    resize_mantis_input,
    resize_moment_input,
    set_random_seed,
)


FEATURE_CACHE_SCHEMA_VERSION = 2


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, num_classes):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"MLP must contain at least one layer, got {num_layers}.")

        layers = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FusionModule(nn.Module):
    def __init__(
        self,
        branch_dims,
        modal_interaction="concat",
        fusion_dim=512,
        fusion_heads=4,
        cross_attn_query="ts",
    ):
        super().__init__()

        if not branch_dims:
            raise ValueError("FusionModule requires at least one enabled branch.")
        if modal_interaction not in {
            "concat",
            "concat_attn",
            "cross_attn_gate",
            "masked_pretrain",
        }:
            raise ValueError(f"Unsupported modal_interaction: {modal_interaction}")

        self.modal_interaction = modal_interaction
        self.fusion_dim = fusion_dim
        self.fusion_heads = fusion_heads
        self.cross_attn_query = cross_attn_query
        self.branch_dims = list(branch_dims)

        self.projections = nn.ModuleList()
        if self.modal_interaction == "concat":
            self.output_dim = int(sum(self.branch_dims))
        else:
            for dim in self.branch_dims:
                self.projections.append(nn.Linear(dim, fusion_dim))

            if self.modal_interaction == "concat_attn":
                self.output_dim = fusion_dim * len(self.branch_dims)
                if fusion_dim % fusion_heads != 0:
                    raise ValueError(
                        f"fusion_dim ({fusion_dim}) must be divisible by fusion_heads ({fusion_heads})."
                    )
                self.attention = nn.MultiheadAttention(
                    embed_dim=fusion_dim,
                    num_heads=fusion_heads,
                    batch_first=True,
                )
            elif self.modal_interaction == "cross_attn_gate":
                if len(self.branch_dims) < 2:
                    raise ValueError(
                        "cross_attn_gate requires at least two enabled branches."
                    )
                self.output_dim = fusion_dim
                if fusion_dim % fusion_heads != 0:
                    raise ValueError(
                        f"fusion_dim ({fusion_dim}) must be divisible by fusion_heads ({fusion_heads})."
                    )
                self.cross_attention = nn.MultiheadAttention(
                    embed_dim=fusion_dim,
                    num_heads=fusion_heads,
                    batch_first=True,
                )
                self.gate = nn.Linear(2 * fusion_dim, fusion_dim)
            elif self.modal_interaction == "masked_pretrain":
                self.output_dim = fusion_dim
                self.fusion_encoder = nn.Sequential(
                    nn.Linear(fusion_dim * len(self.branch_dims), fusion_dim),
                    nn.ReLU(),
                    nn.Linear(fusion_dim, fusion_dim),
                )
                self.reconstruction_heads = nn.ModuleList(
                    [nn.Linear(fusion_dim, fusion_dim) for _ in self.branch_dims]
                )
            else:
                self.attention = None

    def _project_branches(self, branch_embeddings):
        if len(branch_embeddings) != len(self.branch_dims):
            raise ValueError(
                f"Expected {len(self.branch_dims)} branch embeddings, got {len(branch_embeddings)}."
            )

        if self.modal_interaction == "concat":
            return branch_embeddings

        return [
            projection(embedding)
            for projection, embedding in zip(self.projections, branch_embeddings)
        ]

    def forward(self, branch_embeddings):
        if self.modal_interaction == "concat":
            if len(branch_embeddings) != len(self.branch_dims):
                raise ValueError(
                    f"Expected {len(self.branch_dims)} branch embeddings, got {len(branch_embeddings)}."
                )
            return torch.cat(branch_embeddings, dim=1)

        projected = self._project_branches(branch_embeddings)

        if self.modal_interaction == "concat_attn":
            tokens = torch.stack(projected, dim=1)
            attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
            return attended.reshape(attended.shape[0], -1)

        if self.modal_interaction == "cross_attn_gate":
            # For two branches, use them directly as visual and time-series streams.
            # If more than two branches are enabled, we keep the first projected branch
            # as the visual stream and collapse all remaining projected branches into
            # a single auxiliary time-series-side embedding by mean pooling.
            if len(projected) > 2:
                e_img = projected[0]
                e_ts = torch.stack(projected[1:], dim=0).mean(dim=0)
            else:
                e_img, e_ts = projected[0], projected[1]

            if self.cross_attn_query == "ts":
                query, context = e_ts, e_img
                query_is_ts = True
            else:
                query, context = e_img, e_ts
                query_is_ts = False

            e_cross, _ = self.cross_attention(
                query=query.unsqueeze(1),
                key=context.unsqueeze(1),
                value=context.unsqueeze(1),
                need_weights=False,
            )
            e_cross = e_cross.squeeze(1)

            gate_input = torch.cat([e_ts, e_img], dim=1)
            gate = torch.sigmoid(self.gate(gate_input))
            self.last_gate_mean = gate.detach().mean()

            if query_is_ts:
                return gate * e_ts + (1.0 - gate) * e_cross

            return gate * e_img + (1.0 - gate) * e_cross

        if self.modal_interaction == "masked_pretrain":
            return self.fusion_encoder(torch.cat(projected, dim=1))

        # Placeholder for future fusion strategies.
        return torch.cat(projected, dim=1)

    def reconstruct_masked(self, branch_embeddings, mask_prob=0.3, mask_index=None):
        if self.modal_interaction != "masked_pretrain":
            raise ValueError("reconstruct_masked is only available for masked_pretrain.")

        projected = self._project_branches(branch_embeddings)

        if mask_index is None:
            mask_index = torch.randint(len(projected), (1,), device=projected[0].device).item()
        apply_mask = bool((torch.rand((), device=projected[0].device) < mask_prob).item())
        if not apply_mask:
            return None

        masked = []
        for idx, embedding in enumerate(projected):
            if idx == mask_index:
                masked.append(torch.zeros_like(embedding))
            else:
                masked.append(embedding)

        fused = self.fusion_encoder(torch.cat(masked, dim=1))
        recon = self.reconstruction_heads[mask_index](fused)
        target = projected[mask_index]

        return recon, target, mask_index


def _set_trainable(models, is_training):
    for model in models:
        if model is not None:
            model.train(is_training)


def _labels_to_indices(labels):
    classes = np.unique(labels)
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    indices = np.asarray([class_to_idx[label] for label in labels], dtype=np.int64)

    return indices, classes, class_to_idx


def _map_labels(labels, class_to_idx):
    unknown = sorted(set(labels) - set(class_to_idx))
    if unknown:
        raise ValueError(f"Test labels contain classes not present in training: {unknown}")

    return np.asarray([class_to_idx[label] for label in labels], dtype=np.int64)


def _balanced_class_weights(label_indices, num_classes, device):
    counts = np.bincount(label_indices, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError(
            f"Cannot balance classes with zero training samples: {counts.tolist()}"
        )

    weights = len(label_indices) / (num_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _forward_tivit_batch(model, batch, channels, device):
    if model.image_mode in {
        "multichannel_line_plot",
        "activity_graph",
        "activity_matrix",
    }:
        outputs = model(batch.to(device))
        return nn.functional.normalize(outputs, dim=-1)

    batch_embeds_dim = []
    for dim in range(channels):
        batch_dim = batch[:, dim, :].unsqueeze(-1).to(device)
        batch_embeds_dim.append(model(batch_dim))

    outputs = torch.cat(batch_embeds_dim, dim=1)

    return nn.functional.normalize(outputs, dim=-1)


def _forward_moment_batch(model, batch, channels, device):
    batch_embeds_dim = []
    for dim in range(channels):
        batch_dim = batch[:, dim, :].unsqueeze(1)
        batch_dim = resize_moment_input(batch_dim).to(device).float()
        outputs = model(x_enc=batch_dim).embeddings
        batch_embeds_dim.append(outputs)

    outputs = torch.cat(batch_embeds_dim, dim=1)

    return nn.functional.normalize(outputs, dim=-1)


def _forward_mantis_batch(model, batch, channels, device):
    batch_embeds_dim = []
    for dim in range(channels):
        batch_dim = batch[:, dim, :].unsqueeze(1)
        batch_dim = resize_mantis_input(batch_dim).to(device).float()
        outputs = model(batch_dim)
        batch_embeds_dim.append(outputs)

    # Mantis emits one 512-D vector per channel.  NeuroSigViT pools those
    # frozen channel representations into one sample-level temporal token.
    outputs = torch.stack(batch_embeds_dim, dim=1).mean(dim=1)

    return nn.functional.normalize(outputs, dim=-1)


def forward_feature_batch(
    batch,
    channels,
    device,
    vision_model_1=None,
    vision_model_2=None,
    mantis_model=None,
    moment_model=None,
):
    features = []

    if vision_model_1 is not None:
        features.append(_forward_tivit_batch(vision_model_1, batch, channels, device))
    if vision_model_2 is not None:
        features.append(_forward_tivit_batch(vision_model_2, batch, channels, device))
    if mantis_model is not None:
        features.append(_forward_mantis_batch(mantis_model, batch, channels, device))
    if moment_model is not None:
        features.append(_forward_moment_batch(moment_model, batch, channels, device))

    if not features:
        raise ValueError("At least one differentiable feature branch is required.")

    return features


@torch.no_grad()
def _extract_feature_split(loader, feature_models, channels, device):
    _set_trainable(feature_models.values(), False)
    branch_batches = None

    for (batch,) in tqdm(loader, desc="Extract model features", leave=False):
        features = forward_feature_batch(
            batch=batch,
            channels=channels,
            device=device,
            **feature_models,
        )
        if branch_batches is None:
            branch_batches = [[] for _ in features]
        for batches, feature in zip(branch_batches, features):
            batches.append(feature.detach().cpu())

    if branch_batches is None:
        raise ValueError("Cannot extract features from an empty split.")

    return [torch.cat(batches, dim=0) for batches in branch_batches]


def _feature_branch_names(feature_models):
    return [name for name, model in feature_models.items() if model is not None]


def _load_feature_cache(
    path,
    expected_labels,
    expected_branch_names,
    expected_signature,
):
    if not path.is_file():
        return None

    with np.load(path, allow_pickle=False) as cached:
        required_keys = {
            "schema_version",
            "cache_signature",
            "branch_names",
            "labels",
        }
        missing_keys = required_keys - set(cached.files)
        if missing_keys:
            raise ValueError(
                f"Feature cache is missing metadata {sorted(missing_keys)}: {path}"
            )
        schema_version = int(cached["schema_version"].item())
        if schema_version != FEATURE_CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Feature cache schema mismatch in {path}: expected "
                f"{FEATURE_CACHE_SCHEMA_VERSION}, got {schema_version}."
            )
        signature = str(cached["cache_signature"].item())
        if signature != (expected_signature or ""):
            raise ValueError(
                f"Feature cache model/configuration mismatch: {path}"
            )
        branch_names = cached["branch_names"].tolist()
        labels = cached["labels"]
        if branch_names != expected_branch_names:
            raise ValueError(
                f"Feature cache branch mismatch in {path}: "
                f"expected {expected_branch_names}, got {branch_names}."
            )
        if not np.array_equal(labels, np.asarray(expected_labels, dtype=np.int64)):
            raise ValueError(f"Feature cache labels do not match current split: {path}")
        branch_keys = [f"branch_{index}" for index in range(len(branch_names))]
        missing_branches = set(branch_keys) - set(cached.files)
        if missing_branches:
            raise ValueError(
                f"Feature cache is missing branches {sorted(missing_branches)}: {path}"
            )
        features = [torch.from_numpy(cached[key].copy()) for key in branch_keys]

    if any(len(feature) != len(labels) for feature in features):
        raise ValueError(f"Feature cache sample count mismatch: {path}")
    if any(feature.ndim != 2 for feature in features):
        raise ValueError(f"Feature cache branches must be two-dimensional: {path}")
    if any(not torch.isfinite(feature).all() for feature in features):
        raise ValueError(f"Feature cache contains non-finite values: {path}")
    print(f"Loaded model features: {path}")
    return features


def _save_feature_cache(path, labels, branch_names, features, cache_signature):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "schema_version": np.asarray(FEATURE_CACHE_SCHEMA_VERSION, dtype=np.int64),
        "cache_signature": np.asarray(cache_signature or ""),
        "labels": np.asarray(labels, dtype=np.int64),
        "branch_names": np.asarray(branch_names),
    }
    arrays.update(
        {
            f"branch_{index}": feature.numpy()
            for index, feature in enumerate(features)
        }
    )
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary_path, **arrays)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Saved model features: {path}")


def _get_feature_split(
    split,
    loader,
    labels,
    feature_models,
    channels,
    device,
    feature_cache_dir,
    feature_cache_signature,
):
    branch_names = _feature_branch_names(feature_models)
    cache_path = None
    if feature_cache_dir:
        cache_path = Path(feature_cache_dir) / f"{split}.npz"
        cached = _load_feature_cache(
            cache_path,
            labels,
            branch_names,
            feature_cache_signature,
        )
        if cached is not None:
            return cached

    features = _extract_feature_split(loader, feature_models, channels, device)
    if cache_path is not None:
        _save_feature_cache(
            cache_path,
            labels,
            branch_names,
            features,
            feature_cache_signature,
        )
    return features


def _build_feature_loader(features, labels, indices, batch_size, shuffle):
    dataset = TensorDataset(
        *features,
        torch.as_tensor(labels, dtype=torch.long),
    )
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def _infer_feature_dim(features):
    if not features:
        raise ValueError("At least one cached feature branch is required.")

    return [feature.shape[1] for feature in features]


def _run_epoch(mlp, fusion_module, loader, criterion, optimizer, device):
    _set_trainable([mlp, fusion_module], True)
    total_loss = 0.0
    total_samples = 0
    gate_means = []

    for *branch_features, labels in tqdm(loader, desc="Train MLP", leave=False):
        branch_features = [feature.to(device) for feature in branch_features]
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        features = fusion_module(branch_features)
        logits = mlp(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if getattr(fusion_module, "modal_interaction", None) == "cross_attn_gate":
            gate_mean = getattr(fusion_module, "last_gate_mean", None)
            if gate_mean is not None:
                gate_means.append(float(gate_mean.item()))

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    mean_gate = float(np.mean(gate_means)) if gate_means else None

    return total_loss / max(total_samples, 1), mean_gate


def _run_masked_pretrain_epoch(fusion_module, loader, optimizer, device, mask_prob):
    _set_trainable([fusion_module], True)
    total_loss = 0.0
    total_samples = 0

    for *branch_features, labels in tqdm(loader, desc="Pretrain Fusion", leave=False):
        branch_features = [feature.to(device) for feature in branch_features]
        optimizer.zero_grad(set_to_none=True)
        result = fusion_module.reconstruct_masked(
            branch_embeddings=branch_features,
            mask_prob=mask_prob,
        )
        if result is None:
            continue
        recon, target, _ = result
        loss = nn.functional.mse_loss(recon, target)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def _evaluate(mlp, fusion_module, loader, classes, device):
    _set_trainable([mlp, fusion_module], False)
    y_true = []
    y_pred = []
    y_score = []

    for *branch_features, labels in tqdm(loader, desc="Evaluate MLP", leave=False):
        branch_features = [feature.to(device) for feature in branch_features]
        features = fusion_module(branch_features)
        logits = mlp(features)
        probabilities = torch.softmax(logits, dim=1)

        y_true.append(labels.cpu().numpy())
        y_pred.append(probabilities.argmax(dim=1).cpu().numpy())
        y_score.append(probabilities.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_score = np.concatenate(y_score)

    return compute_metrics_from_predictions(y_true, y_pred, y_score, np.arange(len(classes)))


def train_mlp_classifier(
    train_loader,
    train_labels,
    test_loader,
    test_labels,
    channels,
    device,
    batch_size,
    random_seed,
    val_ratio,
    hidden_dim,
    num_layers,
    dropout,
    lr,
    weight_decay,
    epochs,
    early_stop_patience,
    modal_interaction,
    fusion_dim,
    fusion_heads,
    cross_attn_query,
    mask_prob,
    pretrain_epochs,
    class_weight="none",
    vision_model_1=None,
    vision_model_2=None,
    mantis_model=None,
    moment_model=None,
    val_loader=None,
    val_labels=None,
    feature_cache_dir=None,
    feature_cache_signature=None,
):
    has_fixed_validation = val_loader is not None or val_labels is not None
    if has_fixed_validation and (val_loader is None or val_labels is None):
        raise ValueError("val_loader and val_labels must be provided together")

    if has_fixed_validation:
        train_indices = list(range(len(train_loader.dataset)))
        val_indices = list(range(len(val_loader.dataset)))
    else:
        train_indices, val_indices = get_split(
            train_loader.dataset,
            frac=val_ratio,
            random_seed=random_seed,
        )

    train_label_indices, classes, class_to_idx = _labels_to_indices(train_labels)
    test_label_indices = _map_labels(test_labels, class_to_idx)

    feature_models = {
        "vision_model_1": vision_model_1,
        "vision_model_2": vision_model_2,
        "mantis_model": mantis_model,
        "moment_model": moment_model,
    }
    train_features = _get_feature_split(
        "train",
        train_loader,
        train_labels,
        feature_models,
        channels,
        device,
        feature_cache_dir,
        feature_cache_signature,
    )
    test_features = _get_feature_split(
        "test",
        test_loader,
        test_labels,
        feature_models,
        channels,
        device,
        feature_cache_dir,
        feature_cache_signature,
    )
    if has_fixed_validation:
        val_label_indices = _map_labels(val_labels, class_to_idx)
        val_features = _get_feature_split(
            "vali",
            val_loader,
            val_labels,
            feature_models,
            channels,
            device,
            feature_cache_dir,
            feature_cache_signature,
        )

    # Feature extraction and cache misses may consume RNG state. Reset here so
    # classifier initialization and training depend only on the outer seed.
    set_random_seed(random_seed)

    if has_fixed_validation:
        mlp_train_loader = _build_feature_loader(
            train_features,
            train_label_indices,
            train_indices,
            batch_size,
            shuffle=True,
        )
        mlp_val_loader = _build_feature_loader(
            val_features,
            val_label_indices,
            val_indices,
            batch_size,
            shuffle=False,
        )
    else:
        mlp_train_loader = _build_feature_loader(
            train_features,
            train_label_indices,
            train_indices,
            batch_size,
            shuffle=True,
        )
        mlp_val_loader = _build_feature_loader(
            train_features,
            train_label_indices,
            val_indices,
            batch_size,
            shuffle=False,
        )
    mlp_test_loader = _build_feature_loader(
        test_features,
        test_label_indices,
        list(range(len(test_label_indices))),
        batch_size,
        shuffle=False,
    )

    feature_dims = _infer_feature_dim(train_features)
    fusion_module = FusionModule(
        branch_dims=feature_dims,
        modal_interaction=modal_interaction,
        fusion_dim=fusion_dim,
        fusion_heads=fusion_heads,
        cross_attn_query=cross_attn_query,
    ).to(device)

    mlp = MLPClassifier(
        input_dim=fusion_module.output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=len(classes),
    ).to(device)

    parameters = list(mlp.parameters())
    parameters.extend(fusion_module.parameters())

    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if class_weight == "balanced":
        criterion_weights = _balanced_class_weights(
            train_label_indices[train_indices], len(classes), device
        )
        print(
            "MLP class weights: "
            + ", ".join(f"{weight:.4f}" for weight in criterion_weights.tolist())
        )
    elif class_weight == "none":
        criterion_weights = None
    else:
        raise ValueError(f"Unsupported MLP class weighting: {class_weight}")
    criterion = nn.CrossEntropyLoss(weight=criterion_weights)

    if modal_interaction == "masked_pretrain" and pretrain_epochs > 0:
        pretrain_optimizer = torch.optim.AdamW(
            fusion_module.parameters(), lr=lr, weight_decay=weight_decay
        )
        for epoch in range(pretrain_epochs):
            loss = _run_masked_pretrain_epoch(
                fusion_module=fusion_module,
                loader=mlp_train_loader,
                optimizer=pretrain_optimizer,
                device=device,
                mask_prob=mask_prob,
            )
            print(
                f"Fusion pretrain epoch {epoch + 1}/{pretrain_epochs} | mse_loss={loss:.4f}"
            )

    best_val_score = -float("inf")
    best_state = None
    epochs_without_improvement = 0
    val_metrics = None

    for epoch in range(epochs):
        loss, gate_mean = _run_epoch(
            mlp=mlp,
            fusion_module=fusion_module,
            loader=mlp_train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        if gate_mean is None:
            print(f"MLP epoch {epoch + 1}/{epochs} | loss={loss:.4f}")
        else:
            print(
                f"MLP epoch {epoch + 1}/{epochs} | loss={loss:.4f} | gate_mean={gate_mean:.4f}"
            )

        if early_stop_patience > 0:
            val_metrics = _evaluate(
                mlp, fusion_module, mlp_val_loader, classes, device
            )
            val_score = val_metrics["macro_f1"]
            if val_score > best_val_score:
                best_val_score = val_score
                epochs_without_improvement = 0
                best_state = {
                    "mlp": deepcopy(mlp.state_dict()),
                    "fusion_module": deepcopy(fusion_module.state_dict()),
                }
            else:
                epochs_without_improvement += 1

            print(
                f"MLP epoch {epoch + 1}/{epochs} | val_macro_f1={val_score:.4f} | best_val_macro_f1={best_val_score:.4f}"
            )
            if epochs_without_improvement >= early_stop_patience:
                print(
                    f"Early stopping after {epoch + 1} epochs | patience={early_stop_patience}"
                )
                break

    if best_state is not None:
        mlp.load_state_dict(best_state["mlp"])
        fusion_module.load_state_dict(best_state["fusion_module"])

    if val_metrics is None or best_state is not None:
        val_metrics = _evaluate(
            mlp, fusion_module, mlp_val_loader, classes, device
        )
    test_metrics = _evaluate(
        mlp, fusion_module, mlp_test_loader, classes, device
    )

    return val_metrics, test_metrics, train_indices, val_indices
