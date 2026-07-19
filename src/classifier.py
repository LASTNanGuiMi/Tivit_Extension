from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import src.utils as utils


def get_classifier(classifier_type, random_seed):
    if classifier_type == "logistic_regression":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, random_state=random_seed),
        )
    elif classifier_type == "nearest_centroid":
        clf = make_pipeline(StandardScaler(), NearestCentroid())
    elif classifier_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=random_seed
        )
    else:
        raise ValueError(f"Unsupported classifier type {classifier_type}.")

    return clf


def _nearest_centroid_scores(clf, X):
    if len(clf) > 1:
        transformed = clf[:-1].transform(X)
    else:
        transformed = X

    centroids = clf[-1].centroids_
    distances = ((transformed[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)

    return -distances


def get_class_scores(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)

    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
        if scores.ndim == 1:
            scores = scores[:, None]
        return scores

    if isinstance(clf[-1], NearestCentroid):
        return _nearest_centroid_scores(clf, X)

    raise ValueError(
        f"Classifier {type(clf[-1]).__name__} does not provide class scores."
    )


def _macro_curve_metric(y_true, y_score, classes, metric_fn):
    values = []

    for class_idx, class_label in enumerate(classes):
        y_binary = (y_true == class_label).astype(int)
        if y_binary.min() == y_binary.max():
            continue
        values.append(metric_fn(y_binary, y_score[:, class_idx]))

    if not values:
        return float("nan")

    return sum(values) / len(values)


def compute_metrics_from_predictions(y_true, y_pred, y_score, classes):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "macro_auroc": _macro_curve_metric(
            y_true, y_score, classes, roc_auc_score
        ),
        "macro_auprc": _macro_curve_metric(
            y_true, y_score, classes, average_precision_score
        ),
    }


def compute_metrics(clf, X, y):
    y_true = y.ravel()
    y_pred = clf.predict(X)
    classes = clf[-1].classes_
    y_score = get_class_scores(clf, X)

    return compute_metrics_from_predictions(y_true, y_pred, y_score, classes)


def train_classifier(
    train_embeds,
    train_labels,
    test_embeds,
    test_labels,
    classifier_type,
    random_seed,
    val_ratio,
):
    train, val = utils.get_split(
        train_embeds,
        frac=val_ratio,
        random_seed=random_seed,
    )

    clf = get_classifier(classifier_type, random_seed=random_seed)
    clf.fit(train_embeds[train], train_labels[train].ravel())

    val_metrics = compute_metrics(clf, train_embeds[val], train_labels[val])
    test_metrics = compute_metrics(clf, test_embeds, test_labels)

    return val_metrics, test_metrics, train, val
