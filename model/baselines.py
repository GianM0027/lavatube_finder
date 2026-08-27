"""
Baselines that deliberately cannot see morphology.

A CNN score means nothing on its own. What it has to beat is not chance and not
the majority class, but *whatever is obtainable from this data set without
looking at shape* -- because on the previous version of this task that number
was 97.9%, and every point of it came from the data pipeline rather than from
the model.

Three baselines, in increasing order of what they are allowed to know.

``majority``
    Predict the most common class in the training fold. The floor.

``histogram``
    Statistics of the crop's intensity histogram: moments, percentiles, dark and
    bright fractions, entropy. **Every one of these is invariant to permuting
    the pixels of the image.** Shuffle a crop and its morphology is destroyed
    while these features are untouched, so anything this baseline achieves is
    achievable with literally zero shape information. That makes it the honest
    bar for the optical CNN.

``texture``
    ``histogram`` plus gradient-magnitude statistics. No longer
    permutation-invariant -- a gradient is a local spatial relation -- but still
    carries no global shape. Reported separately so the two are not conflated.

``illumination``
    Solar incidence, emission and phase angle, season and local solar time of
    the source HiRISE product. These are constant within a product, so this set
    is only interpretable under **product-level** grouping; under landform-level
    grouping it degenerates into a product-identity lookup. Running it at both
    levels is the cheapest available demonstration that the grouping matters.

``availability``
    Thermal *coverage bookkeeping* and no temperature: how many THEMIS frames a
    site received, at which local solar times, how much of each window was
    valid. THEMIS coverage depends on location and location correlates with
    class, so missingness carries class information by itself. This is the
    control the multimodal claim has to clear.

``metadata``
    Acquisition and geometry columns only: resolution, tile dimensions,
    annotated diameter, latitude, longitude. **This is not a legitimate
    detector.** ``diameter_m`` comes from the annotation itself, so a real
    system would not have it before detecting anything. It is a diagnostic: it
    measures how much of the label is carried by things that are not the image.
    Expect it to score well, because size genuinely separates these classes
    (median annotated diameter 155 m for Type-1 against 779 m for Type-2), and
    report it as a disclosure rather than as a result.

All baselines are evaluated with the same group-aware, stratified k-fold used
for the neural models, so the numbers are directly comparable.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "histogram_features",
    "texture_features",
    "metadata_features",
    "illumination_features",
    "availability_features",
    "extract_image_features",
    "evaluate_baseline",
    "run_baselines",
    "FEATURE_SETS",
]

#: Percentiles kept from each crop's intensity histogram.
_PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def histogram_features(image: np.ndarray) -> Dict[str, float]:
    """
    Permutation-invariant statistics of one crop.

    Every value here is a function of the multiset of pixel values, so it
    survives shuffling the image and carries no morphology whatsoever.

    :param image: 2-D array, any dtype. Values are treated as-is.
    """
    flat = np.asarray(image, dtype=np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return {}

    mean = float(flat.mean())
    std = float(flat.std())
    centred = flat - mean

    features = {
        "mean": mean,
        "std": std,
        "skew": float((centred ** 3).mean() / std ** 3) if std > 0 else 0.0,
        "kurtosis": float((centred ** 4).mean() / std ** 4) if std > 0 else 0.0,
    }

    values = np.percentile(flat, _PERCENTILES)
    features.update({f"p{p:02d}": float(v) for p, v in zip(_PERCENTILES, values)})

    # Spread measures that do not assume a symmetric distribution.
    features["iqr"] = features["p75"] - features["p25"]
    features["range_90"] = features["p95"] - features["p05"]

    # Shadow and highlight fractions -- the crude version of "is there a hole".
    scale = flat.max() if flat.max() > 1.0 else 1.0
    normalised = flat / scale
    for threshold in (0.1, 0.2, 0.3):
        features[f"frac_below_{threshold:.1f}"] = float((normalised < threshold).mean())
    for threshold in (0.7, 0.8, 0.9):
        features[f"frac_above_{threshold:.1f}"] = float((normalised > threshold).mean())

    # Histogram entropy: how much intensity variety the crop holds.
    counts, _ = np.histogram(normalised, bins=32, range=(0.0, 1.0))
    probabilities = counts / max(counts.sum(), 1)
    nonzero = probabilities[probabilities > 0]
    features["entropy"] = float(-(nonzero * np.log2(nonzero)).sum())

    return features


def texture_features(image: np.ndarray) -> Dict[str, float]:
    """
    ``histogram_features`` plus gradient-magnitude statistics.

    Gradients are a local spatial relation, so this is no longer
    permutation-invariant. It still describes no global shape.
    """
    features = histogram_features(image)

    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 2:
        return features

    dy, dx = np.gradient(array)
    magnitude = np.hypot(dx, dy)

    features["grad_mean"] = float(magnitude.mean())
    features["grad_std"] = float(magnitude.std())
    for percentile in (50, 90, 99):
        features[f"grad_p{percentile}"] = float(np.percentile(magnitude, percentile))

    return features


#: Columns of the annotation table that describe acquisition or geometry rather
#: than appearance. ``diameter_m`` and ``box_px`` are oracle features -- see the
#: module docstring.
METADATA_COLUMNS = (
    "resolution_mpp", "tile_width", "tile_height",
    "box_px", "diameter_m", "lat", "lon_east",
)


def metadata_features(annotations: pd.DataFrame) -> pd.DataFrame:
    """Acquisition and geometry columns, as a feature matrix."""
    available = [c for c in METADATA_COLUMNS if c in annotations.columns]
    frame = annotations[available].astype(float).copy()
    if {"tile_width", "tile_height"}.issubset(frame.columns):
        frame["tile_area_px"] = frame["tile_width"] * frame["tile_height"]
    return frame


FEATURE_SETS: Dict[str, Callable[[np.ndarray], Dict[str, float]]] = {
    "histogram": histogram_features,
    "texture": texture_features,
}


def extract_image_features(
    dataset,
    extractor: Callable[[np.ndarray], Dict[str, float]] = texture_features,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run a feature extractor over every crop a dataset produces.

    The dataset should be built with ``augment=False`` so the crops are the same
    ones the neural model sees at validation time. Anything the baseline is
    given must come from that same image, or the comparison is meaningless.

    :return: One row per sample, columns named by the extractor's keys.
    """
    from tqdm.auto import tqdm

    rows = []
    iterator = range(len(dataset))
    if verbose:
        iterator = tqdm(iterator, desc=f"Features ({extractor.__name__})")

    for idx in iterator:
        sample = dataset[idx]
        image = sample[0]
        array = np.asarray(image).squeeze()
        rows.append(extractor(array))

    return pd.DataFrame(rows).fillna(0.0)


def evaluate_baseline(
    features: Optional[pd.DataFrame],
    labels: Sequence[int],
    groups: Sequence[str],
    model: str = "gbm",
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Group-aware, stratified k-fold evaluation of one feature set.

    :param features: Feature matrix, or ``None`` for the majority-class baseline.
    :param model: ``"gbm"`` (histogram gradient boosting), ``"logistic"``
        (standardised linear), or ``"majority"``.
    :return: One row per fold, with accuracy and macro precision/recall/F1.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray(labels)
    groups = np.asarray(groups)

    if model == "majority" or features is None:
        matrix = np.zeros((len(labels), 1))
    else:
        matrix = features.to_numpy(dtype=float)

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )

    rows = []
    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(matrix, labels, groups), start=1
    ):
        if model == "majority" or features is None:
            estimator = DummyClassifier(strategy="most_frequent")
        elif model == "logistic":
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, random_state=seed),
            )
        else:
            estimator = HistGradientBoostingClassifier(random_state=seed)

        estimator.fit(matrix[train_idx], labels[train_idx])
        predictions = estimator.predict(matrix[val_idx])

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels[val_idx], predictions, average="macro", zero_division=0
        )
        rows.append({
            "fold": fold,
            "acc": float((predictions == labels[val_idx]).mean()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    return pd.DataFrame(rows).set_index("fold")


def availability_features(
    annotation_sites: pd.DataFrame, manifest_path: str
) -> Optional[pd.DataFrame]:
    """
    Thermal *coverage bookkeeping* for each annotation -- and no temperature.

    How many THEMIS frames a site received, at which local solar times, and how
    much of each window was valid. Not one Kelvin value is included.

    This is the control experiment for the multimodal claim. THEMIS coverage
    depends on where a site is, and where a site is correlates with what it is,
    so the *pattern of missingness* carries class information on its own.
    Measured on the superseded grid-based run, a classifier given nothing but
    the frame count scored 60.5% on binary against a 54.0% majority, and 51.3%
    on 3-class against 46.0% -- roughly six points from a single integer.

    Use it as a floor: if a multimodal model beats optical-only by no more than
    this baseline beats the majority class, the gain is coverage bookkeeping and
    not thermal physics.

    :param annotation_sites: ``image_name`` / ``site_id`` rows from
        ``data.thermal.thermal_sites.landform_sites``.
    :param manifest_path: ``window_manifest.json`` written by the thermal
        pipeline.
    :return: one row per annotation, or ``None`` if the manifest does not
        describe these sites (for instance a stale one from an earlier run).
    """
    import json
    import os

    if not os.path.exists(manifest_path):
        return None

    with open(manifest_path) as handle:
        manifest = json.load(handle)

    per_site = {}
    for entry in manifest:
        frames = entry.get("frames", [])
        hours = [f.get("mars_lmst_decimal_hours") for f in frames]
        hours = [h for h in hours if h is not None]
        valid = [f.get("valid_fraction", 0.0) for f in frames]
        kelvin = [f.get("kelvin_median") for f in frames]

        per_site[entry["site_id"]] = {
            "n_frames": len(frames),
            "n_usable": sum(1 for k in kelvin if k is not None),
            "lmst_min": min(hours) if hours else -1.0,
            "lmst_max": max(hours) if hours else -1.0,
            "lmst_span": (max(hours) - min(hours)) if len(hours) > 1 else 0.0,
            "valid_mean": float(np.mean(valid)) if valid else 0.0,
            "valid_min": float(np.min(valid)) if valid else 0.0,
        }

    overlap = set(per_site) & set(annotation_sites["site_id"].dropna())
    if not overlap:
        # A manifest from a different site scheme. Joining it would produce
        # confident nonsense, so refuse rather than guess.
        return None

    empty = {
        "n_frames": 0, "n_usable": 0, "lmst_min": -1.0, "lmst_max": -1.0,
        "lmst_span": 0.0, "valid_mean": 0.0, "valid_min": 0.0,
    }
    rows = [per_site.get(site, empty) for site in annotation_sites["site_id"]]
    frame = pd.DataFrame(rows, index=annotation_sites.index)
    frame["has_thermal"] = (frame["n_usable"] > 0).astype(int)
    return frame


def illumination_features(
    annotations: pd.DataFrame, geometry: pd.DataFrame
) -> pd.DataFrame:
    """
    Per-annotation solar and viewing geometry, joined from the product table.

    See ``data/optical/hirise_metadata.py`` for why this is worth testing on its
    own: illumination drives shadow depth, which is what the class definitions
    turn on.

    Note that these values are constant within a HiRISE product, so this feature
    set is *only* interpretable under product-level grouping. Under
    landform-level grouping it becomes a product-identity lookup and will score
    far higher than the physics warrants -- which is itself worth showing.
    """
    columns = [
        "Incidence_angle", "Emission_angle", "Phase_angle",
        "Solar_longitude", "Solar_time",
    ]
    available = [c for c in columns if c in geometry.columns]
    joined = geometry.reindex(annotations["product"])[available]
    return joined.reset_index(drop=True).astype(float).fillna(0.0)


def run_baselines(
    annotations: pd.DataFrame,
    feature_sets: Dict[str, pd.DataFrame],
    groups: Sequence[str],
    labels: Optional[Sequence[int]] = None,
    model: str = "gbm",
    n_splits: int = 5,
    seed: int = 42,
    include_metadata: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate every baseline against one grouping and one label set.

    :param feature_sets: Mapping of name -> feature matrix, e.g. the output of
        :func:`extract_image_features` for ``histogram`` and ``texture``, and
        :func:`illumination_features` for the acquisition geometry.
    :param labels: Defaults to ``annotations["category_id"]``. Pass a binary
        vector to score the Type-1-versus-rest task instead.
    :param include_metadata: Append the metadata oracle. See the module
        docstring for why it is a disclosure rather than a result.
    :return: One row per baseline, with mean and standard deviation over folds.
    """
    if labels is None:
        labels = annotations["category_id"].to_numpy()

    candidates: List[tuple] = [("majority", None, "majority")]
    for name, matrix in feature_sets.items():
        candidates.append((name, matrix, model))
    if include_metadata:
        candidates.append(
            ("metadata (oracle)", metadata_features(annotations), model)
        )

    rows = []
    for name, matrix, estimator in candidates:
        per_fold = evaluate_baseline(
            matrix, labels, groups, model=estimator, n_splits=n_splits, seed=seed
        )
        row = {"baseline": name, "n_features": 0 if matrix is None else matrix.shape[1]}
        for metric in ("acc", "f1"):
            row[f"{metric}_mean"] = per_fold[metric].mean()
            row[f"{metric}_std"] = per_fold[metric].std()
        rows.append(row)
        if verbose:
            print(f"  {name:20s} acc {row['acc_mean'] * 100:5.1f} "
                  f"+/- {row['acc_std'] * 100:4.1f}   "
                  f"F1 {row['f1_mean'] * 100:5.1f} +/- {row['f1_std'] * 100:4.1f}")

    return pd.DataFrame(rows).set_index("baseline")
