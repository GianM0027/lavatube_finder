"""
Illumination and viewing geometry for the source HiRISE products.

Why this exists
---------------
What separates the DeepLandforms classes optically is shadow depth and whether a
floor is visible: Type-1 has an "almost nonvisible bottom", Type-2 a "visible
convex bottom", Type-4 a "visible flat bottom" (Nodjoumi et al. 2023, Fig. 3).
All three are functions of how the sun was standing when the image was taken,
and the same paper warns that Type-1b, Type-2a and Type-4 become hard to tell
apart "especially when the solar incidence angle is low".

So solar incidence is not a nuisance variable here, it is a candidate shortcut:
if it predicts the class on its own, a model trained on the pixels may be
reading illumination rather than morphology. It cannot be checked without
pulling it from the archive, because DeepLandforms tiles are derived products
and carry none of the original observation metadata.

Source: the PDS Orbital Data Explorer REST service at Washington University,
queried once per HiRISE product id and cached to CSV.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Iterable, Optional

import pandas as pd

__all__ = ["ODE_FIELDS", "fetch_product_geometry", "product_geometry_table"]

ODE_URL = "https://oderest.rsl.wustl.edu/live2/"

#: Fields kept from an ODE product record. ``Solar_time`` is Mars local solar
#: time in decimal hours; ``Solar_longitude`` (Ls) is the season.
ODE_FIELDS = (
    "Incidence_angle",
    "Emission_angle",
    "Phase_angle",
    "Solar_longitude",
    "Solar_time",
    "Observation_time",
    "Center_latitude",
    "Center_longitude",
)


def fetch_product_geometry(
    product_id: str, session=None, timeout: int = 40
) -> Optional[Dict[str, float]]:
    """
    Query ODE for one HiRISE product.

    :param product_id: Observation id such as ``ESP_011386_2065``. The ``_RED``
        suffix is added if absent, since that is the product DeepLandforms cut
        its tiles from.
    :return: Mapping of :data:`ODE_FIELDS` to values, or ``None`` if the product
        is not found.
    """
    import requests

    if not product_id.upper().endswith(("_RED", "_COLOR")):
        product_id = f"{product_id}_RED"

    session = session or requests.Session()
    response = session.get(
        ODE_URL,
        params={
            "target": "mars",
            "query": "product",
            "results": "m",
            "output": "JSON",
            "pdsid": product_id,
        },
        timeout=timeout,
    )
    response.raise_for_status()

    products = (
        response.json()
        .get("ODEResults", {})
        .get("Products", {})
        .get("Product")
    )
    if not products:
        return None

    record = products[0] if isinstance(products, list) else products
    values: Dict[str, float] = {}
    for field in ODE_FIELDS:
        raw = record.get(field)
        try:
            values[field] = float(raw)
        except (TypeError, ValueError):
            values[field] = raw
    return values


def product_geometry_table(
    product_ids: Iterable[str],
    cache_path: str = "data/optical/final_data/hirise_geometry.csv",
    pause: float = 0.2,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Geometry for a set of HiRISE products, fetched once and cached.

    Products already present in the cache are not re-queried, so this can be
    re-run after adding data without hitting the archive again.

    :return: One row per product id, indexed by ``product``.
    """
    import requests
    from tqdm.auto import tqdm

    wanted = sorted({p for p in product_ids if isinstance(p, str) and p})

    cached = pd.DataFrame()
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        if "product" in cached.columns:
            cached = cached.set_index("product")

    missing = [p for p in wanted if p not in cached.index]
    if verbose:
        print(f"{len(wanted)} products, {len(wanted) - len(missing)} cached, "
              f"{len(missing)} to fetch")

    if missing:
        session = requests.Session()
        rows = []
        for product in tqdm(missing, desc="ODE geometry", disable=not verbose):
            try:
                values = fetch_product_geometry(product, session=session)
            except Exception as exc:
                print(f"  {product}: {type(exc).__name__} {exc}")
                values = None
            rows.append({"product": product, **(values or {})})
            time.sleep(pause)

        fetched = pd.DataFrame(rows).set_index("product")
        cached = pd.concat([cached, fetched]) if len(cached) else fetched

        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        cached.to_csv(cache_path)
        if verbose:
            print(f"cache written to {cache_path}")

    result = cached.loc[[p for p in wanted if p in cached.index]]
    if verbose:
        found = result["Incidence_angle"].notna().sum()
        print(f"{found}/{len(wanted)} products carry an incidence angle")
    return result
