"""UCI Sepsis Survival Minimal Clinical Records (primary cohort).

Chicco D, Jurman G. Survival prediction of patients with sepsis from age, sex,
and septic episode number alone. Scientific Reports 10, 17156 (2020).
Dataset DOI: https://doi.org/10.24432/C53C8N. The license is read from the
UCI dataset page by the data workflow and recorded in data/manifest_sepsis.json.

Rows are hospital admissions in Norway (2011 to 2012), not unique patients:
the file has no patient identifier, so repeat admissions cannot be removed.
The outcome is in hospital death.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from .uci_diabetes import sha256

URL = "https://archive.ics.uci.edu/static/public/827/sepsis+survival+minimal+clinical+records.zip"
PAGE = "https://archive.ics.uci.edu/dataset/827/sepsis+survival+minimal+clinical+records"
CITATION = ("Chicco, D., & Jurman, G. (2020). Sepsis Survival Minimal Clinical Records [Dataset]. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C53C8N")
COLUMNS = {"age_years", "sex_0male_1female", "episode_number", "hospital_outcome_1alive_0dead"}


def _primary_member(z: zipfile.ZipFile) -> str:
    hits = [m for m in z.namelist() if m.lower().endswith("primary_cohort.csv")]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one primary cohort file, found {hits}; members {z.namelist()}")
    return hits[0]


def build_cohort(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"columns missing from the sepsis file: {sorted(missing)}")
    data = data.dropna(subset=sorted(COLUMNS))
    out = pd.DataFrame({
        "outcome": (data["hospital_outcome_1alive_0dead"] == 0).astype(int),  # death
        "age_years": data["age_years"].astype(float),
        "female": data["sex_0male_1female"].astype(int),
        "episode_number": data["episode_number"].astype(float),
    }).reset_index(drop=True)
    counts = {"admissions": int(len(out)), "deaths": int(out["outcome"].sum()),
              "mortality_rate": float(out["outcome"].mean())}
    return out, counts


def feature_columns(cohort: pd.DataFrame) -> list[str]:
    return ["age_years", "female", "episode_number"]


def load(zip_path: str | Path, manifest_path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    zip_path = Path(zip_path)
    digest = sha256(zip_path)
    if manifest_path and Path(manifest_path).exists():
        recorded = json.loads(Path(manifest_path).read_text())["sha256"]
        if recorded != digest:
            raise ValueError(f"archive checksum {digest} does not match manifest {recorded}")
    with zipfile.ZipFile(zip_path) as z:
        member = _primary_member(z)
        data = pd.read_csv(z.open(member))
    cohort, counts = build_cohort(data)
    counts.update({"sha256": digest, "source": URL, "citation": CITATION, "file": member})
    return cohort, counts
