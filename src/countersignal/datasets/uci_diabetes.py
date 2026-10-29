"""UCI Diabetes 130-US Hospitals for Years 1999-2008.

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals
for Years 1999-2008 [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5230J. Licensed CC BY 4.0.

Cohort rules follow the dataset's introductory paper (Strack et al., 2014,
BioMed Research International, 781670): keep each patient's first encounter
and drop encounters that ended in death or discharge to hospice, since those
patients cannot be readmitted. "First" means lowest encounter_id; the public
file carries no admission dates.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
CITATION = ("Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for "
            "Years 1999-2008 [Dataset]. UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C5230J")

NUMERIC = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
           "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _member(z: zipfile.ZipFile, name: str) -> str:
    """Find a file in the archive by base name, ignoring case and folders."""
    for info in z.namelist():
        if Path(info).name.lower() == name.lower():
            return info
    raise FileNotFoundError(f"{name} not found in archive; members: {z.namelist()}")


def read_archive(zip_path: Path) -> tuple[pd.DataFrame, str]:
    with zipfile.ZipFile(zip_path) as z:
        data = pd.read_csv(z.open(_member(z, "diabetic_data.csv")), na_values="?",
                           low_memory=False)
        mapping = z.read(_member(z, "IDs_mapping.csv")).decode("utf-8", errors="replace")
    return data, mapping


def mapping_table(mapping_text: str, header: str) -> dict[int, str]:
    """Parse one block of IDs_mapping.csv.

    Blocks are separated by lines that are blank or hold only commas.
    """
    blocks, current = [], []
    for line in mapping_text.replace("\r", "").split("\n"):
        if line.replace(",", "").strip():
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    for lines in blocks:
        if lines[0].split(",")[0].strip() == header:
            frame = pd.read_csv(io.StringIO("\n".join(lines)))
            return {int(k): str(v) for k, v in zip(frame.iloc[:, 0], frame.iloc[:, 1], strict=True)
                    if str(k).strip().isdigit()}
    raise KeyError(f"mapping block {header} not found")


def _age_midpoint(series: pd.Series) -> pd.Series:
    lower = series.str.extract(r"\[(\d+)-")[0].astype(float)
    return lower + 5


def _diag_group(code) -> str:
    """Primary diagnosis groups used by Strack et al. (2014), Table 2."""
    if pd.isna(code):
        return "missing"
    s = str(code)
    if s.startswith(("V", "E")):
        return "other"
    try:
        v = float(s)
    except ValueError:
        return "other"
    if 390 <= v <= 459 or int(v) == 785:
        return "circulatory"
    if 460 <= v <= 519 or int(v) == 786:
        return "respiratory"
    if 520 <= v <= 579 or int(v) == 787:
        return "digestive"
    if int(v) == 250:
        return "diabetes"
    if 800 <= v <= 999:
        return "injury"
    if 710 <= v <= 739:
        return "musculoskeletal"
    if 580 <= v <= 629 or int(v) == 788:
        return "genitourinary"
    if 140 <= v <= 239:
        return "neoplasms"
    return "other"


def build_cohort(data: pd.DataFrame, mapping_text: str) -> tuple[pd.DataFrame, dict]:
    """Apply cohort rules and engineer model features. Returns (cohort, counts)."""
    counts = {"encounters": int(len(data)), "patients": int(data["patient_nbr"].nunique())}
    dispo = mapping_table(mapping_text, "discharge_disposition_id")
    ended = sorted(k for k, v in dispo.items() if ("expired" in v.lower() or "hospice" in v.lower()))
    counts["excluded_disposition_ids"] = ended
    counts["excluded_disposition_labels"] = [dispo[k] for k in ended]
    data = data.sort_values("encounter_id")
    data = data.drop_duplicates("patient_nbr", keep="first")
    counts["first_encounters"] = int(len(data))
    data = data[~data["discharge_disposition_id"].isin(ended)]
    counts["cohort"] = int(len(data))

    admission_type = mapping_table(mapping_text, "admission_type_id")
    out = pd.DataFrame(index=data.index)
    out["encounter_id"] = data["encounter_id"].to_numpy()
    out["outcome"] = (data["readmitted"] == "<30").astype(int)
    for col in NUMERIC:
        out[col] = data[col].astype(float)
    out["age"] = _age_midpoint(data["age"])
    out["female"] = (data["gender"] == "Female").astype(int)
    out["emergency_admission"] = data["admission_type_id"].map(
        lambda i: int("emergency" in admission_type.get(int(i), "").lower()))
    home = [k for k, v in dispo.items() if v.strip().lower() == "discharged to home"]
    counts["home_disposition_ids"] = home
    out["discharged_home"] = data["discharge_disposition_id"].isin(home).astype(int)
    # pandas reads the file's "None" (test not done) as missing
    out["a1c_tested"] = data["A1Cresult"].notna().astype(int)
    out["med_change"] = (data["change"] == "Ch").astype(int)
    out["diabetes_med"] = (data["diabetesMed"] == "Yes").astype(int)
    out["insulin"] = (data["insulin"] != "No").astype(int)
    groups = data["diag_1"].map(_diag_group)
    for g in ["circulatory", "respiratory", "digestive", "diabetes", "injury",
              "musculoskeletal", "genitourinary", "neoplasms"]:
        out[f"dx_{g}"] = (groups == g).astype(int)
    counts["readmitted_30d"] = int(out["outcome"].sum())
    counts["readmission_rate"] = float(out["outcome"].mean())
    return out.reset_index(drop=True), counts


def feature_columns(cohort: pd.DataFrame) -> list[str]:
    return [c for c in cohort.columns if c not in ("encounter_id", "outcome")]


def load(zip_path: str | Path, manifest_path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Read the archive, verify it against a recorded checksum if one exists, build the cohort."""
    zip_path = Path(zip_path)
    digest = sha256(zip_path)
    if manifest_path and Path(manifest_path).exists():
        recorded = json.loads(Path(manifest_path).read_text())["sha256"]
        if recorded != digest:
            raise ValueError(f"archive checksum {digest} does not match manifest {recorded}")
    data, mapping = read_archive(zip_path)
    cohort, counts = build_cohort(data, mapping)
    counts["sha256"] = digest
    counts["source"] = URL
    counts["citation"] = CITATION
    if not np.isin(cohort["outcome"].unique(), [0, 1]).all():
        raise ValueError("outcome must be binary")
    return cohort, counts
