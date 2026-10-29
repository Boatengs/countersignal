import io
import zipfile

import pandas as pd

from countersignal.datasets import uci_diabetes as U

MAPPING = """admission_type_id,description
1,Emergency
2,Urgent
3,Elective
,
discharge_disposition_id,description
1,Discharged to home
11,Expired
13,Hospice / home
14,Hospice / medical facility
3,Discharged/transferred to SNF
,
admission_source_id,description
7, Emergency Room
"""


def _zip(tmp_path, rows):
    cols = ["encounter_id", "patient_nbr", "race", "gender", "age", "admission_type_id",
            "discharge_disposition_id", "admission_source_id", *U.NUMERIC, "diag_1",
            "A1Cresult", "change", "diabetesMed", "insulin", "readmitted"]
    frame = pd.DataFrame(rows, columns=cols)
    path = tmp_path / "uci.zip"
    with zipfile.ZipFile(path, "w") as z:
        buf = io.StringIO()
        frame.to_csv(buf, index=False)
        z.writestr("diabetic_data.csv", buf.getvalue())
        z.writestr("IDS_mapping.csv", MAPPING)  # the real archive uses this capitalisation
    return path


def test_cohort_rules(tmp_path):
    base = [1, 2, 3, 4, 5, 0, 1, 7]
    rows = [
        [10, 100, "Caucasian", "Female", "[70-80)", 1, 1, 7, *base, "428", "None", "Ch", "Yes", "Up", "<30"],
        [11, 100, "Caucasian", "Female", "[70-80)", 1, 1, 7, *base, "250.83", "?", "No", "No", "No", "NO"],
        [12, 200, "?", "Male", "[50-60)", 3, 11, 7, *base, "V57", ">7", "No", "Yes", "No", "NO"],
        [13, 300, "AfricanAmerican", "Male", "[40-50)", 2, 3, 7, *base,
         "786", ">8", "No", "Yes", "Steady", ">30"],
    ]
    cohort, counts = U.load(_zip(tmp_path, rows))
    assert counts["encounters"] == 4
    assert counts["first_encounters"] == 3  # patient 100 keeps encounter 10 only
    assert counts["excluded_disposition_ids"] == [11, 13, 14]
    assert counts["cohort"] == 2  # patient 200 expired
    assert cohort["outcome"].tolist() == [1, 0]  # ">30" is not a 30 day readmission
    assert cohort["age"].tolist() == [75.0, 45.0]
    assert cohort["dx_circulatory"].tolist() == [1, 0]
    assert cohort["dx_respiratory"].tolist() == [0, 1]
    assert cohort["discharged_home"].tolist() == [1, 0]
    assert cohort["emergency_admission"].tolist() == [1, 0]
    assert cohort["a1c_tested"].tolist() == [0, 1]
