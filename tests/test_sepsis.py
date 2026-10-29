import io
import zipfile

import pandas as pd

from countersignal.datasets import uci_sepsis as S


def test_sepsis_primary_cohort(tmp_path):
    frame = pd.DataFrame({"age_years": [70, 45, 88], "sex_0male_1female": [0, 1, 1],
                          "episode_number": [1, 2, 1], "hospital_outcome_1alive_0dead": [1, 0, 1]})
    path = tmp_path / "s.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name in ("s41598-020-73558-3_sepsis_survival_primary_cohort.csv",
                     "s41598-020-73558-3_sepsis_survival_study_cohort.csv"):
            buf = io.StringIO()
            frame.to_csv(buf, index=False)
            z.writestr(name, buf.getvalue())
    cohort, counts = S.load(path)
    assert counts["admissions"] == 3 and counts["deaths"] == 1
    assert cohort["outcome"].tolist() == [0, 1, 0]  # outcome is death
    assert counts["file"].endswith("primary_cohort.csv")
