from pathlib import Path
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

root = Path(__file__).parent.parent.absolute()

subjects = [f"sub-{str(nr).zfill(3)}" for nr in range(12, 17)]
df = pd.DataFrame(columns=["subjectID", "frequency", "is_rare", "is_tone", "response"])
for isub, sub in enumerate(subjects):
    sub_folder = root / "raw" / sub / "beh"
    for block in sub_folder.glob("*block*"):
        seq = json.load(open(block))
        is_rare = np.asarray(seq["trials"]) - 1
        frequency = np.asarray(seq["conditions"])[is_rare]
        is_tone = np.asarray([d[0] for d in seq["data"]])
        response = np.asarray([d[1] for d in seq["data"]])
        subjectID = np.repeat(isub, len(response))
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "subjectID": subjectID,
                        "frequency": frequency,
                        "is_tone": is_tone,
                        "is_rare": is_rare,
                        "response": response,
                    }
                ),
            ],
            ignore_index=True,
        )
df.to_csv(root / "results" / "one_interval_beh_pilot.csv")
