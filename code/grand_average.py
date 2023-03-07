from pathlib import Path
import pandas as pd
import numpy as np
from mne import read_epochs, grand_average, write_evokeds

root = Path(__file__).parent.parent.absolute()
subjects = list((root / "preprocessed").glob("sub*"))

vis1 = []
vis2 = []
aud1_std_hit = []
aud1_odd_hit = []
aud1_std_mis = []
aud1_odd_mis = []
aud2_std_hit = []
aud2_odd_hit = []
aud2_std_mis = []
aud2_odd_mis = []
resp_odd_hit = []
resp_odd_mis = []
resp_std_hit = []
resp_std_mis = []

for subject in subjects:
    epochs = read_epochs(subject / f"{subject.name}-epo.fif")
    df = pd.read_csv(subject / f"{subject.name}-interval_detection.csv")
    vis1.append(epochs["vis1"][df.target == 2].average())
    vis2.append(epochs["vis2"][df.target == 1].average())
    epochs = epochs[["aud", "resp"]]
    aud1_std_hit.append(
        epochs["aud"][
            np.logical_and(
                np.logical_and(df.target == 1, df.frequency == 1000),
                df.response == df.target,
            )
        ].average()
    )
    aud1_odd_hit.append(
        epochs["aud"][
            np.logical_and(
                np.logical_and(df.target == 1, df.frequency == 1200),
                df.response == df.target,
            )
        ].average()
    )
    aud1_std_mis.append(
        epochs["aud"][
            np.logical_and(
                np.logical_and(df.target == 1, df.frequency == 1000),
                df.response != df.target,
            )
        ].average()
    )
    aud1_odd_mis.append(
        epochs["aud"][
            np.logical_and(
                np.logical_and(df.target == 1, df.frequency == 1200),
                df.response != df.target,
            )
        ].average()
    )
    aud2_std_hit.append(
        epochs["aud"][
            np.logical_and(
                np.logical_and(df.target == 2, df.frequency == 1000),
                df.response == df.target,
            )
        ].average()
    )
    aud2_odd_hit.append(
        epochs["aud"][
            np.logical_and(
                np.logical_and(df.target == 2, df.frequency == 1200),
                df.response == df.target,
            )
        ].average()
    )
    aud2_std_mis.append(
        epochs["aud"][
            np.logical_and(
                np.logical_and(df.target == 2, df.frequency == 1000),
                df.response != df.target,
            )
        ].average()
    )
    aud2_odd_mis.append(
        epochs["aud"][
            np.logical_and(
                np.logical_and(df.target == 2, df.frequency == 1200),
                df.response != df.target,
            )
        ].average()
    )
    df = df[: len(epochs["resp"])]
    resp_std_hit.append(
        epochs["resp"][
            np.logical_and(df.frequency == 1000, df.response == df.target)
        ].average()
    )
    resp_odd_hit.append(
        epochs["resp"][
            np.logical_and(df.frequency == 1200, df.response == df.target)
        ].average()
    )
    resp_std_mis.append(
        epochs["resp"][
            np.logical_and(df.frequency == 1000, df.response != df.target)
        ].average()
    )
    resp_odd_mis.append(
        epochs["resp"][
            np.logical_and(df.frequency == 1200, df.response != df.target)
        ].average()
    )
    del epochs


vis1 = grand_average(vis1)
vis1.comment = "vis1"
vis2 = grand_average(vis2)
vis2.comment = "vis2"
aud1_std_hit = grand_average(aud1_std_hit)
aud1_std_hit.comment = "aud1_std_hit"
aud1_odd_hit = grand_average(aud1_odd_hit)
aud1_odd_hit.comment = "aud1_odd_hit"
aud1_std_mis = grand_average(aud1_std_mis)
aud1_std_mis.comment = "aud1_std_mis"
aud1_odd_mis = grand_average(aud1_odd_mis)
aud1_odd_mis.comment = "aud1_odd_mis"
aud2_std_hit = grand_average(aud2_std_hit)
aud2_std_hit.comment = "aud2_std_hit"
aud2_odd_hit = grand_average(aud2_odd_hit)
aud2_odd_hit.comment = "aud2_odd_hit"
aud2_std_mis = grand_average(aud2_std_mis)
aud2_std_mis.comment = "aud2_std_mis"
aud2_odd_mis = grand_average(aud2_odd_mis)
aud2_odd_mis.comment = "aud2_odd_mis"
resp_odd_hit = grand_average(resp_odd_hit)
resp_odd_hit.comment = "resp_odd_hit"
resp_odd_mis = grand_average(resp_odd_mis)
resp_odd_mis.comment = "resp_odd_mis"
resp_std_hit = grand_average(resp_std_hit)
resp_std_hit.comment = "resp_std_hit"
resp_std_mis = grand_average(resp_std_mis)
resp_std_mis.comment = "resp_std_mis"

write_evokeds(
    root / "results" / "group_erp-ave.fif",
    [
        vis1,
        vis2,
        aud1_std_hit,
        aud1_std_mis,
        aud1_odd_hit,
        aud1_odd_mis,
        aud2_std_hit,
        aud2_std_mis,
        aud2_odd_hit,
        aud2_odd_mis,
        resp_std_hit,
        resp_std_mis,
        resp_odd_hit,
        resp_odd_mis,
    ],
    overwrite=True,
)
