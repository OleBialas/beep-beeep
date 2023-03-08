from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

root = Path(__file__).parent.parent.parent.absolute()
df = pd.read_csv(root / "results" / "one_interval_beh_pilot.csv")
df["correct"] = df.is_tone == df.response

# compare the over-all hit-rate between common and rare frequencies
rare_hitrate = (
    df[df.is_rare.astype(bool)].groupby("subject_id").sum().correct
    / df[df.is_rare.astype(bool)].groupby("subject_id").count().correct
).to_numpy()

common_hitrate = (
    df[~df.is_rare.astype(bool)].groupby("subject_id").sum().correct
    / df[~df.is_rare.astype(bool)].groupby("subject_id").count().correct
).to_numpy()

for rare, common in zip(rare_hitrate, common_hitrate):
    plt.plot([1, 2], [common, rare])
    plt.xlabel("Frequency probability")
    plt.xticks([1, 2], ("high", "low"))
    plt.ylabel("hitrate")


df.groupby(["subjectID", "is_rare"]).sum() / df.groupby(
    ["subjectID", "is_rare"]
).count()
