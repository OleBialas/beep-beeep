from pathlib import Path
import json
import numpy as np
from matplotlib import pyplot as plt

root = Path(__file__).parent.parent.parent.absolute()
hitrate_frequent, hitrate_rare = np.zeros((20, 3)), np.zeros((20, 3))
subjects = ["sub-008", "sub-009", "sub-011"]
for isub, subject in enumerate(subjects):
    target, response, condition = [], [], []
    for iblock, block in enumerate((root / "raw" / subject / "beh").glob("*block*")):
        seq = json.load(open(block))
        target = np.asarray([t[0][0] for t in seq["data"]])
        response = np.asarray([t[0][1] for t in seq["data"]])
        condition = np.asarray([t for t in seq["trials"]])
        hitrate_frequent[iblock, isub] = sum(
            (response == target)[condition == 1]
        ) / sum(condition == 1)
        hitrate_rare[iblock, isub] = sum((response == target)[condition == 2]) / sum(
            condition == 2
        )
colors = ["blue", "green", "orange"]
for i in range(3):
    plt.scatter(
        [1, 2],
        [hitrate_frequent.mean(axis=0)[i], hitrate_rare.mean(axis=0)[i]],
        color=colors[i],
    )
    plt.plot(
        [1, 2],
        [hitrate_frequent.mean(axis=0)[i], hitrate_rare.mean(axis=0)[i]],
        color=colors[i],
    )
    plt.xticks([1, 2], ["frequent (1khz)", "rare (1.2 khz)"])
    plt.ylabel("hitrate")
    plt.savefig(root / "results" / "plots" / "pilot_report" / "behavior.png")
