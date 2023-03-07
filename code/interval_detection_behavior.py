from pathlib import Path
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

root = Path(__file__).parent.parent.absolute()

rtime1, rtime2, correct1, correct2 = [], [], [], []
for sub in (root / "raw").glob("sub*"):
    data, freqs = [], []
    for block in (sub / "beh").glob("*block*"):
        seq = json.load(open(block))
        con1, con2 = seq["conditions"]
        freqs.append(np.asarray(seq["conditions"])[np.asarray(seq["trials"]) - 1][:50])
        data.append(np.stack(seq["data"])[:, 0, :])
    freqs = np.concatenate(freqs)
    data = np.concatenate(data)
    df = pd.DataFrame(data, columns=["target", "response", "rtime"])
    df["correct"] = data[:, 0] == data[:, 1]
    df["freqs"] = freqs
    correct1.append(df.groupby("freqs").mean()["correct"][con1])
    correct2.append(df.groupby("freqs").mean()["correct"][con2])
    rtime1.append(df.groupby("freqs").mean()["rtime"][con1])
    rtime2.append(df.groupby("freqs").mean()["rtime"][con2])

plt.boxplot([correct1, correct2])
plt.xticks(ticks=[1, 2], labels=[con1, con2])
plt.ylabel("Accuracy [a.u.]")
plt.xlabel("Tone frequency [Hz]")
plt.title("Average Hitrate (n=6)")
plt.show()

plt.boxplot([rtime1, rtime2])
plt.xticks(ticks=[1, 2], labels=[con1, con2])
plt.ylabel("Reaction Time [s]")
plt.xlabel("Tone frequency [Hz]")
plt.title("Average Reaction Time (n=6)")
plt.show()
