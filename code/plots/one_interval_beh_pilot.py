from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

root = Path(__file__).parent.parent.parent.absolute()
df = pd.read_csv(root / "results" / "one_interval_beh_pilot.csv")
df["correct"] = df.is_tone == df.response

df.groupby(["subjectID", "is_rare"]).sum() / df.groupby(
    ["subjectID", "is_rare"]
).count()
