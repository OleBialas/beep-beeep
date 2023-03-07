from pathlib import Path
import json
import pandas as pd
import numpy as np
from mne import find_events
from mne.io import read_raw_fif

root = Path(__file__).parent.parent.absolute()
subjects = list((root / "raw").glob("sub*"))

for subject in subjects:
    if (subject / "eeg").exists():
        raw = read_raw_fif(subject / "eeg" / f"{subject.name}-raw.fif", preload=True)
        # STEP1: process events
        events = find_events(raw)
        # remove duplicate triggers
        idx = [events[i, 2] == events[i + 1, 2] for i in range(len(events) - 1)]
        idx.append(False)
        events = events[~np.asarray(idx)]
        events[:, 0:] -= 2048  # subtract baseline

        blocks = list((subject / "beh").glob("*block*"))
        blocks.sort()
        # get the intervals where the sound was presented, frequency and response
        frequencies, targets, responses = [], [], []
        for block in blocks:
            seq = json.load(open(block))
            frequencies.append(seq["trials"])
            targets.append([d[0][0] for d in seq["data"]])
            responses.append([d[0][1] for d in seq["data"]])
        targets = np.concatenate(targets)
        responses = np.concatenate(responses)
        frequencies = np.asarray(seq["conditions"])[np.concatenate(frequencies) - 1]
        idx = list(range(0, len(events) + 3, 3))
        events = [events[i:j] for i, j in zip(idx[:-1], idx[1:])]

        df = pd.DataFrame(
            columns=["onset_time", "sound_frequency", "target_interval", "response"]
        )
        for i in range(len(events)):
            row = {
                "target_interval": [targets[i]],
                "onset_time": [
                    events[i][targets[i] - 1, 0] / raw.info["sfreq"] + 0.015
                ],
                "sound_frequency": [frequencies[i]],
                "response": [responses[i]],
            }
            df = pd.concat([df, pd.DataFrame.from_dict(row)], ignore_index=True)
        df.to_csv(subject / "beh" / "interval_detection.csv")
