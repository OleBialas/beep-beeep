"""
compute the auditory erp by subtracting the erp from visual-only trials
"""
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mne import read_epochs, combine_evoked

root = Path(__file__).parent.parent.parent.absolute()
subjects = ["sub-008", "sub-009", "sub-011"]
evoked = combine_evoked(
    [
        read_epochs(root / "preprocessed" / subject / f"{subject}_clean-epo.fif")[
            "aud"
        ].average()
        for subject in subjects
    ],
    (1 / 3, 1 / 3, 1 / 3),
)
evoked.crop(-0.1, 0.8)
evoked = evoked.set_eeg_reference("average")
fig = evoked.plot_joint(times=[0.095, 0.165], show=False)
fig.savefig(root / "results" / "plots" / "pilot_report" / "clean_erp.png")

for subject in subjects:
    epochs = read_epochs(root / "preprocessed" / subject / f"{subject}-epo.fif")
    epochs = epochs.set_eeg_reference("average")

    epochs.crop(-0.1, 0.5)
    df = pd.read_csv(
        root / "preprocessed" / subject / f"{subject}-interval_detection.csv"
    )

    aud1 = epochs["aud"][df.target == 1].average()
    vis1 = epochs["vis1"][df.target == 2].average()
    diff1 = combine_evoked([aud1, vis1], (1, -1))

    fig, ax = plt.subplots(3, sharex=True, sharey=True)
    aud1.plot(axes=ax[0], show=False)
    ax[0].set(title="auditory and visual")
    vis1.plot(axes=ax[1], show=False)
    ax[1].set(title="visual only")
    diff1.plot(axes=ax[2], show=False)
    ax[2].set(title="difference / auditory only")
    plt.tight_layout()
    fig.savefig(root / "results" / "plots" / "pilot_report" / f"{subject}_aud_erp.png")

    aud2 = epochs["aud"][df.target == 2].average()
    vis2 = epochs["vis2"][df.target == 1].average()
    diff2 = combine_evoked([aud2, vis2], (1, -1))

    combined = combine_evoked([aud1, aud2, vis1, vis2], (1, 1, -1, -1))
    fig = combined.plot_joint(times=[0.38], show=False)
    fig.savefig(
        root / "results" / "plots" / "pilot_report" / f"{subject}_late_component.png"
    )

    # check if the late component differs between hit and miss trials
    aud1_hit = epochs["aud"][np.logical_and(df.target == 1, df.response == 1)].average()
    aud1_mis = epochs["aud"][np.logical_and(df.target == 1, df.response != 1)].average()
    vis1 = epochs["vis1"][df.target == 2].average()

    aud2_hit = epochs["aud"][np.logical_and(df.target == 2, df.response == 2)].average()
    aud2_mis = epochs["aud"][np.logical_and(df.target == 2, df.response != 2)].average()
    vis2 = epochs["vis2"][df.target == 1].average()

    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    combined_hit = combine_evoked([aud1_hit, aud2_hit, vis1, vis2], (1, 1, -1, -1))
    combined_mis = combine_evoked([aud1_mis, aud2_mis, vis1, vis2], (1, 1, -1, -1))
    combined_hit.plot(axes=ax[0], show=False)
    combined_mis.plot(axes=ax[1], show=False)
    ax[0].set(xlabel=None, title="hit")
    ax[1].set(title="miss")
    fig.savefig(root / "results" / "plots" / "pilot_report" / f"{subject}_hitvmiss.png")
