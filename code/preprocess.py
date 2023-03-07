from pathlib import Path
import math
import json
import numpy as np
import pandas as pd
from mne import find_events, concatenate_epochs
from mne.io import read_raw_bdf
from mne.epochs import Epochs
from mne.channels import make_standard_montage
from pyprep.ransac import find_bad_by_ransac
from meegkit.dss import dss_line

root = Path(__file__).parent.parent.absolute()
montage = make_standard_montage("biosemi128")
desired_sfreq = 512  # frequency that the data is resampled to`
subjects = list((root / "raw").glob("sub*"))
subjects.sort()
for subject in subjects:
    if (subject / "eeg").exists():
        out_dir = root / "preprocessed" / subject.name
        if not out_dir.exists():
            out_dir.mkdir()
        blocks = list((subject / "beh").glob("*block*"))
        blocks.sort()
        df = pd.DataFrame(columns=["time", "target", "response", "frequency"])
        raw = read_raw_bdf(subject / "eeg" / f"{subject.name}.bdf")
        raw.pick_channels(raw.info["ch_names"][:128] + ["EXG1", "EXG2", "Status"])
        # STEP1: split recording into runs
        events = find_events(raw, shortest_event=1)
        # it seems like triggers with the code 66613 are duplicates of 66560 (rec stop)
        events[:, 2][events[:, 2] == 66613] = 66560
        if events[0, 2] == 66560:
            events = events[1:]
        # and sometimes, 3072 gets turned into 1024
        events[:, 2][events[:, 2] == 1024] = 3072
        # remove duplicate triggers
        idx = [events[i, 2] == events[i - 1, 2] for i in range(1, len(events))]
        idx.insert(0, False)
        events = events[~np.asarray(idx)]
        # When recording in the right booth: vis1 == 1073, vis2 ==1074, resp == 1072
        # sound == 3072, rec_start == 66814, rec_stop == 66560
        event_id = {"resp": 1072, "vis1": 1073, "vis2": 1074, "aud": 3072}
        starts = np.where(events[:, 2] == 66814)[0]
        stops = np.where(events[:, 2] == 66560)[0]
        stops = np.append(stops, len(events))  # the last block goes till the end
        epochs = []
        # preprocess each run
        for i, (start, stop) in enumerate(zip(starts, stops)):

            run = raw.copy()
            run_events = events[start + 1 : stop]
            run_events = events[start + 1 : stop]
            if i < 20:
                block = blocks[i]
                seq = json.load(open(block))  # behavioral data
                freq = np.asarray(seq["conditions"])[np.asarray(seq["trials"]) - 1]
                targ = np.asarray([d[0][0] for d in seq["data"]])
                resp = np.asarray([d[0][1] for d in seq["data"]])
            # there should be 4 triggers per trial, if not we'll assume that the last
            # trial is incomplete and remove it
            n_remove = len(run_events) % 4
            if n_remove:
                run_events = run_events[0:-(n_remove)]
            n_trials = int(len(run_events) / 4)

            # shift visual events by the delay between visual and audio in each epoch
            # so that later, we can subtract visual from audio
            delays = np.diff(run_events[:, 0])[run_events[1:, 2] == 3072]
            run_events[:, 0][run_events[:, 2] == 1073] += delays - 1
            run_events[:, 0][run_events[:, 2] == 1074] += delays - 1
            if i < 20:
                run.crop(
                    raw.times[run_events[0, 0]] - 1, raw.times[run_events[-1, 0]] + 2
                )
            else:
                run.crop(raw.times[run_events[0, 0]] - 1, raw.times[run_events[-1, 0]])
            run.load_data()
            # filter and resample
            current_sfreq = raw.info["sfreq"]
            decim = np.round(current_sfreq / desired_sfreq).astype(int)
            obtained_sfreq = current_sfreq / decim
            lowpass_freq = obtained_sfreq / 3.0
            run.filter(0.5, lowpass_freq)
            # re-reference, then remove mastoids and apply standard montage
            run.set_eeg_reference(["EXG1", "EXG2"])
            run.pick_channels(raw.info["ch_names"][:128])
            run.set_montage(montage)
            # denoise and detrend
            X = run.get_data().T
            X, _ = dss_line(X, fline=60, sfreq=raw.info["sfreq"], nremove=5)
            run._data = X.T  # put the data back into raw
            del X
            # run ransac to detect bad channels
            ch_pos = np.stack([ch["loc"][:3] for ch in run.info["chs"]])
            names = np.asarray(run.info["ch_names"])
            bads, _ = find_bad_by_ransac(
                run._data,
                run.info["sfreq"],
                names,
                ch_pos,
                [],
                frac_bad=0.3,
                corr_thresh=0.8,
            )
            run.info["bads"] = bads
            run.interpolate_bads()
            e = Epochs(
                run,
                run_events,
                baseline=None,
                tmin=-0.5,
                tmax=1.5,
                event_id=event_id,
                decim=decim,
            )
            times = e.events[e.events[:, 2] == 3072][:, 0]

            if i < 20:
                epochs.append(e)
                freq, targ, resp, times = (
                    freq[:n_trials],
                    targ[:n_trials],
                    resp[:n_trials],
                    times[:n_trials],
                )
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                "time": times / run.info["sfreq"],
                                "target": targ,
                                "response": resp,
                                "frequency": freq,
                            }
                        ),
                    ]
                )
            else:  # last block is saved separately
                e.save(out_dir / f"{subject.name}_clean-epo.fif", overwrite=True)
        epochs = concatenate_epochs(epochs, add_offset=False)
        # save results
        epochs.save(out_dir / f"{subject.name}-epo.fif", overwrite=True)
        df.to_csv(out_dir / f"{subject.name}-interval_detection.csv")
