from pathlib import Path
import numpy as np
from mne import read_epochs
from mne.viz import plot_topomap
from meegkit.dss import dss0, dss1
from meegkit.utils.covariances import tscov

root = Path(__file__).parent.parent.absolute()

subject = "sub-007"
epochs = read_epochs(root / "preprocessed" / subject / f"{subject}_aud-epo.fif")

to_jd1, from_jd1, to_jd2, from_jd2 = compute_tranformation(epochs, 1000, 1200, keep=10)
stc = apply_transform(epochs.get_data(), [to_jd1, to_jd2])
idx1 = np.where(epochs.events[:, 2] == 1000)[0]
idx2 = np.where(epochs.events[:, 2] == 1200)[0]


def compute_transformation(epochs, condition1, condition2, keep):

    if not (condition1 in epochs.events[:, 2] and condition2 in epochs.events[:, 2]):
        raise ValueError("'conditions' must be values of two event types!")
    X = epochs.get_data().transpose(2, 1, 0)
    events = epochs.events

    to_jd1, from_jd1, _, pwr = dss1(X)  # compute the transformations
    del X
    to_jd1 = to_jd1[:, np.argsort(pwr)[::-1]]  # sort them by magnitude
    from_jd1 = from_jd1[np.argsort(pwr)[::-1], :]
    to_jd1 = to_jd1[:, 0:keep]  # only keep the largest ones
    from_jd1 = from_jd1[0:keep, :]

    Y = apply_transform(
        epochs.get_data(), to_jd1
    )  # apply the unmixing matrix to get the components

    idx1 = np.where(events[:, 2] == condition1)[0]
    idx2 = np.where(events[:, 2] == condition2)[0]
    D = Y[idx1, :, :].mean(axis=0) - Y[idx2, :, :].mean(
        axis=0
    )  # compute the difference between conditions
    Y, D = Y.T, D.T  # shape must be in shape (n_times, n_chans[, n_trials])
    c0, nc0 = tscov(Y)
    c1, nc1 = tscov(D)
    c0 /= nc0  # divide by total weight to normalize
    c1 /= nc1
    to_jd2, from_jd2, _, pwr = dss0(c0, c1)  # compute the transformations
    to_jd2 = to_jd2[:, np.argsort(pwr)[::-1]]  # sort them by magnitude
    from_jd2 = from_jd2[np.argsort(pwr)[::-1], :]

    return to_jd1, from_jd1, to_jd2, from_jd2


def apply_transform(data, transforms):
    if not isinstance(transforms, list):
        transforms = [transforms]
    n_epochs, n_channels, n_times = data.shape
    data = data.transpose(1, 0, 2)
    data = data.reshape(n_channels, n_epochs * n_times).T
    for i, transform in enumerate(transforms):
        if i == 0:
            transformed = data @ transform
        else:
            transformed = transformed @ transform
    transformed = np.reshape(transformed.T, [-1, n_epochs, n_times]).transpose(
        [1, 0, 2]
    )
    return transformed
