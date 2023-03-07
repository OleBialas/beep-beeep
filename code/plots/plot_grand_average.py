from pathlib import Path
from matplotlib import pyplot as plt
from mne import read_evokeds, combine_evoked
from mne.preprocessing import compute_current_source_density

root = Path(__file__).parent.parent.parent.absolute()
evokeds = read_evokeds(root / "results" / "group_erp-ave.fif")
[e.crop(-0.2, 0.5) for e in evokeds]
[e.set_eeg_reference("average") for e in evokeds]
csds = [compute_current_source_density(e) for e in evokeds]

fig, ax = plt.subplots(2, sharex=True, sharey=True)
fig.suptitle("auditory - visual ERP")
combine_evoked(
    [evokeds[2], evokeds[3], evokeds[4], evokeds[5], evokeds[0]], (1, 1, 1, 1, -4)
)

fig, ax = plt.subplots(2, sharex=True, sharey=True)
fig.suptitle("common tone (1000Hz) in interval 1")
evokeds[2].plot(axes=ax[0], show=False)
ax[0].set(title="hit")
evokeds[3].plot(axes=ax[1], show=False)
ax[1].set(title="miss")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, sharex=True, sharey=True)
fig.suptitle("common tone (1000Hz) in interval 1")
csds[2].plot(axes=ax[0], show=False)
ax[0].set(title="hit")
csds[3].plot(axes=ax[1], show=False)
ax[1].set(title="miss")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, sharex=True, sharey=True)
fig.suptitle("rare tone (1000Hz) in interval 1")
evokeds[4].plot(axes=ax[0], show=False)
ax[0].set(title="hit")
evokeds[5].plot(axes=ax[1], show=False)
ax[1].set(title="miss")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, sharex=True, sharey=True)
fig.suptitle("rare tone (1000Hz) in interval 1")
csds[4].plot(axes=ax[0], show=False)
ax[0].set(title="hit")
csds[5].plot(axes=ax[1], show=False)
ax[1].set(title="miss")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, sharex=True, sharey=True)
fig.suptitle("tone in interval 1")
combine_evoked([evokeds[2], evokeds[4]], (1, 1)).plot(axes=ax[0], show=False)
ax[0].set(title="hit")
combine_evoked([evokeds[1], evokeds[5]], (1, 1)).plot(axes=ax[1], show=False)
ax[1].set(title="miss")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, sharex=True, sharey=True)
fig.suptitle("tone in interval 1")
combine_evoked([csds[2], csds[4]], (1, 1)).plot(axes=ax[0], show=False)
ax[0].set(title="hit")
combine_evoked([csds[1], csds[5]], (1, 1)).plot(axes=ax[1], show=False)
ax[1].set(title="miss")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, sharex=True, sharey=True)
fig.suptitle("Auditory")
combine_evoked([csds[2], csds[4], csds[6], csds[8]], (1, 1, 1, 1)).plot(
    axes=ax[0], show=False
)
ax[0].set(title="hits")
combine_evoked([csds[3], csds[5], csds[7], csds[9]], (1, 1, 1, 1)).plot(
    axes=ax[1], show=False
)
ax[1].set(title="misses")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(3)
fig.suptitle("Response")
evokeds[4].plot(axes=ax[0], show=False)
ax[0].set(title="common (1000Hz)")
evokeds[5].plot(axes=ax[1], show=False)
ax[1].set(title="rare (1200Hz)")
combine_evoked([evokeds[4], evokeds[5]], (1, -1)).plot(axes=ax[2], show=False)
ax[2].set(title="common - rare")
plt.tight_layout()
plt.show()
