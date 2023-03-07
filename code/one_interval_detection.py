from argparse import ArgumentParser
import statistics
import shutil
import json
from pathlib import Path
import serial
import numpy as np
from psychopy import visual, event, core, prefs
import psychtoolbox as ptb

prefs.hardware["audioLib"] = ["PTB"]
from psychopy.sound import Sound
import slab

root = Path(__file__).parent.parent.absolute()
win = visual.Window([1920, 1080], fullscr=True, units="pix")
fixation = visual.Circle(win, size=10, lineColor="white", fillColor="lightGrey")
# port = serial.Serial(port="COM3", baudrate=115200)
event.globalKeys.clear()

# define global keys and the functions they execute
def exit_experiment():
    port.close()
    win.close()
    raise SystemExit


def pause_experiment():
    port.write(str.encode(info["trigger"]["pause"]))
    prompt = visual.TextStim(
        win,
        text="The experiment has been paused, wait for the experimenter.",
        height=info["prompt"]["height"],
    )
    prompt.draw()
    win.flip()
    event.waitKeys(keyList=["return"])
    port.write(str.encode(info["trigger"]["unpause"]))
    core.wait(1)
    win.flip()


event.globalKeys.add(key="escape", func=exit_experiment)
event.globalKeys.add(key="p", func=pause_experiment)


def run_experiment(subject):
    global info
    sub_folder = root / "raw" / subject
    if not sub_folder.exists():
        sub_folder.mkdir()
    else:
        resp = None
        while not resp in ["Y", "y", "N", "n"]:
            resp = input(
                "A folder for that subject already exists! Do you want to continue and overwrite existing files (Y/N)"
            )
        if resp.lower() == "n":
            return
    if not (sub_folder / "beh").exists():
        (sub_folder / "beh").mkdir()
    if not (sub_folder / "one_interval_detection_parameters.json").exists():
        shutil.copyfile(
            root / "code" / "one_interval_detection_parameters.json",
            sub_folder / "one_interval_detection_parameters.json",
        )
    info = json.load(open(sub_folder / "one_interval_detection_parameters.json"))
    # determine which frequency is standard and which is deviant
    std, dev = np.random.choice(info["freqs"], 2, replace=False)
    info["standardFreq"], info["deviantFreq"] = std, dev
    prompt = visual.TextStim(
        win, text=info["prompt"]["welcome"], height=info["prompt"]["height"]
    )
    prompt.draw()
    win.flip()
    event.waitKeys(keyList=["return"])
    # STEP1: calibrate the output
    level = level_calibration(info)
    info["hearingThresh"] = int(level)
    json.dump(info, open(sub_folder / "interval_detection_parameters.json", "w"))

    # STEP2: estimate threshold for detecting tones in noise
    prompt = visual.TextStim(
        win, text=info["prompt"]["threshold"], height=info["prompt"]["height"]
    )
    prompt.draw()
    win.flip()
    event.waitKeys(keyList=["return"])
    seq = detection_threshold(info)
    seq.save_json(
        sub_folder / "beh" / f"{subject}_threshold_estimation.json", clobber=True
    )
    info["detectionThresh"] = statistics.mode(seq.intensities)
    json.dump(info, open(sub_folder / "interval_detection_parameters.json", "w"))

    # STEP3: run the experimental blocks
    prompt = visual.TextStim(
        win, text=info["prompt"]["blocks"], height=info["prompt"]["height"]
    )
    prompt.draw()
    win.flip()
    event.waitKeys(keyList=["return"])
    for iblock in range(info["nBlocks"]):
        port.write(str.encode(info["trigger"]["unpause"]))
        core.wait(1)
        seq = run_block(info)
        core.wait(3)
        port.write(str.encode(info["trigger"]["pause"]))
        seq.save_json(
            sub_folder / "beh" / f"{subject}_block{str(iblock+1).zfill(2)}.json",
            clobber=True,
        )
        prompt = visual.TextStim(
            win,
            text=f"Good job, you completed trial {iblock+1} of {info['nBlocks']}! \n \n Feel free to take a break. If you want to exit the booth, ring the bell to let the experimenter know. \n \n Once you are ready, press enter \u23ce to continue.",
            height=info["prompt"]["height"],
        )
        prompt.draw()
        win.flip()
        event.waitKeys(keyList=["return"])

    # now do one last block whith only beeps and no noise
    prompt = visual.TextStim(
        win,
        text="We are almost done! In the last block the tones will be clearly audible and there will be no background noise. Again, use the number keys to indicate the number that was on the screen when you heard the tone",
        height=info["prompt"]["height"],
    )
    prompt.draw()
    win.flip()
    event.waitKeys(keyList=["return"])

    port.write(str.encode(info["trigger"]["unpause"]))
    core.wait(1)
    seq = slab.Trialsequence(1, 100)
    seq.conditions = [1000]
    for target_frequency in seq:
        target_interval = np.random.randint(1, 3)
        response = _run_trial(
            info,
            n_intervals=2,
            target_interval=target_interval,
            target_frequency=target_frequency,
            target_level=info["hearingThresh"] + info["noiseLevel"],
        )
        seq.add_response((target_interval, response))

    port.write(str.encode(info["trigger"]["pause"]))
    prompt = visual.TextStim(
        win,
        text="Thank you for participating in our experiment! \n \n Please ring the bell now.",
        height=info["prompt"]["height"],
    )
    prompt.draw()
    win.flip()
    event.waitKeys(keyList=["return"])
    port.close()
    win.close()


def run_block(info):
    n_trials = int(info["nTrials"] / info["nBlocks"])
    frequencies = np.append(  # 1 = common, 2 = rare frequency
        np.repeat(1, 2 * int(info["standardProb"] / info["deviantProb"])), [2, 2]
    )
    tones = np.concatenate(
        [
            np.repeat(1, int(info["standardProb"] / info["deviantProb"])),
            np.repeat(2, int(info["standardProb"] / info["deviantProb"])),
            np.array([1, 2]),
        ]
    )
    frequencies = np.repeat(frequencies, np.ceil(n_trials / len(frequencies)))
    tones = np.repeat(tones, np.ceil(n_trials / len(tones)))
    idx = np.random.choice(range(len(tones)), len(tones), replace=False)
    tones, frequencies = tones[idx], frequencies[idx]
    tone_seq = slab.Trialsequence(1, n_trials)
    tone_seq.trials = tones[:n_trials].tolist()
    tone_seq.conditions, tone_seq.n_conditions = [0, 1], 2
    freq_seq = slab.Trialsequence(1, n_trials)
    freq_seq.trials, freq_seq.n_conditions = frequencies[:n_trials].tolist(), 2
    freq_seq.conditions = [info["standardFreq"], info["deviantFreq"]]
    assert freq_seq.n_trials == tone_seq.n_trials
    del tones, frequencies
    noise = slab.Sound.whitenoise(  # generate background noise
        samplerate=info["sampleRate"],
        duration=info["noiseDur"],
        level=info["hearingThresh"] + info["noiseLevel"],
        n_channels=2,
    )
    noise.data[:, 1] = 0  # set trigger channel to 0
    present = Sound(stereo=True)  # convert to psychopy
    present._setSndFromArray(noise.data)
    present.play(loops=1000)  # start playing background noise

    # run the block
    for tone, freq in zip(tone_seq, freq_seq):
        seq.add_response(tone)
        print(f"Trial {tone_seq.this_n} of {tone_seq.n_trials}")
        if tone:
            level = noise.level + info["detectionThresh"]
        else:
            tone = None
        response = _run_trial(info, target_frequency=freq, target_level=level)
        seq.add_response(response)
    present.stop()
    return seq


def level_calibration(info):
    prompt = visual.TextStim(
        win, text=info["prompt"]["calibration"], height=info["prompt"]["height"]
    )
    prompt.draw()
    win.flip()
    sound = slab.Sound.whitenoise(
        duration=1.0, samplerate=info["sampleRate"], n_channels=2
    )
    sound = sound.ramp(duration=0.2)  # add 200 ms on and offset ramp
    sound.data[:, 1] = 0
    present = Sound(stereo=True)
    present._setSndFromArray(sound.data)
    done = False
    while done is False:
        present.play(loops=10)
        key = event.waitKeys(keyList=["up", "down", "return"])[0]
        if key == "down":
            sound.channel(0).level -= 1
        elif key == "up":
            sound.channel(0).level += 1
        elif key == "return":
            done = True
        present.stop()
        present._setSndFromArray(sound.data)
        present.play(loops=10)
    present.stop()
    return sound.channel(0).level


def detection_threshold(info):
    """
    Estimate the subjects detection threshold using an adaptive staircase.
    The step size will decrease after the first reversal and the threshold
    will be estimated as the mode of the staircase (i.e. the intensity
    where the subject spent the most time).
    Arguments:
        info (dict): experimental parameters.
    Returns:
        seq (slab.psychoacoustics.Staircase): staircase object containg responses.
    """
    seq = slab.Staircase(
        start_val=info["staircase"]["start"],
        step_sizes=info["staircase"]["step1"],
        min_val=info["staircase"]["stop"],
        n_reversals=1,
    )
    noise = slab.Sound.whitenoise(  # generate background noise
        samplerate=info["sampleRate"],
        duration=info["noiseDur"],
        level=info["hearingThresh"] + info["noiseLevel"],
        n_channels=2,
    )
    noise.data[:, 1] = 0
    present = Sound(stereo=True)  # convert to psychopy
    present._setSndFromArray(noise.data)
    present.play(loops=1000)  # start playing background noise

    for target_level in seq:
        print(f"Trial number {seq.this_trial_n+1}, intensity:{seq.intensities[-1]}dB")
        play_sound = np.random.binomial(1, 0.5)  # whether or not to play a sound
        if play_sound is False:
            level = None
        else:
            noise.channel(0).level + target_level
        response = _run_trial(
            info,
            target_frequency=info["threshFreq"],
            target_level=level,
        )
        seq.add_response(play_sound == response)
    # then, find the threshold using a smaller step size
    seq = slab.Staircase(
        start_val=target_level,
        step_sizes=info["staircase"]["step2"],
        n_reversals=info["staircase"]["nTrials"],
        n_up=1,
        n_down=3,
    )
    while seq.this_trial_n < info["staircase"]["nTrials"]:
        target_level = seq.__next__()
        print(f"Trial number {seq.this_trial_n+1}, intensity:{seq.intensities[-1]}dB")
        play_sound = np.random.binomial(1, 0.5)  # whether or not to play a sound
        if play_sound is False:
            level = None
        else:
            noise.channel(0).level + target_level
        response = _run_trial(
            info,
            target_frequency=info["standardFreq"],
            target_level=level,
        )
        seq.add_response(play_sound == response)
    present.stop()
    return seq


def _run_trial(info, target_frequency, target_level):
    """Run a single trial of the one-interval detection task.
    Arguments:
        info (dict): experimental parameters.
        target_frequency (int): frequency of the target tone in Hz.
        target_level: level of the target tone. If None, no tone is played.
    Returns:
        response (int): 1 if the subject heard a tone, 0 if not.
    """
    trial_dur = info["silStart"] + info["silEnd"] + info["stimWindow"]
    if target_level is not None:  # prepare target tone
        target = slab.Sound.tone(
            frequency=target_frequency,
            samplerate=info["sampleRate"],
            duration=info["stimDur"],
            level=target_level,
            n_channels=2,
        )
        onset = np.random.uniform(
            info["silStart"], trial_dur - info["silEnd"] - info["stimDur"]
        )
        target = target.ramp(duration=0.005)
        target.channel(1).level = 120
        present = Sound(stereo=True)
        present._setSndFromArray(target.data)

    # prompt the subject to start the experiment
    prompt = visual.TextStim(
        win, text=info["prompt"]["start"], height=info["prompt"]["height"]
    )
    prompt.draw()
    win.flip()
    event.waitKeys(keyList=["space"])

    # present fixation dot and schedule sound (if trial contains one)
    fixation.draw()
    win.flip()
    if target_level is not None:
        now = ptb.GetSecs()
        present.play(when=now + onset)  # play in EXACTLY 0.5s
    core.wait(trial_dur)

    # prompt subject to respond
    prompt = visual.TextStim(
        win, text=info["prompt"]["response"], height=info["prompt"]["height"]
    )
    prompt.draw()
    win.flip()
    response = event.waitKeys(keyList=info["keys"].keys())[0]
    port.write(str.encode("0"))  # EEG trigger for response
    win.flip()
    return info["keys"][response]
