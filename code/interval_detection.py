from argparse import ArgumentParser
import shutil
import statistics
import json
from pathlib import Path
import serial
import numpy as np
from psychopy import visual, event, core, prefs

prefs.hardware["audioLib"] = ["PTB"]
from psychopy.sound import Sound
import slab

root = Path(__file__).parent.parent.absolute()
win = visual.Window([1920, 1080], fullscr=True, units="pix")
fixation = visual.Circle(win, size=10, lineColor="white", fillColor="lightGrey")
port = serial.Serial(port="COM3", baudrate=115200)
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
    if not (sub_folder / "interval_detection_parameters.json").exists():
        shutil.copyfile(
            root / "code" / "interval_detection_parameters.json",
            sub_folder / "interval_detection_parameters.json",
        )
    info = json.load(open(sub_folder / "interval_detection_parameters.json"))
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
    trials = np.append(np.repeat(1, int(info["standardProb"] / info["deviantProb"])), 2)
    trials = np.repeat(trials, np.ceil(n_trials / len(trials)))
    np.random.shuffle(trials)
    seq = slab.Trialsequence(1, n_trials)
    seq.trials = trials[:n_trials].tolist()
    seq.conditions = [info["standardFreq"], info["deviantFreq"]]
    seq.n_conditions = 2
    seq.label = f"0: {info['deviantFreq']} Hz; 1:{info['standardFreq']} Hz"
    del trials
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

    # run the block
    for target_frequency in seq:
        print(f"Trial {seq.this_n} of {seq.n_trials}")
        target_interval = np.random.randint(1, 3)
        response = _run_trial(
            info,
            n_intervals=2,
            target_interval=target_interval,
            target_frequency=target_frequency,
            target_level=noise.level + info["detectionThresh"],
        )
        seq.add_response((target_interval, response))
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
    # first, go in larger steps until specified level is reached or wrong answer is given
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
        target_interval = np.random.randint(1, 4)
        response = _run_trial(
            info,
            n_intervals=3,
            target_interval=target_interval,
            target_frequency=info["standardFreq"],
            target_level=noise.channel(0).level + target_level,
        )
        if response == target_interval:
            seq.add_response(1)
        else:
            seq.add_response(0)
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
        print(f"Trial number {seq.this_trial_n}, intensity:{seq.intensities[-1]}dB")
        target_interval = np.random.randint(1, 4)
        response = _run_trial(
            info,
            n_intervals=3,
            target_interval=target_interval,
            target_frequency=info["standardFreq"],
            target_level=noise.level + target_level,
        )
        if response == target_interval:
            seq.add_response(1)
        else:
            seq.add_response(0)
    present.stop()
    return seq


def _run_trial(info, n_intervals, target_interval, target_frequency, target_level):

    texts = [
        visual.TextStim(win, text=t, height=info["prompt"]["height"])
        for t in ["1", "2", "3"]
    ]
    target = slab.Sound.tone(
        frequency=target_frequency,
        samplerate=info["sampleRate"],
        duration=info["stimDur"],
        level=target_level,
        n_channels=2,
    )
    target = target.ramp(duration=0.005)
    target.channel(1).level = 120
    present = Sound(stereo=True)
    present._setSndFromArray(target.data)
    fixation.draw()
    win.flip()
    core.wait(info["fixDur"])

    for i, text in enumerate(texts[:n_intervals]):
        text.draw()
        if i + 1 == target_interval:
            win.callOnFlip(present.play)
        win.callOnFlip(port.write, str.encode(text.text))
        win.flip()
        core.wait(info["stimDur"])
        win.flip()
        core.wait(info["stimDur"])
    if n_intervals == 2:
        text = info["prompt"]["response2"]
    elif n_intervals == 3:
        text = info["prompt"]["response3"]
    prompt = visual.TextStim(win, text=text, height=info["prompt"]["height"])
    prompt.draw()
    win.flip()
    response = int(event.waitKeys(keyList=info["keyList"][:n_intervals])[0])
    port.write(str.encode("0"))
    win.flip()
    return response


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("subjectID")
    args = parser.parse_args()
    subject = f"sub-{str(args.subjectID).zfill(3)}"
    run_experiment(subject)
