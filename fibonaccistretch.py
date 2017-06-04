"""
Fibonacci Stretch
by David Su http://usdivad.com/

A method of time-stretching an existing audio track such that its rhythmic
pulses become expanded or contracted along the Fibonacci sequence, using
Euclidean rhythms as the basis for modification.
"""

# ## Part 1 - Representing rhythm as symbolic data

# Standard libraries
import math

# External libraries
import IPython.display as ipd
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Fork of Brian House's implementation of Bjorklund's algorithm
# https://github.com/brianhouse/bjorklund
import bjorklund


# Generate clicks based on input rhythm bit array (e.g. [1,0,0,1,0,0,1,0])
def generate_rhythm_clicks(rhythm, click_interval=0.25, sr=44100):
    step_length_samples = int(librosa.time_to_samples(click_interval, sr=sr))
    rhythm_length_samples = step_length_samples * (len(rhythm))
    
    # Generate click times
    pulse_click_times, step_click_times = generate_rhythm_times(rhythm, click_interval)
    
    # Generate pulse clicks
    pulse_click_times = np.array([i * click_interval for i in range(len(rhythm))
                                           if rhythm[i] != 0])
    pulse_clicks = librosa.clicks(times=pulse_click_times, click_freq=2000.0, sr=sr, length=rhythm_length_samples)

    # Generate step clicks
    step_click_times = np.array([i * click_interval for i in range(len(rhythm))])
    step_clicks = librosa.clicks(times=step_click_times, click_freq=1000.0, sr=sr, length=rhythm_length_samples)    
    step_clicks = np.hstack((step_clicks, np.zeros(step_length_samples, dtype="int"))) # add last step samples

    # Add zeros to pulse clicks so that it's the same length as the step clicks signal
    pulse_clicks = np.hstack((pulse_clicks, np.zeros(len(step_clicks)-len(pulse_clicks), dtype="int")))
    
    # Ensure proper length
    pulse_clicks = pulse_clicks[:rhythm_length_samples]
    step_clicks = step_clicks[:rhythm_length_samples]
    
    return (pulse_clicks, step_clicks)

# Generate times for a rhythm
def generate_rhythm_times(rhythm, interval):
    pulse_times = np.array([float(i * interval) for i in range(len(rhythm)) if rhythm[i] != 0])
    step_times = np.array([float(i * interval) for i in range(len(rhythm))])
    return (pulse_times, step_times)



# Function to calculate pulse lengths based on rhythm patterns
def calculate_pulse_lengths(rhythm):
    pulse_indices = np.array(([i for i,p in enumerate(rhythm) if p > 0]))
    pulse_indices = np.hstack((pulse_indices, len(rhythm)))
    pulse_lengths = np.array([pulse_indices[i+1] - pulse_indices[i] for i in range(len(pulse_indices) - 1)])
    
    return pulse_lengths


# ## Part 2 - Fibonacci rhythms


# Calculate nth fibonacci number
def fibonacci(n):
    if n == 0 or n == 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)


# Find the index of a Fibonacci number
def find_fibonacci_index(n):
    phi = (1 + math.sqrt(5)) / 2 # Golden ratio; 1.61803398875...
    return int(math.log((n * math.sqrt(5)) + 0.5) / math.log(phi))

# Expand pulse lengths by factor of 1
def fibonacci_expand_pulse_lengths(pulse_lengths):
    return fibonacci_scale_pulse_lengths(pulse_lengths, 1)


# Function to scale pulse lengths along the Fibonacci sequence
# 
# Note that `scale_amount` determines the direction and magnitude of the scaling.
# If `scale_amount` > 0, it corresponds to a rhythmic expansion.
# If `scale_amount` < 0, it corresponds to a rhythmic contraction.
# If `scale_amount` == 0, the original scale is maintained and no changes are made.
def fibonacci_scale_pulse_lengths(pulse_lengths, scale_amount=0):
    scaled_pulse_lengths = np.array([], dtype="int")
    for pulse_length in pulse_lengths:
        fib_i = find_fibonacci_index(pulse_length)
        # if fib_i + scale_amount < 0:
        #     print("ERROR: Scale amount out of bounds")
        #     return pulse_lengths
        scaled_pulse_length = fibonacci(max(fib_i + scale_amount, 0))
        scaled_pulse_lengths = np.hstack((scaled_pulse_lengths, scaled_pulse_length))
    return scaled_pulse_lengths


# Define the functions we'll use to scale rhythms along the Fibonacci sequence
def fibonacci_scale_rhythm(rhythm, scale_amount):
    pulse_lengths = calculate_pulse_lengths(rhythm)
    scaled_pulse_lengths = fibonacci_scale_pulse_lengths(pulse_lengths, scale_amount)
    scaled_pulse_lengths = np.array([p for p in scaled_pulse_lengths if p > 0])

    scaled_rhythm = pulse_lengths_to_rhythm(scaled_pulse_lengths)
    return scaled_rhythm

def pulse_lengths_to_rhythm(pulse_lengths):
    rhythm = np.array([], dtype="int")
    for p in pulse_lengths:
        pulse = np.zeros(p, dtype="int")
        pulse[0] = 1
        rhythm = np.hstack((rhythm, pulse))
    return rhythm


# ## Part 3 - Mapping rhythm to audio


# Estimate tempo from an input signal
def estimate_tempo(y, sr, start_bpm=120.0):
    # Estimate tempo
    onset_env = librosa.onset.onset_strength(y, sr=sr) # TODO: Compare this with librosa.beat.beat_track
    tempo = librosa.beat.tempo(y, sr=sr, onset_envelope=onset_env, start_bpm=start_bpm)
    
    return float(tempo)


# Calculate beat times
def calculate_beat_times(y, sr, tempo):
    # Calculate params based on input
    T = len(y)/float(sr) # Total audio length in seconds
    seconds_per_beat = 60.0/tempo

    # Start beat at first onset rather than time 0
    # TODO: Let this first onset also be user-supplied for more accurate results
    beat_times = np.arange(detect_first_onset_time(y, sr), T, seconds_per_beat)
    
    return beat_times

# Detect first onset
def detect_first_onset_time(y, sr, hop_length=1024):
    onset_frames = librosa.onset.onset_detect(y, sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(onset_frames)
    return onset_times[0]


# Calculate measure indices in samples
def calculate_measure_samples(y, beat_samples, beats_per_measure):
    max_samples = len(y)
    start_sample = beat_samples[0]
    beat_interval = beat_samples[1] - beat_samples[0]
    measure_interval = beat_interval * beats_per_measure
    if measure_interval >= beat_interval:
        return np.array(beat_samples[::beats_per_measure], dtype="int")
    else:
        beat_indices = np.indices([len(beat_samples)])[0]
        measure_indices = np.indices([len(beat_samples)/beats_per_measure])[0]
        return np.interp(measure_indices, beat_indices/beats_per_measure, beat_samples)


"""Generating clicks for tresillo rhythm at the proper tempo and start time,
   to overlay onto an audio track"""
def generate_rhythm_overlay(rhythm, measure_samples, steps_per_measure, sr):
    # Calculate click interval
    measure_length = measure_samples[1]-measure_samples[0]
    # click_tempo = tempo * (steps_per_measure/float(beats_per_measure))
    # click_interval = 60.0/click_tempo
    measure_length_seconds = librosa.samples_to_time(measure_length, sr=sr)
    click_interval = measure_length_seconds / float(steps_per_measure)

    # Generate click times for single measure
    pulse_times_measure, step_times_measure = generate_rhythm_times(rhythm, click_interval)

    # Generate clicks for single measure
    pulse_clicks_measure, step_clicks_measure = generate_rhythm_clicks(rhythm, click_interval, sr=sr)

    # Concatenate clicks and click times for all measures
    pulse_times, step_times, pulse_clicks, step_clicks = np.array([]), np.array([]), np.array([]), np.array([])
    for s in measure_samples:
        t = float(librosa.samples_to_time(s, sr=sr))
        pulse_clicks = np.hstack((pulse_clicks, pulse_clicks_measure))
        step_clicks = np.hstack((step_clicks, step_clicks_measure))
        pulse_times = np.hstack((pulse_times, pulse_times_measure + t))
        step_times = np.hstack((step_times, step_times_measure + t))

    # Offset clicks by first onset
    pulse_clicks = np.hstack((np.zeros(measure_samples[0]), pulse_clicks))
    step_clicks = np.hstack((np.zeros(measure_samples[0]), step_clicks))
    
    return (pulse_times, step_times, pulse_clicks, step_clicks)

"""Visualizing and hearing the result"""
def overlay_rhythm_onto_audio(rhythm, audio_samples, measure_samples, sr=44100, plt_size=(16,4), click_colors={"measure": "r",
                                                                                                 "pulse": "r",
                                                                                                 "step": "r"}):
    
    # Get overlay data
    pulse_times, step_times, pulse_clicks, step_clicks = generate_rhythm_overlay(rhythm,
                                                                                 measure_samples,
                                                                                 len(rhythm),
                                                                                 sr)
    measure_times = librosa.samples_to_time(measure_samples, sr=sr)
    measure_clicks = librosa.clicks(times=measure_times, sr=sr, click_freq=3000.0, length=len(audio_samples))
    
    # Calculate max length in samples
    available_lengths = [len(audio_samples), len(measure_clicks), len(pulse_clicks), len(step_clicks)]
    length_samples = min(available_lengths)
    
    # Plot original waveform
    plt.figure(figsize=plt_size)
    librosa.display.waveplot(audio_samples, sr=sr, alpha=0.5)
    
    # Plot rhythm clicks
    plt.vlines(measure_times, -1, 1, color=click_colors["measure"])
    plt.vlines(pulse_times, -0.5, 0.5, color=click_colors["pulse"])
    plt.vlines(step_times, -0.25, 0.25, color=click_colors["step"], alpha=0.75)
    
    # Play both clicks together with audio track
    concatenated_audio_samples = ((audio_samples[:length_samples]*2.0)
                                  + (measure_clicks[:length_samples]*0.25)
                                  + (pulse_clicks[:length_samples]*0.25)
                                  + (step_clicks[:length_samples]*0.25))
    audio_display = ipd.Audio(concatenated_audio_samples, rate=sr)
    return audio_display





# ## Part 4 - Time-stretching audio

# Calculate ratios between pulses for two rhythm sequences
# NOTE: This assumes that both rhythm sequences have the same number of pulses! Extra pulses in the longer rhythm will be ignored
def calculate_pulse_ratios(original_rhythm, target_rhythm):
    original_pulse_lengths = calculate_pulse_lengths(original_rhythm)
    target_pulse_lengths = calculate_pulse_lengths(target_rhythm)
    num_pulses = min(len(original_pulse_lengths), len(target_pulse_lengths))
    pulse_ratios = np.array([original_pulse_lengths[i]/float(target_pulse_lengths[i]) for i in range(num_pulses)])
    #if len(pulse_ratios) < len(original_pulse_lengths):  # Add 0s to pulse ratios if there aren't enough
    #    pulse_ratios = np.hstack((pulse_ratios, np.zeros(len(original_pulse_lengths) - len(pulse_ratios))))
    return pulse_ratios


# Modify a single measure
def modify_measure(data, original_rhythm, target_rhythm, stretch_method):
    modified_data = np.array([])
    
    # Define the rhythmic properties we'll use
    original_num_samples = len(data)
    original_num_steps = len(original_rhythm)
    target_num_steps = len(target_rhythm)
    
    # Get indices of steps for measure
    original_step_interval = original_num_samples / float(original_num_steps)
    original_step_indices = np.arange(0, original_num_samples, original_step_interval, dtype="int")
    
    # Get only indices of pulses based on rhythm
    original_pulse_indices = np.array([original_step_indices[i] for i in range(original_num_steps) if original_rhythm[i] > 0])
    
    # Calculate pulse ratios
    pulse_ratios = calculate_pulse_ratios(original_rhythm, target_rhythm)
    
    # Calculate pulse lengths
    original_pulse_lengths = calculate_pulse_lengths(original_rhythm)
    target_pulse_lengths = calculate_pulse_lengths(target_rhythm)
    
    # Concatenate time-stretched versions of rhythm's pulses
    for i,p in enumerate(original_pulse_indices):
        # Get pulse sample data; samples between current and next pulse, or if it's the final pulse,
        # samples between pulse and end of audio
        pulse_start = p
        pulse_stop = len(data)-1
        if i < len(original_pulse_indices)-1:
            pulse_stop = original_pulse_indices[i+1]        
        pulse_samples = data[pulse_start:pulse_stop]

        # Time-stretch this step based on ratio of old to new rhythm length
        # TODO: Try out other methods of manipulation, such as using onset detection in addition to steps and pulses
        if stretch_method == "timestretch":
            pulse_samples = librosa.effects.time_stretch(pulse_samples, pulse_ratios[i])
        elif stretch_method == "euclidean":
            pulse_samples = euclidean_stretch(pulse_samples,
                                              original_pulse_lengths[i],
                                              target_pulse_lengths[min(i, len(target_pulse_lengths)-1)])
        else:
            print("ERROR: Invalid stretch method {}".format(stretch_method))
        
        # Add the samples to our modified audio time series
        modified_data = np.hstack((modified_data, pulse_samples))
    
    # Time-stretch entire measure to maintain original measure length (so that it sounds more natural)
    stretch_multiplier = len(modified_data)/float(len(data))
    modified_data = librosa.effects.time_stretch(modified_data, stretch_multiplier)
    
    return modified_data


# Modify an entire audio track; basically just loops through a track's measures
# and calls modify_measure() on each measure
def modify_track(data, measure_samples, original_rhythm, target_rhythm, stretch_method="timestretch"):
    modified_track_data = np.zeros(measure_samples[0])
    modified_measure_samples = np.array([], dtype="int")
    for i, sample in enumerate(measure_samples[:-1]):
        modified_measure_samples = np.hstack((modified_measure_samples, len(modified_track_data)))
        measure_start = measure_samples[i]
        measure_stop = measure_samples[i+1]
        measure_data = data[measure_start:measure_stop]
        modified_measure_data = modify_measure(measure_data, original_rhythm, target_rhythm, stretch_method)
        modified_track_data = np.hstack((modified_track_data, modified_measure_data))
    return (modified_track_data, modified_measure_samples)


# ## Part 5 - Euclidean stretch


# Euclid's algorithm to find greatest common divisor
def euclid(a, b):
    m = max(a, b)
    k = min(a, b)
    
    if k==0:
        return m
    else:
        return euclid(k, m%k)


# Euclidean stretch for modifying a single pulse (basically time-stretching subdivisions based on Euclidean rhythms)
def euclidean_stretch(pulse_samples, original_pulse_length, target_pulse_length):
    target_pulse_samples = np.array([])
    
    # Return empty samples array if target pulse length < 1
    if target_pulse_length < 1:
        return target_pulse_samples

    # Ensure original pulse rhythm ("opr") has length equal to or less than target_pulse_length

    # ... by using target pulse length
    # original_pulse_length = min(original_pulse_length, target_pulse_length)
    
    # ... by using divisors of original pulse length
    # if original_pulse_length > target_pulse_length:
    #     # print("WARNING: original_pulse_length {} "
    #     #       "is greater than target_pulse_length {}".format(original_pulse_length,
    #     #                                                       target_pulse_length))
    #     for i in range(1, original_pulse_length+1):
    #         opl_new = int(original_pulse_length / float(i))
    #         if opl_new <= target_pulse_length:
    #             original_pulse_length = opl_new
    #             # print("original_pulse_length is now {}".format(original_pulse_length))
    #             break

    # ... by using lowest common multiple as target pulse length
    if original_pulse_length > target_pulse_length:
        # print("Target pulse length before: {}".format(target_pulse_length))
        gcd = euclid(original_pulse_length, target_pulse_length)
        lcm = (original_pulse_length*target_pulse_length) / gcd
        target_pulse_length = lcm
        # print("Target pulse length after: {}".format(target_pulse_length))
        # original_pulse_length = target_pulse_length
    
    opr = np.ones(original_pulse_length, dtype="int")

    # Generate target pulse rhythm ("tpr")
    tpr = bjorklund.bjorklund(pulses=original_pulse_length, steps=target_pulse_length)
    tpr_pulse_lengths = calculate_pulse_lengths(tpr)
    tpr_pulse_ratios = calculate_pulse_ratios(opr, tpr)
    
    # Subdivide (i.e. segment) the pulse based on original pulse length
    pulse_subdivision_step = int(len(pulse_samples) / float(original_pulse_length))
    pulse_subdivision_indices = np.arange(0, len(pulse_samples), pulse_subdivision_step, dtype="int")
    pulse_subdivision_indices = pulse_subdivision_indices[:original_pulse_length]

    # Time-stretch each subdivision based on ratios
    for i,si in enumerate(pulse_subdivision_indices):
        subdivision_start = si
        subdivision_stop = len(pulse_samples) - 1
        if i < len(pulse_subdivision_indices)-1:
            subdivision_stop = pulse_subdivision_indices[i+1]        
        pulse_subdivision_samples = pulse_samples[subdivision_start:subdivision_stop]        
    
        # Stretch the relevant subdivisions based on target pulse rhythm
        pulse_subdivision_samples = librosa.effects.time_stretch(pulse_subdivision_samples, tpr_pulse_ratios[i])
        
        # Concatenate phrase
        target_pulse_samples = np.hstack((target_pulse_samples, pulse_subdivision_samples))
    
    return target_pulse_samples


# ## Part 6 - Fibonacci stretch: implementation and examples


# An end-to-end implementation of Fibonacci stretch
# Example usage:
#     fibonacci_stretch_track("data/imtheone_cropped_chance_60s.mp3",
#                             tempo=162,
#                             original_rhythm=np.array([1,0,0,0,0,1,0,0]),
#                             target_rhythm=np.array([1,0,0,0,0,1,0,0,0,0]),
#                             overlay_clicks=True)
def fibonacci_stretch_track(audio_filepath,
                            sr=44100,
                            original_rhythm=np.array([1,0,0,1,0,0,1,0], dtype="int"),
                            stretch_method="euclidean",
                            stretch_factor=1,
                            target_rhythm=None,
                            tempo=None,
                            beats_per_measure=4,
                            hop_length=1024,
                            overlay_clicks=False,
                            plt_size=(16,4),
                            render_track=True):    
    # Load input audio
    y, sr = librosa.load(audio_filepath, sr=sr)
    
    # Extract rhythm features from audio
    if tempo is None:
        tempo = estimate_tempo(y, sr)
    beat_times = calculate_beat_times(y, sr, tempo)
    beat_samples = librosa.time_to_samples(beat_times, sr=sr)
    measure_samples = calculate_measure_samples(y, beat_samples, beats_per_measure)
    
    # Generate target rhythm
    if target_rhythm is None:
        target_rhythm = fibonacci_scale_rhythm(original_rhythm, stretch_factor)
        
    # Modify the track
    y_modified, measure_samples_modified = modify_track(y, measure_samples,
                                                        original_rhythm, target_rhythm,
                                                        stretch_method="euclidean")

    # Render the track and any plots
    rendered_track = ipd.Audio(y_modified, rate=sr)
    if overlay_clicks:
        rendered_track = overlay_rhythm_onto_audio(target_rhythm, y_modified, measure_samples_modified, sr, plt_size=plt_size)
    else:
        plt.figure(figsize=plt_size)
        librosa.display.waveplot(y_modified, sr=sr)

    # Return rendered track...
    if render_track:
        return rendered_track
    # ... or return modified track and measure samples
    else:
        return (y_modified, measure_samples_modified)


# ================================

# From other nbs

# Calculate stretch ratios for each original step, for use in real-time
def calculate_step_stretch_ratios(original_rhythm, target_rhythm):
    # Original and target pulse lengths
    original_pulse_lengths = list(calculate_pulse_lengths(original_rhythm))
    target_pulse_lengths = list(calculate_pulse_lengths(target_rhythm))

    # Pulse ratios
    # Format pulse ratios so there's one for each step
    pulse_ratios = list(calculate_pulse_ratios(original_rhythm, target_rhythm))
    if len(pulse_ratios) < len(original_pulse_lengths):  # Add 0s to pulse ratios if there aren't enough
        for _ in range(len(original_pulse_lengths) - len(pulse_ratios)):
            pulse_ratios.append(0.0)
    assert(len(pulse_ratios) == len(original_pulse_lengths))
    pulse_ratios_by_step = []
    for i,pulse_length in enumerate(original_pulse_lengths):
        for _ in range(pulse_length):
            pulse_ratios_by_step.append(pulse_ratios[i])

    # Calculate stretch ratios for each original step
    # Adapted from Euclidean stretch
    step_stretch_ratios = []
    for i in range(min(len(original_pulse_lengths), len(target_pulse_lengths))):
        # Pulse lengths
        opl = original_pulse_lengths[i]
        tpl = target_pulse_lengths[i]
        
        # Adjust target pulse length if it's too small
        #if opl > tpl:
        #    tpl = opl
        while opl > tpl:
           tpl *= 2

        # Use steps as original pulse rhythm ("opr")
        opr = [1] * len(original_rhythm)

        # Generate target pulse rhythm ("tpr") using Bjorklund's algorithm
        tpr = bjorklund.bjorklund(pulses=opl, steps=tpl)
        tpr_pulse_lengths = calculate_pulse_lengths(tpr)
        tpr_pulse_ratios = calculate_pulse_ratios(opr, tpr)

        # Scale the tpr pulse ratios by the corresponding ratio from pulse_ratios_by_step
        tpr_pulse_ratios *= pulse_ratios_by_step[i]

        step_stretch_ratios.extend(tpr_pulse_ratios)
        
    # Multiply by stretch multiplier to make sure the length is the same as original
    stretch_multiplier = 1.0 / (sum(step_stretch_ratios) / len(original_rhythm))
    step_stretch_ratios = [r * stretch_multiplier for r in step_stretch_ratios]
    assert(round(sum(step_stretch_ratios) / len(original_rhythm), 5) == 1)  # Make sure it's *close enough* to original length.
    
    return step_stretch_ratios
