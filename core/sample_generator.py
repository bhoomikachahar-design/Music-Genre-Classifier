"""
Sample Audio Generator for Music Genre Classifier & Explorer
Synthesizes demonstration audio clips for key musical genres so users
can test the application immediately without uploading their own files.
"""

import os
import numpy as np
import soundfile as sf

def generate_samples(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    sr = 22050
    duration = 5.0  # 5 seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 1. Classical: Elegant piano-like arpeggio (C major - G major - A minor chord tones with gentle envelope)
    classical = np.zeros_like(t)
    notes_classical = [261.63, 329.63, 392.00, 523.25, 392.00, 329.63, 261.63, 220.00, 261.63, 329.63]
    note_dur = duration / len(notes_classical)
    for i, freq in enumerate(notes_classical):
        t_start = int(i * note_dur * sr)
        t_end = int((i + 1) * note_dur * sr)
        t_seg = t[t_start:t_end] - (i * note_dur)
        # decaying acoustic tone
        env = np.exp(-3.5 * t_seg)
        tone = (
            np.sin(2 * np.pi * freq * t_seg) +
            0.5 * np.sin(2 * np.pi * (2 * freq) * t_seg) +
            0.2 * np.sin(2 * np.pi * (3 * freq) * t_seg)
        ) * env
        classical[t_start:t_end] += tone
    classical = classical / (np.max(np.abs(classical)) + 1e-6)
    sf.write(os.path.join(output_dir, "sample_classical.wav"), classical, sr)

    # 2. Rock: Driving electric guitar power-chord rhythm with distortion and kick/snare transients
    rock = np.zeros_like(t)
    bpm_rock = 135
    beat_dur = 60.0 / bpm_rock
    for beat in range(int(duration / beat_dur)):
        t_b_start = int(beat * beat_dur * sr)
        t_b_end = min(len(t), int((beat + 1) * beat_dur * sr))
        t_seg = t[t_b_start:t_b_end] - (beat * beat_dur)
        # Power chord: E2 (82.4 Hz), B2 (123.5 Hz), E3 (164.8 Hz)
        raw_chord = (
            np.sin(2 * np.pi * 82.41 * t_seg) +
            0.8 * np.sin(2 * np.pi * 123.47 * t_seg) +
            0.9 * np.sin(2 * np.pi * 164.81 * t_seg)
        )
        # Overdrive / distortion
        distorted = np.tanh(4.0 * raw_chord) * np.exp(-1.5 * t_seg)
        # Add drum punch
        if beat % 2 == 0:
            # kick
            drum = np.sin(2 * np.pi * 60.0 * np.exp(-15 * t_seg) * t_seg) * np.exp(-8 * t_seg)
        else:
            # snare noise burst
            drum = np.random.normal(0, 0.4, len(t_seg)) * np.exp(-12 * t_seg)
        rock[t_b_start:t_b_end] += distorted + drum
    rock = rock / (np.max(np.abs(rock)) + 1e-6)
    sf.write(os.path.join(output_dir, "sample_rock.wav"), rock, sr)

    # 3. Jazz: Smooth swing rhythm with 7th chords (D minor 7 - G7) and walking bass
    jazz = np.zeros_like(t)
    bpm_jazz = 110
    swing_dur = 60.0 / bpm_jazz
    for step in range(int(duration / (swing_dur / 2))):
        s_start = int(step * (swing_dur / 2) * sr)
        s_end = min(len(t), int((step + 1) * (swing_dur / 2) * sr))
        t_seg = t[s_start:s_end] - (step * (swing_dur / 2))
        # Dm7: D (146.8), F (174.6), A (220.0), C (261.6)
        jazz_chord = (
            np.sin(2 * np.pi * 146.83 * t_seg) +
            0.6 * np.sin(2 * np.pi * 174.61 * t_seg) +
            0.5 * np.sin(2 * np.pi * 220.00 * t_seg) +
            0.4 * np.sin(2 * np.pi * 261.63 * t_seg)
        ) * np.exp(-2.0 * t_seg)
        # ride cymbal ting
        cymbal = np.random.normal(0, 0.15, len(t_seg)) * np.exp(-25 * t_seg)
        jazz[s_start:s_end] += jazz_chord + cymbal
    jazz = jazz / (np.max(np.abs(jazz)) + 1e-6)
    sf.write(os.path.join(output_dir, "sample_jazz.wav"), jazz, sr)

    # 4. Disco / Pop: 124 BPM four-on-the-floor beat with synth bass and hi-hats
    disco = np.zeros_like(t)
    bpm_disco = 124
    quarter = 60.0 / bpm_disco
    for q in range(int(duration / quarter)):
        q_start = int(q * quarter * sr)
        q_end = min(len(t), int((q + 1) * quarter * sr))
        t_seg = t[q_start:q_end] - (q * quarter)
        # Four on the floor Kick drum
        kick = np.sin(2 * np.pi * 120.0 * np.exp(-25 * t_seg) * t_seg) * np.exp(-10 * t_seg)
        # Off-beat open hi-hat
        half_pt = len(t_seg) // 2
        hihat = np.zeros_like(t_seg)
        hihat[half_pt:] = np.random.normal(0, 0.35, len(t_seg) - half_pt) * np.exp(-18 * (t_seg[half_pt:] - t_seg[half_pt]))
        # Synth bass tone
        synth = np.sin(2 * np.pi * 110.0 * t_seg) * np.exp(-3 * t_seg)
        disco[q_start:q_end] += kick + hihat + synth
    disco = disco / (np.max(np.abs(disco)) + 1e-6)
    sf.write(os.path.join(output_dir, "sample_disco.wav"), disco, sr)

    # 5. Metal: Aggressive fast double-kick 160 BPM with high distortion harmonics
    metal = np.zeros_like(t)
    bpm_metal = 160
    eighth = (60.0 / bpm_metal) / 2
    for e_idx in range(int(duration / eighth)):
        e_start = int(e_idx * eighth * sr)
        e_end = min(len(t), int((e_idx + 1) * eighth * sr))
        t_seg = t[e_start:e_end] - (e_idx * eighth)
        # Low E chug (41.2 Hz)
        chug = np.sign(np.sin(2 * np.pi * 41.20 * t_seg)) * np.exp(-6 * t_seg)
        # Fast double kick drum
        kick = np.sin(2 * np.pi * 90.0 * np.exp(-20 * t_seg) * t_seg) * np.exp(-12 * t_seg)
        metal[e_start:e_end] += chug * 0.7 + kick * 0.5
    metal = metal / (np.max(np.abs(metal)) + 1e-6)
    sf.write(os.path.join(output_dir, "sample_metal.wav"), metal, sr)

    print(f"Generated 5 demonstration sample clips in: {output_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    samples_dir = os.path.join(base_dir, "samples")
    generate_samples(samples_dir)
