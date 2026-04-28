# Topic 2 improvement plan

This document explains how the current VE216 Topic 2 project can be improved.
It is written as an engineering roadmap: each section states what is weak now,
what to change, how to implement it, and what it would prove in the report.

## Current state

The current project is already complete enough for a first submission:

- real audio data are used;
- four required voiced consonants `/d/`, `/g/`, `/l/`, `/n/` are analyzed;
- spectrograms, mean spectra, confusion matrices, and binary metrics are in the
  report;
- a traditional detector and an AI detector are compared;
- the LaTeX report compiles successfully.

The main weakness is that the current labels are word-level initial-consonant
labels.  For example, `down` is treated as `/d/`, but the detector sees the
whole word, including the vowel and final nasal.  This makes the experiment a
reasonable course project, but not yet a strict phone-level consonant detector.

## 1. Data and label improvements

### 1.1 Move from word-level labels to phone-level labels

**Problem now:** the model may learn word identity instead of consonant identity.
For example, it may classify `down` by the vowel-plus-nasal pattern rather than
only the `/d/` onset.

**Best fix:** use TIMIT, because it has hand-verified time-aligned phonetic
transcriptions.

**Implementation path:**

1. Obtain TIMIT if license access is available.
2. Parse each `.phn` file.
3. Extract only segments with labels such as `d`, `g`, `l`, `n`, `r`, `y`, plus
   negative classes.
4. Pad/crop each phone segment to a fixed window, for example 120 ms centered on
   the phone midpoint, or use variable-length sequence models.
5. Train/test by speaker, not by file.

**Report value:** this changes the project from “word initial consonant
classification” to true “voiced consonant identification and detection.”

### 1.2 Add custom recordings for the exact prompt examples

**Problem now:** the project statement gives examples like detecting `b` in
`book` or `boy`, but our data set does not include those words.

**Fix:** record a small supplementary data set:

- target `/b/`: `book`, `boy`, `back`, `bad`, `big`;
- target `/d/`: `down`, `day`, `dog`, `door`;
- target `/g/`: `go`, `good`, `game`, `green`;
- target `/l/`: `left`, `late`, `light`, `low`;
- target `/n/`: `no`, `name`, `nine`, `new`.

Use 5-10 speakers if possible, 3 repetitions per word.  If time is short, record
one speaker and clearly mark it as a qualitative demo.

**Report value:** directly answers the prompt’s “detect presence in words”
requirement and gives a demo section.

### 1.3 Use speaker-independent splitting

**Problem now:** Speech Commands filenames encode speaker IDs, but the current
split is random at file level.  The same speaker can appear in both train and
test.

**Fix:** parse the speaker hash before `_nohash_` and split by speaker ID.

**Implementation sketch:**

```python
speaker = wav_path.name.split('_nohash_')[0]
# group train/test split by speaker, not by individual file
```

**Report value:** gives a more honest generalization metric.

## 2. Traditional-method improvements

### 2.1 Add delta and delta-delta MFCCs

**Problem now:** the classical feature vector includes mean/std and flattened
MFCCs, but not explicit dynamic derivatives.

**Fix:** compute first and second temporal differences:

\[
\Delta c_t = \frac{\sum_{n=1}^N n(c_{t+n}-c_{t-n})}{2\sum_{n=1}^N n^2}.
\]

Then compare:

- MFCC only;
- MFCC + delta;
- MFCC + delta + delta-delta;
- MFCC + delta + spectral statistics.

**Report value:** gives an ablation table and stronger signal-processing theory.

### 2.2 Add LPC / formant features

**Problem now:** the report discusses formant-like structure but does not compute
formants.

**Fix:** estimate LPC roots per voiced frame and derive F1/F2/F3 candidates.
Because formant extraction is unreliable, also compute a confidence score based
on peak sharpness or frame energy.

**Ablation:**

- MFCC baseline;
- formants only;
- MFCC + formants;
- MFCC + confidence-weighted formants.

**Report value:** connects the project to acoustic phonetics, especially for
nasals, liquids, and glides.

### 2.3 Replace nearest centroid with DTW template matching

**Problem now:** centroid matching ignores temporal alignment.

**Fix:** keep a few template MFCC sequences per consonant/word and use Dynamic
Time Warping distance.

**Why it helps:** DTW can align slow and fast pronunciations without needing a
neural model.

**Report value:** gives a stronger traditional baseline than nearest centroid.

### 2.4 Add HMM sequence modeling

**Problem now:** consonants have temporal phases, but the classical detector has
no state model.

**Fix:** train a small left-to-right HMM per target.  Each state emits Gaussian
MFCC features.  For voiced stops, states can roughly correspond to closure,
release, and transition; for sonorants, states can model onset, middle, and
transition.

**Report value:** this is the most canonical traditional speech-recognition
upgrade.

## 3. AI-method improvements

### 3.1 Add a CNN on log-mel spectrograms

**Problem now:** the MLP receives flattened handcrafted features.  It does not
explicitly exploit local time-frequency structure.

**Fix:** build a small 2D CNN:

- input: log-mel spectrogram, shape roughly `(time_frames, 40)`;
- Conv2D -> ReLU -> MaxPool;
- Conv2D -> ReLU -> GlobalAveragePooling;
- Dense classifier.

**Expected outcome:** should outperform the MLP or at least reduce confusion
between `/g/`, `/l/`, and `/n/`.

**Report value:** directly mirrors the TensorFlow audio tutorial and Honk-style
keyword spotting.

### 3.2 Add data augmentation

**Problem now:** the model sees clean one-second clips only.

**Fix:** during training, randomly apply:

- time shift: +/- 100 ms;
- random gain: e.g. 0.7 to 1.3;
- additive background noise from Speech Commands `_background_noise_`;
- small Gaussian noise;
- random crop around loudest region.

**Report value:** demonstrates robustness and gives a real deployment discussion.

### 3.3 Use wav2vec2 embeddings as a modern baseline

**Problem now:** MFCCs are compact and interpretable but limited.

**Fix:** use a pretrained wav2vec2 model as a frozen feature extractor, then train
only a small classifier on top.

**Why not first:** it is heavier, harder to explain in a VE216 signals-and-systems
report, and may distract from core course concepts.

**Report value:** good for an “AI extension” section or bonus appendix.

### 3.4 Multi-task learning: word + consonant

**Problem now:** word identity and consonant identity are entangled.

**Fix:** train a model with two output heads:

- head 1: command word;
- head 2: consonant class.

If the consonant head performs well while word prediction is separated, the
claim “we detect consonant cues” becomes stronger.

## 4. Evaluation improvements

### 4.1 Add ablation table

Current report has two rows: classical vs AI.  Add a table like:

| Features / model | Accuracy | /d/ F1 | /g/ F1 | /l/ F1 | /n/ F1 |
|---|---:|---:|---:|---:|---:|
| MFCC mean/std + centroid | ... | ... | ... | ... | ... |
| MFCC sequence + centroid | ... | ... | ... | ... | ... |
| MFCC + delta + DTW | ... | ... | ... | ... | ... |
| MLP | 76.30% | 0.765 | 0.624 | 0.679 | 0.635 |
| CNN spectrogram | ... | ... | ... | ... | ... |

### 4.2 Add noise robustness curves

Evaluate each model under controlled SNR:

- clean;
- 20 dB;
- 10 dB;
- 5 dB;
- 0 dB.

Plot accuracy vs SNR.  This would make the comparison between traditional and AI
methods much more convincing.

### 4.3 Add per-class error analysis

For each confused pair, inspect 5 example files:

- `/g/` predicted as `/n/`;
- `/n/` predicted as `other`;
- `/l/` predicted as `/j/`.

Add a small qualitative table: waveform/spectrogram reason, likely cause, and
whether the error is acoustically understandable.

### 4.4 Report confidence calibration

For AI models, report confidence distribution:

- correct high-confidence predictions;
- wrong high-confidence predictions;
- uncertain predictions.

This helps explain when a detector can abstain instead of making a forced wrong
classification.

## 5. Report-writing improvements

### 5.1 Make the limitation explicit but positive

Current limitation should be phrased as:

> This project uses word-level command labels as a controlled proxy for initial
> consonant detection.  The next step is phone-level segmentation using TIMIT or
> manual annotation.

This is honest and does not weaken the project; it shows methodological clarity.

### 5.2 Add one methods diagram

A simple block diagram would improve readability:

```text
WAV -> normalize -> frame/window -> STFT/log-mel -> MFCC/statistics
    -> classical detector / AI detector -> metrics + confusion matrix
```

### 5.3 Add a “why this is still signals and systems” paragraph

Emphasize:

- STFT is the central transform;
- window length trades time resolution and frequency resolution;
- MFCC is a transformed spectral-envelope representation;
- classification is downstream of signal representation.

## 6. Recommended next implementation order

If we keep the current data and have limited time, do these in order:

1. **Speaker-independent split** - low effort, high credibility.
2. **Delta / delta-delta MFCC ablation** - low effort, improves traditional side.
3. **CNN on log-mel spectrograms** - medium effort, likely improves AI side.
4. **Noise robustness curve** - medium effort, excellent report figure.
5. **Custom recordings for `book` / `boy` demo** - medium effort, matches prompt.
6. **TIMIT phone-level version** - best scientific upgrade, but depends on data
   access and time.

## 7. Minimum strong final version

A stronger final report before submission should include:

- current report PDF;
- one new ablation table;
- speaker-independent split result;
- CNN result;
- noise robustness plot;
- one paragraph clarifying word-level vs phone-level detection.

That would turn the project from a working prototype into a well-supported
course report.
