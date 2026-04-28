# Research notes for Topic 2

## Scope decision

The project asks for spectral analysis of at least four voiced consonant letters
and a program that detects the presence of selected letters in spoken words.
Without local MATLAB, the workflow is implemented in Python.  The closest stable
real-data substitute is the TensorFlow mini Speech Commands subset: short,
16-kHz, one-second spoken words with balanced labels.

## Acoustic model

- Voiced consonants preserve periodic vocal-fold excitation, so short-time
  spectra should contain voicing energy and harmonic/formant-like structure.
- Different manners of articulation motivate time-localized spectra:
  - /d/ and /g/: voiced stops; transient release plus following vowel transition.
  - /l/ and /r/: liquids; formant-like spectral envelope.
  - /n/: nasal; lower-frequency resonance and possible antiresonance effects.
- A whole-word FFT is too crude because initial consonant cues are short and are
  mixed with vowel energy.  STFT plus frame features is the correct course-level
  representation.

## Feature choice

- STFT/spectrogram: visual explanation and time-frequency localization.
- MFCC: compact spectral-envelope representation used historically in speech
  recognition, based on mel filterbank log energies and DCT.
- Spectral centroid, bandwidth, rolloff, RMS, and zero-crossing rate: interpretable
  descriptors for the report and a lightweight classical baseline.

## Detector choice

- Classical baseline: standardized MFCC/spectral features plus nearest-centroid
  cosine matching.  This is simple, explainable, and close to a template-matching
  detector.
- AI redo: small MLP classifier on the same signal-processing features.  This
  satisfies the project bonus direction without hiding the signal-processing
  front end behind a full black-box ASR system.

## Main empirical conclusion

On the fixed 8000-file real-data experiment, the classical detector reaches
52.25% seven-class accuracy, while the MLP reaches 76.30%.  For the four required
letters /d/, /g/, /l/, /n/, the AI method improves every one-vs-rest F1 score.
This supports the report's conclusion: spectral features are necessary for
interpretable consonant analysis, but nonlinear data-driven classification is
more robust to speaker and timing variation.
