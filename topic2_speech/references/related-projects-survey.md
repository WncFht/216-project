# Topic 2 related-project survey: traditional vs AI methods

Collected on 2026-04-24 for VE216 Project Topic 2.  The goal is not to list every
speech-recognition paper, but to identify project patterns that can directly
improve our voiced-consonant detector.

## 0. Our current baseline for comparison

Current project path: `216/project/topic2_speech/`

- Data: TensorFlow `mini_speech_commands`, 8000 one-second WAV files across 8
  command words.
- Label design: command word initial consonants mapped to `/d/`, `/g/`, `/l/`,
  `/n/`, `/r/`, `/j/`, and `other`.
- Traditional method: STFT + MFCC + spectral statistics + nearest-centroid cosine
  matching.
- AI method: the same handcrafted features fed into an MLP classifier.
- Current result: classical 52.25% seven-class accuracy; MLP 76.30%.

This is a clean first version, but it is closer to keyword-initial-consonant
classification than true phone-level consonant detection.  The projects below
show how to improve that gap.

## 1. Traditional / signal-processing projects and papers

| Project / source | What it does | Why it matters for us | Direct adaptation |
|---|---|---|---|
| Davis & Mermelstein (1980), MFCC paper | Compares parametric speech representations for word recognition; establishes mel-frequency cepstral representation as a compact spectral-envelope feature. | Our current features use MFCCs.  This is the correct historical anchor for why MFCC is used instead of raw FFT bins. | Add delta and delta-delta MFCCs; compare MFCC-only vs MFCC + spectral stats in an ablation table. |
| Rabiner (1989), HMM tutorial | Classic tutorial on Hidden Markov Models and speech recognition.  HMMs model sequential state transitions and frame likelihoods. | Our nearest-centroid detector ignores temporal state structure.  HMM is the natural traditional upgrade for consonant onset, closure, release, and vowel transition. | Train one small left-to-right HMM per target consonant/word using MFCC frames, or use `hmmlearn`/custom Viterbi for a simplified version. |
| TIMIT corpus (LDC93S1) | About five hours of English read speech with time-aligned orthographic, phonetic, and word transcriptions; 630 speakers across eight American English dialect regions; 16 kHz, 16-bit audio. | This is the cleanest way to turn our word-level labels into true phone-level labels.  It provides actual `/d/`, `/g/`, `/l/`, `/n/` time intervals, not just word labels. | Replace command-word initial labels with `.phn` intervals; train/test on cropped consonant phone segments; report phone-level detection. |
| Wilkinson & Russell (ICSLP 2002), formant + MFCC on TIMIT | Integrates formant frequency estimates and conventional MFCC data for phone recognition.  The paper reports that confidence-weighted formant information can help, while naive formant replacement can hurt. | Our report currently mentions formant-like spectral patterns but does not measure formants.  This paper gives a traditional route to make that discussion experimental. | Estimate F1/F2/F3 or LPC peaks for voiced segments; add confidence-weighted formant features; compare MFCC vs MFCC+formants. |
| Kaldi TIMIT recipe | A reproducible TIMIT phone-recognition recipe: monophone, triphone, LDA+MLLT, SAT, SGMM, and DNN hybrids.  The recipe trains with 48 phones and evaluates on 39 phones. | It is a mature reference for evaluation protocol and train/dev/test split.  It also shows how traditional acoustic modeling improves from monophone to SAT/SGMM. | Do not fully port Kaldi for this course project, but copy the evaluation idea: separate speakers, use 61/48/39-phone mapping, report per-phone confusion and phone error rate. |

### Traditional-method lessons

1. **Use frame sequences, not just utterance summaries.**  Consonants are short;
   the release and transition region may be decisive.
2. **Add temporal derivatives.**  Delta and delta-delta MFCCs are a low-cost way
   to capture movement in the spectral envelope.
3. **Use phone-level labels if possible.**  Word-level labels hide the actual
   consonant boundary.
4. **Formants are useful but fragile.**  They should be confidence-weighted or
   used as auxiliary features, not blindly substituted for MFCCs.
5. **Speaker-independent splitting matters.**  Random file-level split can leak
   speaker characteristics; speaker-level split is more honest.

## 2. AI / deep-learning projects and papers

| Project / source | What it does | Why it matters for us | Direct adaptation |
|---|---|---|---|
| TensorFlow Simple Audio Recognition tutorial | Uses the mini Speech Commands data set, pads/clips audio to 16000 samples, converts audio to spectrogram images, and trains a small CNN. | It exactly matches our current data source.  It is the simplest next step beyond MLP-on-features. | Add a CNN-on-spectrogram model and compare it to our MLP. |
| TensorFlow Datasets `speech_commands` | Official TFDS wrapper for the full Speech Commands task; designed for small keyword-spotting models and includes train/validation/test splits. | Our current mini subset is small; full TFDS gives more words, unknown class handling, and background noise. | Switch from `mini_speech_commands` to full `speech_commands`; keep selected voiced targets and add unknown/background classes. |
| Honk: PyTorch CNNs for keyword spotting | PyTorch reimplementation of TensorFlow CNN keyword-spotting models; includes CNN/residual models, MFCC or PCEN preprocessing, speaker grouping, noise mixing, time shift, and custom keyword support. | This is the most directly relevant AI project for our implementation.  It shows how to train better models without abandoning interpretability. | Borrow data augmentation: noise mixing, time shifting, silence/unknown probability, speaker grouping; implement a small residual CNN. |
| Microsoft ELL keyword spotter tutorial | Trains an audio keyword spotter with PyTorch on Speech Commands and targets deployment on small low-power devices. | Useful if we want to discuss practical deployment and computational constraints, not just accuracy. | Add model-size / inference-time comparison; frame the project as an embedded detector. |
| wav2vec 2.0 | Self-supervised speech representation model; pretrains on unlabeled speech and fine-tunes with limited labeled data. | This is the modern AI direction.  It is overkill for VE216, but excellent for a final “future work” section. | Use pretrained wav2vec2 embeddings as features, then train a small classifier for `/d/`, `/g/`, `/l/`, `/n/`; compare against MFCC. |
| Deep residual small-footprint KWS models, linked from Honk | Uses residual CNNs for small-footprint keyword spotting. | Our MLP result is already good, but a small CNN/residual model should improve accuracy and is still project-scale. | Add `res8-narrow` or a small 2D CNN over log-mel spectrograms; report accuracy and confusion matrix. |

### AI-method lessons

1. **CNNs are the best next AI baseline.**  Our current MLP flattens features;
   CNNs preserve local time-frequency structure.
2. **Augmentation is critical.**  Time shift, noise mixing, random gain, and
   background-noise classes make keyword/consonant detectors robust.
3. **Speaker grouping avoids leakage.**  Honk explicitly exposes speaker grouping;
   we should do the same by parsing Speech Commands file IDs.
4. **Full Speech Commands supports unknown/background classes.**  This makes the
   detection task closer to real deployment.
5. **wav2vec2 is a feature extractor upgrade, not the first thing to implement.**
   It is strong but less transparent and heavier than a CNN.

## 3. Source ledger

- Pete Warden, “Speech Commands: A Dataset for Limited-Vocabulary Speech
  Recognition”, arXiv:1804.03209.  https://arxiv.org/abs/1804.03209
- TensorFlow tutorial, “Simple audio recognition: Recognizing keywords”.
  https://www.tensorflow.org/tutorials/audio/simple_audio
- TensorFlow Datasets catalog, `speech_commands`.
  https://www.tensorflow.org/datasets/catalog/speech_commands
- Tang & Lin, “Honk: A PyTorch Reimplementation of Convolutional Neural Networks
  for Keyword Spotting”, arXiv:1710.06554.  https://arxiv.org/abs/1710.06554
- Honk GitHub README.  https://github.com/castorini/honk
- Microsoft ELL, “Training an audio keyword spotter with PyTorch”.
  https://microsoft.github.io/ELL/tutorials/Training-audio-keyword-spotter-with-pytorch/
- LDC TIMIT catalog entry.  https://catalog.ldc.upenn.edu/LDC93S1
- Rabiner, “A Tutorial on Hidden Markov Models and Selected Applications in
  Speech Recognition”.  https://www.cs.cornell.edu/courses/cs481/2004fa/rabiner.pdf
- Davis & Mermelstein, “Comparison of parametric representations for monosyllabic
  word recognition in continuously spoken sentences”.  https://doi.org/10.1109/TASSP.1980.1163420
- Wilkinson & Russell, “Improved Phone Recognition on TIMIT using Formant
  Frequency Data and Confidence Measures”.  https://www.isca-archive.org/icslp_2002/wilkinson02_icslp.pdf
- Kaldi TIMIT recipe.  https://github.com/kaldi-asr/kaldi/tree/master/egs/timit/s5
- Baevski et al., “wav2vec 2.0: A Framework for Self-Supervised Learning of
  Speech Representations”.  https://arxiv.org/abs/2006.11477
