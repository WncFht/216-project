# Dataset replacement survey for voiced-consonant detection

Date: 2026-04-25.

The original short-command setup was too narrow: it used isolated one-second
keywords and could easily fit word identity rather than voiced-consonant
presence.  I therefore compared a few broader options and kept the final main
experiment at the word level with Speech Commands v0.02.

## Selection criteria

For this project, a better replacement dataset should satisfy most of these
conditions:

1. Real speech rather than synthetic examples.
2. More lexical diversity than a tiny keyword list.
3. Enough speakers to justify speaker-independent evaluation.
4. Reasonable local reproducibility before the April 29 deadline.
5. Preferably phone-aligned, but transcript-only data are still acceptable if
   they clearly improve over the short-word baseline.

## Candidate comparison

| Dataset | Official facts | Why it is relevant | Main drawback for this project | Decision |
|---|---|---|---|---|
| Speech Commands | 16 kHz keyword-recognition benchmark with short isolated commands. | Very easy starting point; good for the first prototype and CNN sanity check. | Too short and too few lexical patterns; consonant labels are only word proxies. | Keep only as prototype / baseline. |
| Speech Commands v0.02 | Still word-level, but with a larger and more varied word set than the tiny prototype subset. | Keeps the task close to the assignment, stays reproducible, and supports multi-label voiced-consonant detection on real words. | Still not phone-level segmentation. | Chosen current main dataset. |
| TIMIT | 630 speakers from 8 major dialect regions, with time-aligned orthographic, phonetic, and word transcriptions. | Best scientific match for true voiced-consonant analysis and phone-level detection. | Requires LDC access, so it is not freely reproducible in the current environment. | Ideal gold-standard future dataset. |
| VCTK | 110 English speakers, about 400 sentences each, prompts selected for contextual and phonetic coverage. | Free multi-speaker read speech with much broader phonetic coverage than keyword datasets. | Larger download and still no ready-made phone timestamps in the default release. | Strong future extension. |
| LibriSpeech / Mini LibriSpeech | 16 kHz read English speech derived from audiobooks. | Good supplementary comparison if we want longer utterances later. | Changes the task from word-level detection to sentence-level presence detection. | Supplementary, not the main route. |
| Common Voice | Large crowd-sourced speech corpus with broad accent diversity and transcripts. | Good robustness dataset if we later want accent and channel variability. | Transcript normalization and audio cleaning are heavier; no phone alignment by default. | Keep as future robustness data. |

## Chosen replacement

Use Speech Commands v0.02 as the main data source:

- keep the word-level setup;
- expand from a tiny prototype subset to a larger multi-word set;
- run the main experiments on the 25-word multi-label subset;
- preserve the assignment framing while avoiding sentence-level drift.

Current run size:

- 25 words
- up to 550 examples per word
- 16500 utterances total
- 2473 speakers

Label generation:

1. Read each selected word recording.
2. Assign all voiced-consonant labels that the word pronunciation supports.
3. Train/test split by speaker hash.

This choice keeps the task exactly in the report scope: word-level voiced-consonant detection.  It is broader than the tiny prototype set, but much cleaner than sentence-level presence detection.

## Source ledger

- Speech Commands paper: https://arxiv.org/abs/1804.03209
- TensorFlow dataset page: https://www.tensorflow.org/datasets/catalog/speech_commands
- TIMIT LDC page: https://catalog.ldc.upenn.edu/LDC93S1
- VCTK corpus page: https://datashare.ed.ac.uk/handle/10283/3443
- LibriSpeech official OpenSLR page: https://www.openslr.org/12
- Mini LibriSpeech OpenSLR page: https://www.openslr.org/31/
- CMU Pronouncing Dictionary: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
- Mozilla Common Voice: https://commonvoice.mozilla.org/
