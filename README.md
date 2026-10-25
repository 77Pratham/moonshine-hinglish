# Hinglish speech-to-text: fine-tuning Moonshine Tiny

**Output format: romanized Hinglish (Latin script).**

Moonshine Tiny (27M parameters) fine-tuned on 100 hours of human-transcribed spontaneous Hindi speech, with Devanagari transcripts romanized to Latin script. It runs offline on a laptop CPU at about 12x real time.

**Links:** Demo video: https://youtu.be/8uWZ3VgDSzg | 
Model (private HF): https://huggingface.co/PrathamR777/moonshine-tiny-hinglish | 
Dataset (private HF): https://huggingface.co/datasets/PrathamR777/hinglish-moonshine (read-only token provided by email)

## Results

Held-out clips, normalized (lowercased, punctuation removed) romanized text. 300 test clips: 166 from speakers never seen in training (the headline number) and 134 from speakers that also appear in training.

| Model | WER (166 speaker-disjoint) | CER (disjoint) | WER (all 300) | CER (all 300) | Real-time factor (laptop CPU) |
|---|---:|---:|---:|---:|---:|
| Moonshine Tiny, not fine-tuned (English-only reference) | 140.4% | 94.9% | 155.5% | 104.6% | n/a* |
| **Fine-tuned, fp32** | **39.5%** | **20.9%** | **36.5%** | **18.8%** | **0.082** |
| Fine-tuned, INT8 dynamic quantization | 41.5% | 21.7% | 38.5% | 19.6% | 0.106 |

*The baseline is English-only and hallucinates English text on Hindi audio (WER above 100% from inserted words); it is shown as a reference point for the starting model only. Its speed is not comparable because it runs into the token cap.

INT8 gave no speedup on this CPU and cost about 2 WER points, so the demo uses fp32. Real-time factor is compute time divided by audio time, measured once on a laptop CPU, so treat it as approximate. The disjoint subset is only about 17 minutes of audio, so small differences are within noise.

## Data

- **Source:** AI4Bharat IndicVoices, Hindi config, https://huggingface.co/datasets/ai4bharat/IndicVoices (CC BY 4.0, human-transcribed).
  Paper: arXiv:2403.01926.
- **Used:** 51,898 training clips (100.0 h, 1,818 speakers) and 881 test clips (1.5 h) from the dataset's `valid` split.
- **Final dataset (private):** https://huggingface.co/datasets/PrathamR777/hinglish-moonshine
  (columns: audio 16 kHz FLAC, `text` romanized, `orig_text` Devanagari, `dur`, `speaker_id`).

715 of the 881 test clips come from speakers who also appear in training; 166 clips are speaker-disjoint, and I report both.

## Preprocessing (`prep_data.py`, `romanize.py`)

- Used the dataset's cleaned `text` column (spelling normalized, noise tags removed).
- Kept clips of 1-20 s; dropped clips with digits and clips with implausible characters per second (outside 3-25).
- Devanagari to Latin with a deterministic rule-based romanizer: schwa deletion, a small English-loanword dictionary, lowercasing, punctuation removal. No machine translation.
- The data was prepared on Modal (CPU).

## Training (`train.py`)

Moonshine Tiny (`UsefulSensors/moonshine-tiny`), effective batch size 64 (16 x 4 gradient accumulation), learning rate 1e-4 with cosine decay, 5,000 steps (about 6 passes over the data), fp16, one NVIDIA T4 on Kaggle, 71 minutes. Training loss fell from about 2.4 to about 0.37.

Modal's GPU credits required a payment method I could not complete, so training ran on Kaggle's free GPU.

## Inference / on-device (`demo_live.py`)

Microphone demo, running offline from a local copy of the model on a laptop CPU: `python demo_live.py` (or `python demo_live.py clip.wav`).

Typical latency is 0.15-0.3 s for 6 s of audio. **Not tested on a phone; ONNX export was not completed.**

## Limitations

- Romanized Hinglish has no standard spelling, so WER overstates real errors; CER is more informative.
- English words written in Devanagari in the source transcripts are romanized phonetically (for example "kol", "stail"), and the model reproduces that convention. Only a small loanword dictionary maps words to English spellings, so English-heavy speech appears phonetically.
- Rare words, proper nouns, and English loanwords are the main error sources; occasional repeated words appear.
- 715 of 881 test clips share speakers with training, so the 166-clip speaker-disjoint result is the reliable one.
- Small model; accuracy drops on noisy audio. Tested on a laptop CPU only.

## Reproduce

1. `modal run modal_prep.py --hours 100`  (needs a Hugging Face token; builds and uploads the dataset)
2. `python train.py --steps 5000`  (run on a GPU; I used Kaggle, see `kaggle_notebook.ipynb`)
3. `python eval_local.py PrathamR777/moonshine-tiny-hinglish`
