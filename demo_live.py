import os, sys, math, time
import torch, soundfile as sf
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
from transformers import AutoProcessor, MoonshineForConditionalGeneration

REPO = "PrathamR777/moonshine-tiny-hinglish"
LOCAL = "moonshine-hinglish-model"

if not os.path.exists(LOCAL):   # one-time download (needs HF_TOKEN set)
    tok = os.environ["HF_TOKEN"]
    AutoProcessor.from_pretrained(REPO, token=tok).save_pretrained(LOCAL)
    MoonshineForConditionalGeneration.from_pretrained(REPO, token=tok).save_pretrained(LOCAL)

proc = AutoProcessor.from_pretrained(LOCAL)
model = MoonshineForConditionalGeneration.from_pretrained(LOCAL).eval()
if os.environ.get("INT8"):
    model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    print("running INT8 (dynamic quantization) on CPU")
else:
    print("running fp32 on CPU")

def transcribe(a):
    f = proc.feature_extractor(a, sampling_rate=16000, return_tensors="pt")
    t = time.time()
    with torch.inference_mode():
        ids = model.generate(**f, max_new_tokens=max(8, math.ceil(len(a) / 16000 * 10)))
    text = proc.tokenizer.decode(ids[0], skip_special_tokens=True)
    return text, time.time() - t

if len(sys.argv) > 1:   # python demo_live.py clip.wav
    a, sr = sf.read(sys.argv[1], dtype="float32")
    if a.ndim > 1: a = a.mean(axis=1)
    assert sr == 16000, "use a 16 kHz wav"
    text, dt = transcribe(a)
    print(text, f"   [{dt:.2f}s for {len(a)/16000:.1f}s audio]")
else:
    import sounddevice as sd
    SECS = 6
    while True:
        input(f"\nPress Enter, then speak for {SECS} seconds (Ctrl+C to quit)... ")
        a = sd.rec(SECS * 16000, samplerate=16000, channels=1, dtype="float32"); sd.wait()
        text, dt = transcribe(a[:, 0])
        print(">>", text, f"   [{dt:.2f}s for {SECS}s audio]")