import io, os, re, sys, unicodedata
from collections import Counter
from math import gcd
import soundfile as sf
from scipy.signal import resample_poly
from datasets import load_dataset, Audio, Dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi
from romanize import romanize_text, LOANWORDS

TOKEN = os.environ["HF_TOKEN"]
REPO = "PrathamR777/hinglish-moonshine"
AUDIO_COL, TEXT_COL, DUR_COL = "audio_filepath", "text", "duration"
TRAIN_H = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
TEST_H = min(1.5, TRAIN_H)

def get_flac16k(b: bytes):
    info = sf.info(io.BytesIO(b))
    if info.samplerate == 16000 and info.channels == 1 and info.format == "FLAC":
        return b, info.duration
    data, sr = sf.read(io.BytesIO(b), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        g = gcd(int(sr), 16000)
        data = resample_poly(data, 16000 // g, int(sr) // g).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, data, 16000, format="FLAC", subtype="PCM_16")
    return buf.getvalue(), len(data) / 16000

def gen(split, hours):
    ds = load_dataset("ai4bharat/IndicVoices", "hindi", split=split, streaming=True, token=TOKEN)
    ds = ds.cast_column(AUDIO_COL, Audio(decode=False))
    if split == "train":
        ds = ds.shuffle(seed=0, buffer_size=2000)
    total = 0.0
    for ex in ds:
        d0 = ex.get(DUR_COL)
        if d0 is not None and not (1.0 <= d0 <= 20.0): continue
        a = ex[AUDIO_COL]
        if not a or not a.get("bytes"): continue
        orig = ex[TEXT_COL] or ""
        if re.search(r"[0-9०-९]", orig): continue
        text = romanize_text(orig)
        if not text: continue
        b, dur = get_flac16k(a["bytes"])
        if not (1.0 <= dur <= 20.0): continue
        if not (3 <= len(text) / dur <= 25): continue
        yield {"audio": {"bytes": b, "path": None}, "text": text, "orig_text": orig,
               "dur": dur, "speaker_id": ex.get("speaker_id") or ""}
        total += dur
        if total >= hours * 3600: break

feats = Features({"audio": Audio(sampling_rate=16000), "text": Value("string"),
                  "orig_text": Value("string"), "dur": Value("float32"),
                  "speaker_id": Value("string")})
train = Dataset.from_generator(gen, gen_kwargs={"split": "train", "hours": TRAIN_H}, features=feats)
test = Dataset.from_generator(gen, gen_kwargs={"split": "valid", "hours": TEST_H}, features=feats)
print(f"train: {len(train)} clips, {sum(train['dur'])/3600:.1f} h | test: {len(test)} clips, {sum(test['dur'])/3600:.2f} h")

# speaker-disjoint test set
train_spk = set(train["speaker_id"])
filtered = test.filter(lambda s: s not in train_spk, input_columns=["speaker_id"])
print(f"speakers in train: {len(train_spk)}; test clips {len(test)} -> {len(filtered)} after removing train speakers")
if len(filtered) >= 200:
    test = filtered
else:
    print("WARNING: too few speaker-disjoint test clips; keeping unfiltered test. Disclose this in the README.")

tr = train.cast_column("audio", Audio(decode=False))
for r in tr.select(range(min(8, len(tr)))):
    print(r["orig_text"], "=>", r["text"])

if TRAIN_H >= 1.0:
    DatasetDict({"train": train, "test": test}).push_to_hub(REPO, private=True, token=TOKEN)
    cnt = Counter()
    for t in train["orig_text"]:
        for w in t.split():
            w = unicodedata.normalize("NFC", w)
            if re.search(r"[\u0900-\u097F]", w) and w not in LOANWORDS:
                cnt[w] += 1
    lines = [f"{n}\t{w}" for w, n in cnt.most_common(2500)]
    HfApi(token=TOKEN).upload_file(path_or_fileobj="\n".join(lines).encode("utf-8"),
                                   path_in_repo="top_words.txt", repo_id=REPO, repo_type="dataset")
    print("pushed to", REPO, "(+ top_words.txt)")
else:
    print("short test run: not pushed")