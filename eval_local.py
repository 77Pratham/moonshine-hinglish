import io, os, re, sys, time, math
import torch, jiwer, soundfile as sf
from datasets import load_dataset, Audio
from transformers import AutoProcessor, MoonshineForConditionalGeneration

TOKEN = os.environ["HF_TOKEN"]
DATA = "PrathamR777/hinglish-moonshine"
N = int(os.environ.get("N", "881"))
repos = sys.argv[1:] or ["UsefulSensors/moonshine-tiny"]

def norm(t):
    t = t.lower()
    t = re.sub(r"[^\w\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()

spk = None
try:   # speaker ids of the training set (reads only that column)
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=TOKEN)
    spk = set()
    for f in fs.glob(f"datasets/{DATA}/data/train-*.parquet"):
        with fs.open(f, "rb") as fh:
            spk |= set(pq.read_table(fh, columns=["speaker_id"]).column(0).to_pylist())
    print("train speakers:", len(spk))
except Exception as e:
    print("could not compute speaker-disjoint subset:", e)

test = load_dataset(DATA, split="test", token=TOKEN).cast_column("audio", Audio(decode=False))
dis_idx = [i for i, s in enumerate(test["speaker_id"]) if spk is not None and s not in spk]
dis_set = set(dis_idx)
rest = [i for i in range(len(test)) if i not in dis_set]
keep = sorted(dis_idx + rest[:max(0, N - len(dis_idx))])
test = test.select(keep)
disjoint = [j for j, i in enumerate(keep) if i in dis_set]
print("test clips:", len(test), "| speaker-disjoint:", len(disjoint))
torch.set_num_threads(os.cpu_count())

for repo in repos:
    proc = AutoProcessor.from_pretrained(repo, token=TOKEN)
    model = MoonshineForConditionalGeneration.from_pretrained(repo, token=TOKEN).eval()
    if os.environ.get("INT8"):
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    hyps, refs, rt, t_audio = [], [], 0.0, 0.0
    for ex in test:
        a, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
        f = proc.feature_extractor(a, sampling_rate=16000, return_tensors="pt")
        t = time.time()
        with torch.inference_mode():
            ids = model.generate(**f, max_new_tokens=max(8, math.ceil(len(a) / 16000 * 10)))
        rt += time.time() - t; t_audio += len(a) / 16000
        hyps.append(norm(proc.tokenizer.decode(ids[0], skip_special_tokens=True)))
        refs.append(ex["text"])

    def score(idx):
        idx = list(idx)
        return (round(jiwer.wer([refs[i] for i in idx], [hyps[i] for i in idx]), 4),
                round(jiwer.cer([refs[i] for i in idx], [hyps[i] for i in idx]), 4))

    print("\n===", repo, "(INT8)" if os.environ.get("INT8") else "", "===")
    print("ALL      (WER, CER):", score(range(len(refs))), f"[{len(refs)} clips]")
    if disjoint:
        print("DISJOINT (WER, CER):", score(disjoint), f"[{len(disjoint)} clips]")
    print("real-time factor:", round(rt / t_audio, 3), "(compute time / audio time; lower is faster)")
    for i in list(range(0, len(refs), max(1, len(refs) // 8)))[:8]:
        print("REF:", refs[i]); print("HYP:", hyps[i]); print()