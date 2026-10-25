import argparse, io, os, time, itertools, torch, soundfile as sf
from dataclasses import dataclass
from datasets import load_dataset, Audio
from huggingface_hub import login, HfApi
from transformers import (AutoProcessor, MoonshineForConditionalGeneration,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--smoke", type=int, default=0)     # >0: stream this many clips, no full download
ap.add_argument("--push", type=int, default=1)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--fp32", type=int, default=0)
a = ap.parse_args()

HF = os.environ["HF_TOKEN"]; login(HF)
DATA, OUT, BASE = "PrathamR777/hinglish-moonshine", "PrathamR777/moonshine-tiny-hinglish", "UsefulSensors/moonshine-tiny"
if a.push:
    HfApi(token=HF).create_repo(OUT, private=True, exist_ok=True)   # fail early if token can't write
cap = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
bf16 = cap >= 8 and not a.fp32
fp16 = 0 < cap < 8 and not a.fp32
print("cuda:", torch.cuda.is_available(), "| capability:", cap, "| bf16:", bf16, "| fp16:", fp16)

proc = AutoProcessor.from_pretrained(BASE)
model = MoonshineForConditionalGeneration.from_pretrained(BASE)
if a.smoke:
    s = load_dataset(DATA, split="train", streaming=True, token=HF).cast_column("audio", Audio(decode=False))
    train_ds = list(itertools.islice(iter(s), a.smoke))
else:
    train_ds = load_dataset(DATA, split="train", token=HF).cast_column("audio", Audio(decode=False))
EOS = model.config.eos_token_id
if model.config.pad_token_id is None: model.config.pad_token_id = EOS
if model.config.decoder_start_token_id is None: model.config.decoder_start_token_id = model.config.bos_token_id
print("EOS:", EOS, "| BOS:", model.config.bos_token_id, "| decoder_start:", model.config.decoder_start_token_id, "| pad:", model.config.pad_token_id)
assert EOS is not None
print("train clips:", len(train_ds), "| params (M):", sum(p.numel() for p in model.parameters()) / 1e6)

@dataclass
class Collator:
    proc: object
    eos: int
    def __call__(self, batch):
        wavs = [sf.read(io.BytesIO(b["audio"]["bytes"]), dtype="float32")[0] for b in batch]
        f = self.proc.feature_extractor(wavs, sampling_rate=16000, return_tensors="pt",
                                        padding=True, return_attention_mask=True)
        tk = self.proc.tokenizer
        ids = [tk(b["text"], add_special_tokens=False).input_ids + [self.eos] for b in batch]
        L = max(map(len, ids))
        labels = torch.full((len(ids), L), -100, dtype=torch.long)
        for i, x in enumerate(ids):
            labels[i, :len(x)] = torch.tensor(x)
        out = {"input_values": f["input_values"], "labels": labels}
        if "attention_mask" in f:
            out["attention_mask"] = f["attention_mask"]
        return out

args = Seq2SeqTrainingArguments(
    output_dir="out", per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum,
    learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=max(3, min(300, a.steps // 10)),
    max_steps=a.steps, bf16=bf16, fp16=fp16, save_steps=500, save_total_limit=2,
    logging_steps=10, remove_unused_columns=False, dataloader_num_workers=2, report_to="none")
t0 = time.time()
Seq2SeqTrainer(model=model, args=args, train_dataset=train_ds, data_collator=Collator(proc, EOS)).train()
print(f"training took {time.time()-t0:.0f}s for {a.steps} steps = {(time.time()-t0)/a.steps:.2f} s/step")
model.save_pretrained("final"); proc.save_pretrained("final")
if a.push:
    model.push_to_hub(OUT, private=True); proc.push_to_hub(OUT, private=True)
    print("pushed", OUT)