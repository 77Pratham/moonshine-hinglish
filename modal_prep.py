import os, subprocess
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install(
        "torch",
        "datasets<3.0.0",
        "librosa",
        "soundfile",
        "scipy",
        "numpy",
        "huggingface_hub",
        "pandas",
        "indic-nlp-library",
        "pyarrow",
    )
    .add_local_file("romanize.py", "/root/romanize.py")
    .add_local_file("prep_data.py", "/root/prep_data.py")
)

app = modal.App("moonshine-prep", image=image)


@app.function(
    secrets=[modal.Secret.from_name("hf")],
    cpu=4,
    memory=16384,
    timeout=6 * 3600,
)
def prep(hours: float = 60.0):
  subprocess.run(
      ["python", "/root/prep_data.py", str(hours)], cwd="/root", check=True
  )


@app.local_entrypoint()
def main(hours: float = 60.0):
  prep.remote(hours)