"""Finite, resumable 9x9 KataGo rehearsal; outputs stay in this workspace."""
import argparse
import codecs
from datetime import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from progress import Progress

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/local.json").read_text(encoding="utf-8"))
if CFG["board_size"] != 9:
    raise ValueError("This deployment targets 9x9. Adapt configuration generation, training and evaluation together for a new board size.")
SOURCE = ROOT / "vendor" / ("KataGo-" + CFG["source_commit"])
RUN = ROOT / CFG["run_dir"]
ENV = os.environ.copy()
ENV.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4", KATAGO_FUSED_SWIGLU_KERNEL="0", KATAGO_FUSED_ROPE_BACKWARD="0", KATAGO_COMPILE_TRAINING_LOSS="0")


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def save_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def command(args, name, timeout=None):
    log = RUN / "logs" / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + name + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    progress = Progress(name)
    print(f"[{progress.label}] 日志：{log}", flush=True)
    progress.emit(force=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as f:
        f.write(json.dumps([str(a) for a in args]) + "\n")
        f.flush()
        with subprocess.Popen([str(a) for a in args], cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT) as proc:
            try:
                # Tail the file instead of piping stdout: full diagnostics are
                # retained, and a quiet subprocess cannot block status updates.
                with log.open("rb") as reader:
                    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                    pending = ""
                    while True:
                        pending += decoder.decode(reader.read())
                        lines = pending.split("\n")
                        pending = lines.pop()
                        for line in lines: progress.feed(line)
                        progress.emit()
                        rc = proc.poll()
                        if rc is not None:
                            # Child may have written between read() and poll().
                            pending += decoder.decode(reader.read(), final=True)
                            for line in pending.splitlines(): progress.feed(line)
                            break
                        if timeout is not None and time.monotonic()-started >= timeout:
                            raise subprocess.TimeoutExpired(args, timeout)
                        time.sleep(0.25)
            except BaseException:
                proc.terminate()
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise
    progress.finished(rc)
    if rc:
        print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]), flush=True)
        raise RuntimeError(f"{name} failed; inspect {log}")
    return log


def configure():
    replacements = {
        "dataBoardLen": "9", "bSizes": "9", "bSizeRelProbs": "1", "allowRectangleProb": "0",
        "numGameThreads": "8", "maxMovesPerGame": "400", "maxVisits": "128",
        "cheapSearchVisits": "32", "reducedVisitsMin": "32", "nnMaxBatchSize": "16",
        "nnCacheSizePowerOfTwo": "18", "nnMutexPoolSizePowerOfTwo": "12",
        "maxRowsPerTrainFile": "1024", "firstFileRandMinProp": "1.0",
        "logGamesEvery": "4", "estimateLeadProb": "0.02",
    }
    text = (SOURCE / "cpp/configs/training/selfplay1.cfg").read_text(encoding="utf-8")
    for key, value in replacements.items():
        text, count = re.subn(rf"(?m)^{key}\s*=.*$", key + " = " + value, text)
        if count != 1: raise ValueError(f"Expected one upstream key {key}, got {count}")
    text += "\nhomeDataDir = " + (RUN / "engine-cache").as_posix() + "\n"
    (ROOT / "configs/selfplay9.cfg").write_text(text, encoding="utf-8")
    (ROOT / "configs/gtp9.cfg").write_text(
        "logToStderr = true\nlogAllGTPCommunication = false\nlogSearchInfo = false\nnumSearchThreads = 1\nmaxVisits = 64\n"
        "rules = chinese\nnnMaxBatchSize = 8\nnnCacheSizePowerOfTwo = 16\n"
        "cudaUseFP16 = false\ncudaUseNHWC = false\n"
        "homeDataDir = " + (RUN / "engine-cache").as_posix() + "\n", encoding="utf-8")
    baseline = RUN / "models/baseline"
    baseline.mkdir(parents=True, exist_ok=True)
    target = baseline / "model.txt.gz"
    if not target.exists(): shutil.copy2(path(CFG["base_model"]), target)
    print("Configured", RUN)


def doctor():
    import numpy as np
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    x = torch.ones((64, 64), device="cuda")
    assert float((x @ x)[0, 0]) == 64
    data = {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(0), "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
            "source_commit": CFG["source_commit"], "engine": CFG["engine"]}
    data["engine_version"] = subprocess.check_output([CFG["engine"], "version"], text=True, encoding="utf-8")
    save_json(RUN / "environment.json", data)
    print(json.dumps(data, ensure_ascii=False, indent=2))


def selfplay(games, label="train", smoke=False, model=None):
    if games < 1: raise ValueError("games must be positive")
    (RUN / "selfplay" / label).mkdir(parents=True, exist_ok=True)
    (RUN / "engine-cache").mkdir(parents=True, exist_ok=True)
    modeldir = RUN / "models"
    if (RUN / "incumbent.json").exists():
        modeldir = path(json.loads((RUN / "incumbent.json").read_text(encoding="utf-8"))["models_dir"])
    if model:
        modeldir = RUN / "candidate-selfplay-models" / hashlib.sha256(path(model).read_bytes()).hexdigest()[:12]
        only = modeldir / "candidate"
        only.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path(model), only / "model.bin.gz")
    overrides = ""
    if smoke:
        overrides = "maxVisits=32,cheapSearchProb=0,reducedVisitsMin=32,numGameThreads=4,estimateLeadProb=0,earlyForkGameProb=0,forkGameProb=0,sekiForkHackProb=0"
    args = [CFG["engine"], "selfplay", "-config", ROOT / "configs/selfplay9.cfg",
            "-models-dir", modeldir, "-output-dir", RUN / "selfplay" / label,
            "-max-games-total", games]
    if overrides: args += ["-override-config", overrides]
    return command(args, "selfplay-" + label)


def shuffle(smoke=False):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    staging = RUN / "shuffleddata" / (stamp + ".tmp")
    staging.mkdir(parents=True, exist_ok=False)
    for label in ("train", "val"):
        (RUN / "scratch" / stamp / label).mkdir(parents=True, exist_ok=True)
        inputs = RUN / "selfplay" / label
        if not list(inputs.rglob("*.npz")): raise RuntimeError(f"No {label} NPZ files in {inputs}")
        command([sys.executable, SOURCE / "python/shuffle.py", inputs,
                 "-out-dir", staging / label, "-out-tmp-dir", RUN / "scratch" / stamp / label,
                 "-num-processes", "2", "-min-rows", "32", "-keep-target-rows", "all" if smoke else "250000",
                 "-max-rows", "250000", "-approx-rows-per-out-file", "1024"], "shuffle-" + label)
    if not (staging / "train.json").exists() or not (staging / "val.json").exists():
        raise RuntimeError("Incomplete shuffle; refusing to publish")
    output = staging.with_name(stamp)
    staging.rename(output)
    return output


def train(data=None, smoke=False, epochs=None):
    initial = path(CFG["initial_checkpoint"])
    if not initial.exists(): raise RuntimeError(f"Missing pretrained checkpoint {initial}; refusing random initialization")
    args = [sys.executable, SOURCE / "python/train.py", "-traindir", RUN / "train",
            "-initial-checkpoint", initial, "-pos-len", "9", "-batch-size", CFG["batch_size"],
            "-samples-per-epoch", "256" if smoke else CFG["samples_per_epoch"],
            "-max-epochs-this-instance", epochs or CFG["epochs_per_cycle"],
            "-lr-scale", CFG["lr_scale"], "-use-adamw", "-no-compile", "-use-fp16",
            "-lookahead-k", "1", "-lookahead-alpha", "1.0", "-max-val-samples", "256",
            "-no-export", "-no-longterm-checkpoints", "-quit-if-no-data"]
    args += ["-datadir", data] if data else ["-latestdatadir", RUN / "shuffleddata"]
    if not smoke:
        args += ["-no-lr-warmup", "-max-train-bucket-per-new-data", "1.0", "-max-train-bucket-size", "250000",
                 "-initial-train-bucket-level", CFG["samples_per_epoch"], "-stop-when-train-bucket-limited"]
    return command(args, "train")


def export(checkpoint=None):
    ckpt = path(checkpoint) if checkpoint else RUN / "train/checkpoint.ckpt"
    if not ckpt.exists(): raise FileNotFoundError(ckpt)
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()[:12]
    out = RUN / "exports" / digest
    command([sys.executable, SOURCE / "python/export_model_pytorch.py", "-checkpoint", ckpt,
             "-export-dir", out, "-model-name", "b6c96-9x9-" + digest,
             "-filename-prefix", "model"], "export")
    with (out / "model.bin").open("rb") as src, (out / "model.bin.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as dst:
            shutil.copyfileobj(src, dst)
    save_json(out / "provenance.json", {"checkpoint": str(ckpt), "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(), "source_commit": CFG["source_commit"]})
    save_json(RUN / "last-export.json", {"model": str(out / "model.bin.gz"), "checkpoint": str(ckpt), "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()})
    print("EXPORTED", out / "model.bin.gz")
    return out / "model.bin.gz"


def cycle(smoke=False):
    print("[cycle 1/6] 生成训练自对弈数据", flush=True)
    selfplay(12 if smoke else CFG["selfplay_games"], "train", smoke)
    print("[cycle 2/6] 生成独立验证数据", flush=True)
    selfplay(4 if smoke else max(16, CFG["selfplay_games"] // 8), "val", smoke)
    print("[cycle 3/6] 整理、打乱训练与验证数据", flush=True)
    data = shuffle(smoke)
    print("[cycle 4/6] 继续训练并保存 checkpoint", flush=True)
    train(data, smoke)
    print("[cycle 5/6] 导出新权重", flush=True)
    result = export()
    print("[cycle 6/6] 新权重试运行自对弈", flush=True)
    selfplay(2 if smoke else 8, "candidate-probe", True, str(result))
    save_json(RUN / "last-cycle.json", {"finished": datetime.now().isoformat(), "smoke": smoke,
              "data": str(data), "candidate": str(result), "promotion": "pending evaluation"})
    print("Cycle complete. Candidate ready for evaluation:", result)


def promote():
    report = json.loads((RUN / "last-evaluation.json").read_text(encoding="utf-8"))
    if not report["gate_passed"]: raise RuntimeError("Evaluation gate did not pass; candidate remains separate")
    model = path(report["candidate"])
    if hashlib.sha256(model.read_bytes()).hexdigest() != report["candidate_sha256"]:
        raise RuntimeError("Evaluated candidate changed")
    current = RUN / "incumbent.json"
    before = json.loads(current.read_text(encoding="utf-8")) if current.exists() else {"model": str(path(CFG["base_model"])), "models_dir": str(RUN / "models")}
    if path(before["model"]).resolve() != path(report["incumbent"]).resolve():
        raise RuntimeError("Incumbent changed since evaluation; re-evaluate")
    active = RUN / "champions" / report["candidate_sha256"][:12]
    target = active / "network/model.bin.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, target)
    save_json(RUN / "previous-incumbent.json", before)
    save_json(current, {"model": str(target), "models_dir": str(active), "evaluation": report})
    print("Promoted", target)


def rollback():
    save_json(RUN / "incumbent.json", {"model": str(path(CFG["base_model"])), "models_dir": str(RUN / "models")})
    print("Selfplay restored to original baseline; training checkpoint retained")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["configure", "doctor", "selfplay", "shuffle", "train", "export", "cycle", "evaluate", "promote", "rollback"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--games", type=int)
    p.add_argument("--label", default="train", choices=["train", "val", "candidate-probe"])
    p.add_argument("--checkpoint")
    p.add_argument("--epochs", type=int)
    p.add_argument("--model")
    p.add_argument("--visits", type=int, default=128)
    a = p.parse_args()
    if a.action == "configure": configure()
    elif a.action == "doctor": doctor()
    elif a.action == "selfplay": selfplay(a.games or CFG["selfplay_games"], a.label, a.smoke, a.model)
    elif a.action == "shuffle": shuffle(a.smoke)
    elif a.action == "train": train(smoke=a.smoke, epochs=a.epochs)
    elif a.action == "export": export(a.checkpoint)
    elif a.action == "cycle": cycle(a.smoke)
    elif a.action == "evaluate":
        from evaluate import evaluate
        model = a.model or json.loads((RUN / "last-export.json").read_text(encoding="utf-8"))["model"]
        evaluate(model, a.games or 200, a.visits)
    elif a.action == "promote": promote()
    elif a.action == "rollback": rollback()


if __name__ == "__main__": main()
