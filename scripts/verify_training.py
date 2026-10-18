"""Acceptance check against saved real training artifacts, including optimizer resume."""
import json
import sys
import torch
import numpy as np
from pipeline import CFG, ROOT, RUN, SOURCE, path, save_json

sys.path.insert(0, str(SOURCE / "python"))
from katago.train.load_model import load_checkpoint


def main():
    initial = load_checkpoint(path(CFG["initial_checkpoint"]))
    first = load_checkpoint(RUN / "verification/first-trained.ckpt")
    resumed = load_checkpoint(RUN / "train/checkpoint.ckpt")
    samples = [c["train_state"]["global_step_samples"] for c in (initial, first, resumed)]
    def delta(a, b):
        return {k: float((b["model"][k].cpu() - a["model"][k].cpu()).abs().max()) for k in a["model"]}
    d1, d2 = delta(initial, first), delta(first, resumed)
    finite = all(bool(torch.isfinite(t).all()) for t in resumed["model"].values())
    def optstep(c):
        return max(float(s.get("step", 0)) for s in c["optimizer"]["state"].values())
    steps = [optstep(c) for c in (first, resumed)]
    shape_report = {}
    filesets = []
    for label in ("train", "val"):
        files = list((RUN / "selfplay" / label).rglob("*.npz"))
        file_hashes = set()
        rows = 0
        for file in files:
            import hashlib
            file_hashes.add(hashlib.sha256(file.read_bytes()).hexdigest())
            with np.load(file) as data:
                b = data["binaryInputNCHWPacked"]
                assert b.shape[1:] == (22, 11), b.shape
                rows += b.shape[0]
        filesets.append(file_hashes)
        shape_report[label] = {"files": len(files), "rows": rows, "spatial_features": 22, "packed_9x9_bytes": 11}
    no_identical_files = not bool(filesets[0] & filesets[1])
    report = {"passed": finite and samples[0] < samples[1] < samples[2] and steps[0] < steps[1]
               and max(d1.values()) > 0 and max(d2.values()) > 0 and no_identical_files,
              "sample_counts_initial_first_resumed": samples, "optimizer_steps_first_resumed": steps,
              "all_parameters_finite": finite, "first_update_max_abs": max(d1.values()),
              "resumed_update_max_abs": max(d2.values()), "data": shape_report,
              "train_val_no_identical_files": no_identical_files,
              "training_parameter_tensors_changed_first": sum(v > 0 for v in d1.values()),
              "training_parameter_tensors_changed_resumed": sum(v > 0 for v in d2.values())}
    save_json(RUN / "training-verification.json", report)
    print(json.dumps(report, indent=2))
    if not report["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
