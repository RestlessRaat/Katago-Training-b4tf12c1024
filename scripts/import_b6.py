"""Migrate official fixup TF v8 b6c96 into the current PyTorch trainer.

Preserves input/trunk, primary policy, value, ownership and existing auxiliary
spatial heads. New soft policies copy the corresponding existing policy.
The changed score-distribution architecture and new error/TD heads initialize
fresh; the complete tensor map and omissions are recorded for inspection.
"""
import copy
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = json.loads((ROOT / "configs/local.json").read_text(encoding="utf-8"))
SOURCE = ROOT / "vendor" / ("KataGo-" + SETTINGS["source_commit"])
sys.path.insert(0, str(SOURCE / "python"))
from katago.train import modelconfigs
from katago.train.model_pytorch import Model


def main():
    torch.manual_seed(20261017)
    source = ROOT / "models/pretrained/b6c96-tf/tensors.npz"
    dest = ROOT / SETTINGS["initial_checkpoint"]
    if dest.exists():
        raise SystemExit(f"Refusing to overwrite existing initial checkpoint: {dest}")
    archive = np.load(source, allow_pickle=False)
    # Official download contains the published SWA parameters.
    tf = {name.removeprefix("swa_model/"): archive[name] for name in archive.files}
    cfg = copy.deepcopy(modelconfigs.config_of_name["b6c96"])
    cfg.update(version=11, initial_conv_kernel_size=5, gamma_weight_decay_center_1=False)
    model = Model(cfg, pos_len=9)
    model.initialize()
    state = model.state_dict()
    mapped, used = [], set()

    def get(name):
        # The SavedModel variables use plain TF1 names in the official archive.
        if name not in tf:
            raise KeyError(f"Missing source tensor {name}; available: {list(tf)}")
        used.add(name)
        return np.array(tf[name], copy=True)

    def put(pt, name, kind="identity"):
        x = get(name)
        if kind == "conv":
            x = x.transpose(3, 2, 0, 1)
        elif kind == "linear":
            x = x.T
        t = torch.tensor(x, dtype=state[pt].dtype).reshape(state[pt].shape)
        state[pt] = t
        mapped.append({"pytorch": pt, "tensorflow": name, "shape": list(t.shape), "transform": kind})

    put("conv_spatial.weight", "conv1/w", "conv")
    put("linear_global.weight", "ginputw", "linear")
    for i, (_, kind) in enumerate(cfg["block_kind"]):
        old, new = f"rconv{i+1}", f"blocks.{i}"
        put(new + ".normactconv1.norm.beta", old + "/norm1/beta")
        put(new + ".normactconv2.norm.beta", old + "/norm2/beta")
        put(new + ".normactconv2.norm.gamma", old + "/norm2/gamma")
        put(new + ".normactconv2.conv.weight", old + "/w2", "conv")
        if kind == "regular":
            put(new + ".normactconv1.conv.weight", old + "/w1", "conv")
        else:
            prefix = new + ".normactconv1.convpool"
            put(prefix + ".conv1r.weight", old + "/w1a", "conv")
            put(prefix + ".conv1g.weight", old + "/w1b", "conv")
            put(prefix + ".normg.beta", old + "/norm1b/beta")
            put(prefix + ".linear_g.weight", old + "/w1r", "linear")
    put("norm_trunkfinal.beta", "trunk/norm/beta")
    pairs = {
        "policy_head.conv1p.weight": ("p1/intermediate_conv/w", "conv"),
        "policy_head.conv1g.weight": ("g1/w", "conv"),
        "policy_head.biasg.beta": ("g1/norm/beta", "identity"),
        "policy_head.linear_g.weight": ("matmulg2w", "linear"),
        "policy_head.bias2.beta": ("p1/norm/beta", "identity"),
        "value_head.conv1.weight": ("v1/w", "conv"),
        "value_head.bias1.beta": ("v1/norm/beta", "identity"),
        "value_head.linear2.weight": ("v2/w", "linear"),
        "value_head.linear2.bias": ("v2/b", "identity"),
        "value_head.linear_valuehead.weight": ("v3/w", "linear"),
        "value_head.linear_valuehead.bias": ("v3/b", "identity"),
        "value_head.linear_miscvaluehead.weight": ("mv3/w", "linear"),
        "value_head.linear_miscvaluehead.bias": ("mv3/b", "identity"),
        "value_head.conv_ownership.weight": ("vownership/w", "conv"),
        "value_head.conv_scoring.weight": ("vscoring/w", "conv"),
        "value_head.conv_futurepos.weight": ("futurepos/w", "conv"),
        "value_head.conv_seki.weight": ("seki/w", "conv"),
    }
    for pt, (name, kind) in pairs.items():
        put(pt, name, kind)
    for pt, old, transform in [
        ("policy_head.conv2p.weight", "p2/w", "conv"),
        ("policy_head.linear_pass.weight", "matmulpass", "linear"),
    ]:
        x = get(old)
        x = x.transpose(3, 2, 0, 1) if transform == "conv" else x.T
        # v11 adds softened policy heads; preserve original two outputs exactly.
        y = np.concatenate([x, x], axis=0)
        state[pt] = torch.tensor(y, dtype=state[pt].dtype)
        mapped.append({"pytorch": pt, "tensorflow": old, "transform": "copy primary/opponent into soft heads"})
    model.load_state_dict(state, strict=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "model": model.state_dict(), "train_state": {"global_step_samples": 0}}, dest)
    mapped_keys = {x["pytorch"] for x in mapped}
    report = {
        "source_archive_sha256": hashlib.sha256((ROOT / "downloads/b6c96-original.zip").read_bytes()).hexdigest(),
        "source_name": "b6c96-s175395328-d26788732", "source_model_version": 8,
        "target_model_version": 11, "input_feature_version": 7,
        "mapped_parameters": sum(state[k].numel() for k in mapped_keys),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "mapped": mapped,
        "newly_initialized": [k for k in state if k not in mapped_keys],
        "unused_tf_tensors": sorted(set(tf) - used),
        "limitations": ["Score-distribution head changed architecture and is initialized fresh", "New short-term error and third TD/TD-score head initialized fresh", "Variance-time scaling changed from 150 to 40; this auxiliary prediction requires adaptation", "Optimizer history unavailable; new optimizer state starts at zero"]
    }
    dest.with_suffix(".migration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["mapped_parameters", "total_parameters", "newly_initialized"]}, indent=2))
    print(dest)


if __name__ == "__main__":
    main()
