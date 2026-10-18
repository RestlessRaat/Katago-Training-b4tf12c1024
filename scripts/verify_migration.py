"""Compare original TF-export and migrated PyTorch-export through the same engine."""
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline import CFG, RUN, path, save_json


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("converted_model")
    a = p.parse_args()
    # Fixed symmetry, float32, explicit rules, three independent legal positions.
    positions = [[], [("b", "C3"), ("w", "G7"), ("b", "G3"), ("w", "C7")],
                 [("b", "E5"), ("w", "D5"), ("b", "D4"), ("w", "E4"), ("b", "F4"), ("w", "C4"), ("b", "E3"), ("w", "C5")]]
    commands = ["boardsize 9", "kata-set-rules chinese", "komi 7.5"]
    for i, moves in enumerate(positions):
        commands += ["clear_board"] + [f"play {c} {loc}" for c, loc in moves] + [f"{100+i} kata-raw-nn b 0"]
    commands += ["quit"]
    outputs = []
    for label, model in [("original", path(CFG["base_model"])), ("converted", path(a.converted_model))]:
        proc = subprocess.run([CFG["engine"], "gtp", "-config", str(ROOT / "configs/gtp9.cfg"), "-model", str(model)],
                              input="\n".join(commands) + "\n", capture_output=True, text=True, encoding="utf-8", timeout=120, cwd=ROOT)
        (RUN / f"migration-{label}-gtp.txt").write_text(proc.stdout + "\nSTDERR\n" + proc.stderr, encoding="utf-8")
        if proc.returncode or "?" in proc.stdout:
            raise RuntimeError(f"GTP failed for {label}: {proc.stderr}\n{proc.stdout}")
        records = []
        for i in range(len(positions)):
            block = re.search(rf"={100+i}\s+(.*?)(?=\n\n)", proc.stdout, re.S).group(1)
            scalars = {key: float(re.search(rf"(?m)^{key} ([-+0-9.eE]+)", block).group(1))
                       for key in ["whiteWin", "whiteLoss", "noResult", "whiteLead", "whiteScoreSelfplay", "whiteScoreSelfplaySq"]}
            policy = [float(x) for x in block.split("policy\n")[1].split("whiteOwnership")[0].replace("policyPass", "").split()]
            ownership = [float(x) for x in block.split("whiteOwnership\n")[1].split()]
            records.append({"scalars": scalars, "policy": policy, "ownership": ownership})
        outputs.append(records)
    comparisons = []
    for i, (left, right) in enumerate(zip(*outputs)):
        diffs = {key: abs(left["scalars"][key] - right["scalars"][key]) for key in left["scalars"]}
        lp, rp = np.array(left["policy"]), np.array(right["policy"])
        same_illegal_mask = bool(np.array_equal(np.isnan(lp), np.isnan(rp)))
        legal = np.isfinite(lp) & np.isfinite(rp)
        diffs["policy_max_abs"] = float(np.max(np.abs(lp[legal] - rp[legal])))
        diffs["ownership_max_abs"] = float(np.max(np.abs(np.array(left["ownership"]) - right["ownership"])))
        # GTP outputs scores rounded to .001 and policy probabilities to .000001.
        passed = all(diffs[k] <= 0.002 for k in ["whiteLead", "whiteScoreSelfplay", "whiteScoreSelfplaySq"])
        passed &= all(diffs[k] <= 2e-5 for k in ["whiteWin", "whiteLoss", "noResult", "policy_max_abs", "ownership_max_abs"])
        passed &= same_illegal_mask
        comparisons.append({"position": i, "passed": bool(passed), "same_illegal_move_mask": same_illegal_mask, "differences": diffs})
    report = {"passed": all(x["passed"] for x in comparisons), "positions": comparisons,
              "compared": "Primary policy incl pass, value probabilities, score moments, lead, ownership; FP32, symmetry 0",
              "excluded": "Changed auxiliary score-distribution/error/variance-time heads; not an assertion of bit-identical full training state"}
    save_json(RUN / "migration-verification.json", report)
    print(json.dumps(report, indent=2))
    if not report["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
