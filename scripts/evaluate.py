"""Bounded candidate-vs-incumbent matches on 9x9. Never promotes automatically."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import re
from pipeline import CFG, ROOT, RUN, SOURCE, command, path, save_json


def evaluate(candidate, games=200, visits=128):
    if games < 2 or games % 2:
        raise ValueError("Use an even number of games >= 2 for balanced colors")
    if visits < 1: raise ValueError("visits must be positive")
    incumbent = RUN / "incumbent.json"
    baseline = path(json.loads(incumbent.read_text(encoding="utf-8"))["model"]) if incumbent.exists() else path(CFG["base_model"])
    candidate = path(candidate).resolve()
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    out = RUN / "evaluation" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + digest[:8])
    (out / "sgfs").mkdir(parents=True, exist_ok=False)
    config = (SOURCE / "cpp/configs/match_example.cfg").read_text(encoding="utf-8")
    replacement = {"numBots": "2", "botName": None, "nnModelFile": None,
                   "numGameThreads": "4", "numGamesTotal": str(games), "maxMovesPerGame": "400",
                   "allowResignation": "false", "koRules": "POSITIONAL", "scoringRules": "AREA",
                   "taxRules": "NONE", "multiStoneSuicideLegals": "false", "hasButtons": "false",
                   "bSizes": "9", "bSizeRelProbs": "1", "komiAuto": "false",
                   "maxVisits": str(visits), "nnMaxBatchSize": "16", "nnCacheSizePowerOfTwo": "18",
                   "nnMutexPoolSizePowerOfTwo": "12", "logGamesEvery": "4"}
    for key, val in replacement.items():
        config, count = re.subn(rf"(?m)^{key}\s*=.*$", "" if val is None else f"{key} = {val}", config)
        if count != 1: raise ValueError(f"Missing/duplicate upstream match key {key}")
    config += f"\nbotName0 = incumbent\nbotName1 = candidate\nnnModelFile0 = {baseline.as_posix()}\nnnModelFile1 = {candidate.as_posix()}\nkomiMean = 7.5\nkomiStdev = 0\nallowRectangleProb = 0\nhomeDataDir = {(RUN / 'engine-cache').as_posix()}\n"
    (out / "match.cfg").write_text(config, encoding="utf-8")
    command([CFG["engine"], "match", "-config", out / "match.cfg", "-log-file", out / "match.log",
             "-sgf-output-dir", out / "sgfs"], "evaluate")
    # KataGo writes one SGF per line in .sgfs files. Parse results, not log estimates.
    wins = losses = draws = 0
    colors = {"B": 0, "W": 0}
    for file in (out / "sgfs").rglob("*"):
        if not file.is_file(): continue
        for game in file.read_text(encoding="utf-8").splitlines():
            if not game.strip(): continue
            props = {k: re.search(rf"(?:^|[;)])?{k}\[([^\]]*)\]", game) for k in ["PB", "PW", "RE"]}
            if not all(props.values()): raise RuntimeError(f"Missing SGF player/result: {file}")
            black, white, result = [props[k].group(1) for k in ["PB", "PW", "RE"]]
            side = "B" if black == "candidate" else "W" if white == "candidate" else None
            if side is None: raise RuntimeError(f"Candidate missing: {black}, {white}")
            colors[side] += 1
            if result.startswith(side + "+"): wins += 1
            elif result.startswith(("W" if side == "B" else "B") + "+"): losses += 1
            elif result in ("0", "Draw", "Jigo"): draws += 1
            else: raise RuntimeError(f"Unscored or void match: {result}")
    n = wins + losses + draws
    if n != games: raise RuntimeError(f"Expected {games} completed games, got {n}")
    # Conservative approximate Wilson interval; draws count as half a point.
    score = (wins + 0.5 * draws) / n
    z = 1.96
    center = (score + z*z/(2*n)) / (1+z*z/n)
    half = z*math.sqrt(score*(1-score)/n + z*z/(4*n*n)) / (1+z*z/n)
    balanced = colors["B"] == colors["W"]
    report = {"candidate": str(candidate), "candidate_sha256": digest, "incumbent": str(baseline),
              "games": n, "visits": visits, "rules": "Chinese/area, positional ko, komi 7.5, 9x9",
              "wins": wins, "losses": losses, "draws": draws, "colors": colors,
              "score_rate": score, "approximate_95_percent_interval": [center-half, center+half],
              "gate_passed": n >= 200 and balanced and center-half > 0.5,
              "caveat": "Operational screening only; stochastic openings are not paired; interval assumes approximately independent games; confirm close results on a fresh match set"}
    save_json(out / "result.json", report)
    save_json(RUN / "last-evaluation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--visits", type=int, default=128)
    a = p.parse_args()
    evaluate(a.candidate, a.games, a.visits)
