"""Download pinned official assets, validate checksums, apply documented local patch."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = [
    {"file": "downloads/KataGo-source.zip", "url": "https://codeload.github.com/lightvector/KataGo/zip/d91ea855110dae533f0aada947b2b7d78cc8a4e1",
     "sha256": "8f3d26956d3ad94eec7ab5bff841d7310efd2c6e067ed53787ff679e39cf3793", "destination": "vendor",
     "sentinel": "vendor/KataGo-d91ea855110dae533f0aada947b2b7d78cc8a4e1/python/train.py"},
    {"file": "downloads/b6c96-original.zip", "url": "https://media.katagotraining.org/uploaded/networks/zips/kata1/kata1-b6c96-s175395328-d26788732.zip",
     "sha256": "405875c5cfc41ab6b4b5ff818edf3301103d6639fe9bb2c7f2e1489fce5289f4", "destination": "models/pretrained/b6c96-tf",
     "sentinel": "models/pretrained/b6c96-tf/b6c96-s175395328-d26788732/saved_model/variables/variables.index"},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    for asset in ASSETS:
        file = ROOT / asset["file"]
        file.parent.mkdir(parents=True, exist_ok=True)
        if not file.exists():
            if args.offline: raise FileNotFoundError(file)
            temp = file.with_suffix(file.suffix + ".partial")
            print("Downloading", asset["url"], flush=True)
            with urllib.request.urlopen(asset["url"], timeout=120) as response, temp.open("wb") as output:
                while chunk := response.read(1024*1024): output.write(chunk)
            actual = hashlib.sha256(temp.read_bytes()).hexdigest()
            if actual != asset["sha256"]: raise RuntimeError(f"Checksum mismatch: {temp}")
            temp.replace(file)
        if hashlib.sha256(file.read_bytes()).hexdigest() != asset["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {file}; preserve and inspect it")
        destination = ROOT / asset["destination"]
        if not (ROOT / asset["sentinel"]).exists():
            destination.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(file) as archive:
                for member in archive.infolist():
                    resolved = (destination / member.filename).resolve()
                    if not resolved.is_relative_to(destination.resolve()): raise RuntimeError("Unsafe archive path")
                archive.extractall(destination)
        print("Verified", asset["file"], flush=True)
    cfg = json.loads((ROOT / "configs/local.json").read_text(encoding="utf-8"))
    model = ROOT / "vendor" / ("KataGo-" + cfg["source_commit"]) / "python/katago/train/model_pytorch.py"
    old = 'self.conv_spatial = torch.nn.Conv2d(22, self.c_trunk, kernel_size=3, padding="same", bias=False)'
    new = 'self.conv_spatial = torch.nn.Conv2d(22, self.c_trunk, kernel_size=config.get("initial_conv_kernel_size", 3), padding="same", bias=False)'
    text = model.read_text(encoding="utf-8")
    if new not in text:
        if text.count(old) != 1: raise RuntimeError("Unexpected source; legacy patch cannot be applied safely")
        model.write_text(text.replace(old, new), encoding="utf-8")
    (ROOT / "deployment-assets.json").write_text(json.dumps(ASSETS, indent=2), encoding="utf-8")
    print("Local legacy b6 input-kernel compatibility patch is present")


if __name__ == "__main__": main()
