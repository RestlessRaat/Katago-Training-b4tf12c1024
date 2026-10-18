"""One-time, CPU-only extraction in .venv-convert (Python 3.10 / NumPy 1.26).
No TensorFlow runtime, GPU, model execution, or unpickling is needed here.
"""
import argparse
import json
import os
import threading
from pathlib import Path
import numpy as np

# The lightweight upstream reader assumes POSIX pread. Supply its read-only
# equivalent on Windows, with binary mode and a lock around seek/read/restore.
if not hasattr(os, "pread"):
    import msvcrt
    _read_lock = threading.Lock()
    def _pread(fd, size, offset):
        with _read_lock:
            msvcrt.setmode(fd, os.O_BINARY)
            old = os.lseek(fd, 0, os.SEEK_CUR)
            try:
                os.lseek(fd, offset, os.SEEK_SET)
                return os.read(fd, size)
            finally:
                os.lseek(fd, old, os.SEEK_SET)
    os.pread = _pread
from tensorflow_checkpoint_reader import py_checkpoint_reader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prefix")
    p.add_argument("output")
    a = p.parse_args()
    reader = py_checkpoint_reader.NewCheckpointReader(a.prefix)
    shapes = reader.get_variable_to_shape_map()
    values = {name: reader.get_tensor(name) for name in shapes}
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **values)
    out.with_suffix(".json").write_text(json.dumps(shapes, indent=2), encoding="utf-8")
    print(f"Extracted {len(values)} tensors to {out}")


if __name__ == "__main__":
    main()
