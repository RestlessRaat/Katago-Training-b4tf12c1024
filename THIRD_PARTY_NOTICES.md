# Third-party sources

`research/official-source/` contains reference snapshots from
[lightvector/KataGo](https://github.com/lightvector/KataGo), not the complete
runtime checkout. The upstream license is retained at
[`research/official-source/LICENSE`](research/official-source/LICENSE).

Most snapshots are pinned to commit
`d91ea855110dae533f0aada947b2b7d78cc8a4e1`; source URLs, hashes and retrieval
status are recorded in `research/official-source/manifest.json`.
Files prefixed `legacy-v1.4.5-` come from the official `v1.4.5` tag:

- [python/model.py](https://github.com/lightvector/KataGo/blob/v1.4.5/python/model.py)
- [python/export_model.py](https://github.com/lightvector/KataGo/blob/v1.4.5/python/export_model.py)

The full upstream runtime source and neural-network downloads are excluded
from Git. `deployment-assets.json` records their official URLs and SHA256
checksums. Network licensing information is available at
[KataGo network license](https://katagotraining.org/network_license/).

The one-time converter uses `tensorflow-checkpoint-reader` (MIT), installed
through `requirements-convert.txt`; its source is not bundled here.
