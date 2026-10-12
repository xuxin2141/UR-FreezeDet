from pathlib import Path
import yaml


def load_config(name):
    path = Path(name)
    if path.suffix.lower() not in {".yaml", ".yml"}:
        path = Path(__file__).resolve().parents[2] / "configs" / f"{name}.yaml"
    with path.open(encoding="utf-8") as handle: return yaml.safe_load(handle)
