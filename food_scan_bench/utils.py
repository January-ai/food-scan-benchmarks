import base64
from pathlib import Path


def img2b64(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/jpeg;base64,{encoded}"

def ensure_dir(dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)