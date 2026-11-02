"""Fetch only the official MuJoCo files, with an explicit size limit."""
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

REPO = "Vector-Wangel/XLeRobot"
COMMIT = "b017b5e6354bd9f61f4247a920c72622ca0aade0"
PREFIX = "simulation/mujoco/"
DEST = Path("third_party/XLeRobot")
LIMIT = 100 * 1024 * 1024


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "PAI-step01-model-fetch"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read(LIMIT + 1)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            print(f"Temporary download error; retry {attempt + 1}/2", flush=True)
            time.sleep(2)


def main():
    tree = json.loads(fetch(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"))
    if tree.get("truncated"):
        raise RuntimeError("Truncated upstream tree; inspect before downloading")
    files = [entry for entry in tree["tree"] if entry["type"] == "blob" and
             (entry["path"] == "LICENSE" or entry["path"] in
              (PREFIX + "scene.xml", PREFIX + "xlerobot.xml", PREFIX + "README.md") or
              entry["path"].startswith(PREFIX + "assets/"))]
    size = sum(entry["size"] for entry in files)
    print(json.dumps({"commit": COMMIT, "file_count": len(files), "total_bytes": size}), flush=True)
    if size > LIMIT or not any(e["path"] == "LICENSE" for e in files):
        raise RuntimeError("Model size exceeds 100 MiB or root license missing; review required")
    manifest = {"repository": f"https://github.com/{REPO}", "commit": COMMIT,
                "total_bytes": size, "files": []}
    for entry in files:
        rel = Path(entry["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError("Invalid upstream path")
        target = DEST / rel
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{rel.as_posix()}"
        data = target.read_bytes() if target.exists() else fetch(url)
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if len(data) != entry["size"] or blob != entry["sha"]:
            raise RuntimeError(f"Upstream hash mismatch or modified local asset: {rel}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open("xb") as output:
                output.write(data)
        manifest["files"].append({"path": rel.as_posix(), "bytes": len(data),
                                  "sha256": hashlib.sha256(data).hexdigest()})
    output = Path("artifacts/step01/model_manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(files)} verified files; no upstream scripts executed.", flush=True)


if __name__ == "__main__":
    main()
