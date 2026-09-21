from hashlib import sha256
from pathlib import Path
import csv
import re


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST_SHA256.csv"
FORBIDDEN_NAMES = {
    "CY08MSP_STU_QQQ.SAV",
    "STU_QQQ_SPSS.zip",
    "oof_predictions.csv.gz",
    "sensitivity_predictions.csv.gz",
    "revision_predictions.csv.gz",
    "visibility_assignments.csv.gz",
    "fold_assignments.csv.gz",
    "school_multiplicities.csv.gz",
}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".yml", ".yaml", ".ps1", ".sh"}


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


files = [path for path in ROOT.rglob("*") if path.is_file()]
assert not FORBIDDEN_NAMES.intersection(path.name for path in files)
assert not any(path.suffix.lower() in {".sav", ".zsav", ".por", ".doc", ".docx"} for path in files)
assert not any(path.suffix.lower() in {".pyc", ".pyo"} or "__pycache__" in path.parts for path in files)
assert max(path.stat().st_size for path in files) < 100_000_000

pattern = re.compile(r"(?:[A-Za-z]:\\|file:///|Users[/\\]ADMIN)", re.IGNORECASE)
for path in files:
    if path.suffix.lower() in TEXT_SUFFIXES and path.name != MANIFEST.name and path.resolve() != Path(__file__).resolve():
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        assert not pattern.search(text), f"Local path found in {path.relative_to(ROOT)}"

if MANIFEST.is_file():
    with MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {path.relative_to(ROOT).as_posix() for path in files if path != MANIFEST}
    assert {row["relative_path"] for row in rows} == expected
    for row in rows:
        path = ROOT / row["relative_path"]
        assert path.stat().st_size == int(row["size_bytes"])
        assert digest(path) == row["sha256"]

print(f"PASS: {len(files)} public-safe files; maximum file size below GitHub's 100 MB limit.")
