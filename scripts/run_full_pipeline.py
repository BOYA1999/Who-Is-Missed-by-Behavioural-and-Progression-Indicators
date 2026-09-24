from hashlib import sha256
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "pisa2022" / "CY08MSP_STU_QQQ.SAV"
EXPECTED_DATA_SHA256 = "9e4eddacd25e10c145f7bfe817fc99a5b8df544536e5ea1dfd7230c988916a8c"


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if not DATA.is_file():
    raise SystemExit(f"Missing OECD file: {DATA}. See DATA.md.")
if file_hash(DATA) != EXPECTED_DATA_SHA256:
    raise SystemExit("The extracted OECD SAV file does not match the recorded SHA-256 in DATA.md.")

steps = [
    "run_main.py",
    "validate_main.py",
    "run_sensitivities.py",
    "validate_campaign.py",
    "run_visibility_profiles.py",
    "validate_visibility_profiles.py",
    "run_reviewer_revision.py",
    "validate_reviewer_revision.py",
    "run_profile_bootstrap_reselected.py",
    "validate_profile_bootstrap_reselected.py",
    "run_targeted_revision_round2.py",
]
for name in steps:
    print(f"\n=== {name} ===", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / name)], cwd=ROOT, check=True)

for name in ["run_stability.py", "within_school_overlap_from_oof.py"]:
    print(f"\n=== {name} ===", flush=True)
    subprocess.run(
        [sys.executable, str(ROOT / "artifacts" / "stability_revision_20260923" / name)],
        cwd=ROOT,
        check=True,
    )

print("\n=== validate_public_aggregates.py ===", flush=True)
subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "validate_public_aggregates.py")],
    cwd=ROOT,
    check=True,
)

print("\nFull local pipeline completed. Keep unit-level intermediates private.")
