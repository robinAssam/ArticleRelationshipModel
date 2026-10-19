

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (glob relative to the project root) -> copied to the same path in the Space
APP_FILES = [
    "app.py",
    "src/**/*.py",
    "data/annotations/*.json",
    ".streamlit/config.toml",
]
SPACE_FILES = PROJECT_ROOT / "deploy" / "hf_space"  # copied to the Space root
NEVER_UPLOAD = {"secrets.toml"}


def collect(staging: Path) -> list[str]:
    copied = []
    for pattern in APP_FILES:
        for src in sorted(PROJECT_ROOT.glob(pattern)):
            if src.name in NEVER_UPLOAD or "__pycache__" in src.parts:
                continue
            rel = src.relative_to(PROJECT_ROOT)
            dest = staging / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied.append(rel.as_posix())
    for src in sorted(SPACE_FILES.iterdir()):
        shutil.copy2(src, staging / src.name)
        copied.append(src.name)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--space", required=True,
                        help="Space id, e.g. robinassam/article-relationship-model")
    parser.add_argument("--private", action="store_true",
                        help="make the Space private (only you can open it)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the files that would be uploaded, upload nothing")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        files = collect(staging)
        assert not any(Path(f).name in NEVER_UPLOAD for f in files)
        print(f"{len(files)} files to upload:")
        for f in files:
            print(f"  {f}")
        if args.dry_run:
            return

        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.space, repo_type="space", space_sdk="docker",
                        private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=staging, repo_id=args.space, repo_type="space",
            commit_message="Deploy Article Relationship Model app",
            # remove files from earlier deploys that no longer exist locally
            delete_patterns=["src/**", "data/**", "*.py"],
        )
    print(f"\nUploaded. The Space builds itself (first build takes ~10 minutes):")
    print(f"  https://huggingface.co/spaces/{args.space}")
    print("Remember to add ANTHROPIC_API_KEY and APP_PASSWORD under Settings > "
          "Variables and secrets.")


if __name__ == "__main__":
    sys.exit(main())
