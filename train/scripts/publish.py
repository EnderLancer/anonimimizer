#!/usr/bin/env python3
"""Publish a trained model folder to the Hugging Face Hub: public by default, --private keeps it private.

    uv run scripts/publish.py --model runs/wechsel_base/model                    # asks for the token, public repo <you>/wechsel_base
    HF_TOKEN=hf_... uv run scripts/publish.py --model runs/xlmr_large --repo my-org/xlmr_large --private

The token needs write access; it is taken from --token, then HF_TOKEN, then a hidden prompt. Only huggingface_hub is
imported, so the file also runs on its own in Colab:
    !python publish.py --model /content/drive/MyDrive/ai-camp/runs/wechsel_base/model
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

WEIGHTS = ("model.safetensors", "pytorch_model.bin")
DEFAULT_REPO = "wechsel_base"


def find_model_dir(path: Path) -> Path:
    """Accepts the model folder itself or the run folder that holds `model/`."""
    for d in (path, path / "model"):
        if (d / "config.json").exists() and any((d / w).exists() for w in WEIGHTS):
            return d.resolve()
    sys.exit(f"no model in {path}: expected config.json and one of {', '.join(WEIGHTS)}")


def model_card(model_dir: Path, repo: str, license_id: str | None = None, note: str | None = None) -> str:
    """A minimal card: base model from the run's config.json next to `model/`, entity types from the model config,
    `note` as a paragraph of its own (the training data and the credits its sources ask for)."""
    run_config = model_dir.parent / "config.json"
    base = json.loads(run_config.read_text(encoding="utf-8")).get("model") if run_config.exists() else None
    labels = json.loads((model_dir / "config.json").read_text(encoding="utf-8")).get("id2label", {}).values()
    types = list(dict.fromkeys(l.split("-", 1)[1] for l in labels if "-" in l))
    meta = ["language: uk", "library_name: transformers", "pipeline_tag: token-classification"]
    meta += [f"license: {license_id}"] if license_id else []
    meta += [f"base_model: {base}"] if base else []
    source = f", fine-tuned from [{base}](https://huggingface.co/{base})" if base else ""
    lines = ["---", *meta, "---", "", f"# {repo.split('/')[-1]}", "",
             f"Token classification model that finds personal data in Ukrainian court decisions for pseudonymization{source}."]
    if types:
        lines += ["", "Entity types: " + ", ".join(f"`{t}`" for t in types) + "."]
    if note:
        lines += ["", note.strip()]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model folder (config.json + weights) or the run folder that holds model/")
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"user/name or name under the token's account (default: {DEFAULT_REPO})")
    ap.add_argument("--token", default=None, help="HF token with write access (default: HF_TOKEN, then a hidden prompt)")
    ap.add_argument("--private", action="store_true", help="private repo; without it the repo is made public, also if it existed as private")
    ap.add_argument("--license", default="mit", help="licence id for the model card (used only when the card is created); the task rules "
                    "ask for an open licence, mit is the one of the training corpus lang-uk/court-decisions-uk and of the wechsel base model")
    ap.add_argument("--note", default=None, help="a paragraph for the model card (used only when the card is created), e.g. the training "
                    "data and the credit its sources ask for")
    ap.add_argument("--message", default=None, help="commit message")
    a = ap.parse_args()

    from huggingface_hub import HfApi

    model_dir = find_model_dir(Path(a.model))
    token = a.token or os.environ.get("HF_TOKEN") or getpass.getpass("Hugging Face token (write access): ").strip()
    if not token:
        sys.exit("no token given")
    api = HfApi(token=token)
    try:
        user = api.whoami()["name"]
    except Exception as e:
        sys.exit(f"the token was rejected: {e}")

    repo = a.repo if "/" in a.repo else f"{user}/{a.repo}"

    api.create_repo(repo, private=a.private, exist_ok=True)
    api.update_repo_settings(repo, private=a.private)  # an existing repo keeps its visibility on create_repo
    print(f"uploading {model_dir} -> {repo} ({'private' if a.private else 'public'})", file=sys.stderr)
    api.upload_folder(folder_path=str(model_dir), repo_id=repo, commit_message=a.message or f"Upload {repo.split('/')[-1]}")
    if not api.file_exists(repo, "README.md"):
        api.upload_file(path_or_fileobj=model_card(model_dir, repo, a.license, a.note).encode("utf-8"), path_in_repo="README.md",
                        repo_id=repo, commit_message="Add model card")
    print(f"https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
