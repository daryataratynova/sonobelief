"""Frozen checkpoints on the Hugging Face Hub.

    python scripts/weights.py download [--repo <user>/sonobelief]    -> weights/<component>/*.pt
    python scripts/weights.py upload   --repo <user>/sonobelief       (maintainers; needs `hf auth login`)
"""
import argparse
import os

from huggingface_hub import HfApi, snapshot_download

WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
DEFAULT_REPO = os.environ.get("SONOBELIEF_HF_REPO", "daryataratynova8/sonobelief")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["download", "upload"])
    ap.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face model repo id")
    args = ap.parse_args()
    assert args.repo, "pass --repo <user>/sonobelief or set SONOBELIEF_HF_REPO"
    if args.action == "download":
        snapshot_download(args.repo, allow_patterns=["*/*.pt", "*/*.json"], local_dir=WEIGHTS)
        print(f"checkpoints -> {WEIGHTS}")
    else:
        api = HfApi()
        api.create_repo(args.repo, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=WEIGHTS, repo_id=args.repo, repo_type="model",
                          allow_patterns=["README.md", "*/*.pt", "*/*.json"],
                          ignore_patterns=["*/embeddings_*", "*/z_L_*", "*/atypical_*", "*/hybrid_traj_*"])
        print(f"uploaded {WEIGHTS} -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
