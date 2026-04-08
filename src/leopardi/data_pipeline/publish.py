from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

from leopardi.data_pipeline.schemas import PublishResult


def parse_hf_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("hf://"):
        raise ValueError(f"Unsupported persistent URI: {uri}")
    payload = uri.removeprefix("hf://").strip("/")
    if not payload:
        raise ValueError("HF URI is missing repository id")
    pieces = payload.split("/", 1)
    repo_id = pieces[0]
    prefix = pieces[1] if len(pieces) == 2 else ""
    return repo_id, prefix


def publish_folder_to_hf(
    *,
    local_folder: str | Path,
    hf_uri: str,
    repo_type: str = "dataset",
    token_env: str = "HF_TOKEN",
    num_workers: int = 4,
) -> PublishResult:
    from huggingface_hub import HfApi

    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(f"{token_env} is required for HF publication")

    folder = Path(local_folder)
    repo_id, prefix = parse_hf_uri(hf_uri)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    upload_folder = folder
    with tempfile.TemporaryDirectory(prefix="leopardi-hf-upload-") as temp_root:
        if prefix:
            upload_folder = Path(temp_root) / prefix
            upload_folder.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(folder, upload_folder, dirs_exist_ok=True)
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=upload_folder,
            num_workers=num_workers,
            print_report=False,
        )
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    if prefix:
        files = [item for item in files if item.startswith(prefix)]
    return PublishResult(
        uri=hf_uri,
        repo_id=repo_id,
        verified_file_count=len(files),
    )
