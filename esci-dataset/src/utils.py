"""Utilities module for shared functionality across the project."""

import os
import subprocess


def get_git_info() -> dict[str, str]:
    """Get git commit hash (short) and branch name.

    Returns:
        Dict containing commit_hash (short) and branch_name
    """
    project_root = os.environ.get("root_folder")
    if not project_root:
        project_root = os.getcwd()

    try:
        # Use --short flag for consistent short hashes
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=project_root
        ).decode().strip()

        branch_name = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=project_root
        ).decode().strip()

        return {"commit_hash": commit_hash, "branch_name": branch_name}
    except Exception:
        # Fallback values if git commands fail
        return {"commit_hash": "unknown", "branch_name": "unknown"}
