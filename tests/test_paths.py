from pathlib import Path

from orr_vae.common.paths import get_repo_root, resolve_data_dir, resolve_result_dir


def test_resolve_default_dirs_are_paths():
    data_dir = resolve_data_dir()
    result_dir = resolve_result_dir()
    assert isinstance(data_dir, Path)
    assert isinstance(result_dir, Path)


def test_repo_root_exists():
    root = get_repo_root(Path(__file__))
    assert root.exists()
