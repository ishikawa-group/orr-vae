"""Repository and path helpers."""

from pathlib import Path


def get_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists() and (candidate / "src").exists():
            return candidate
    return Path.cwd()


def resolve_data_dir(base: str | Path | None = None) -> Path:
    if base is None:
        return get_repo_root() / "data"
    return Path(base).resolve()


def resolve_result_dir(base: str | Path | None = None) -> Path:
    if base is None:
        return get_repo_root() / "result"
    return Path(base).resolve()
