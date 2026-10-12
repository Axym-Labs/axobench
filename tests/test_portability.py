from pathlib import Path
import subprocess


def _repository_files(root: Path) -> list[Path]:
    """Tracked and unignored files: what the public repository contains, not local generation artifacts."""
    try:
        listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root,
                                capture_output=True, text=True, check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return [path for path in root.rglob("*") if ".git" not in path.parts]
    return [root / name for name in listed]


def test_public_repository_contains_no_machine_specific_home_paths():
    repository_root = Path(__file__).resolve().parents[1]
    forbidden = ("/home/" + "davwis/", "C:/" + "Users/")
    suffixes = {
        ".json",
        ".md",
        ".py",
        ".sh",
        ".swc",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
    offenders = [
        str(path.relative_to(repository_root))
        for path in _repository_files(repository_root)
        if path.is_file()
        and not any(part.startswith(".venv") for part in path.parts)
        and path.suffix.lower() in suffixes
        and any(
            marker in path.read_text(encoding="utf-8", errors="replace")
            for marker in forbidden
        )
    ]
    assert offenders == []
