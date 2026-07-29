from pathlib import Path


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
        for path in repository_root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and not any(part.startswith(".venv") for part in path.parts)
        and path.suffix.lower() in suffixes
        and any(
            marker in path.read_text(encoding="utf-8", errors="replace")
            for marker in forbidden
        )
    ]
    assert offenders == []
