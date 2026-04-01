"""Model persistence utilities."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb


class ModelStore:
    """Save/load LightGBM models with metadata."""

    def __init__(self, base_dir: str = "data/models") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model: lgb.Booster, metadata: dict, version: str | None = None) -> str:
        version = version or self._next_version()
        version_dir = self.base_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        model.save_model(str(version_dir / "model.txt"))
        metadata = {
            **metadata,
            "version": version,
            "created_at": datetime.now(UTC).isoformat(),
        }
        (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        (self.base_dir / "latest").write_text(version)
        return version

    def version_dir(self, version: str = "latest") -> Path:
        return self.base_dir / self._resolve_version(version)

    def load(self, version: str = "latest") -> tuple[lgb.Booster, dict]:
        actual_version = self._resolve_version(version)
        version_dir = self.base_dir / actual_version
        model = lgb.Booster(model_file=str(version_dir / "model.txt"))
        metadata = json.loads((version_dir / "metadata.json").read_text())
        return model, metadata

    def list_versions(self) -> list[dict]:
        versions: list[dict] = []
        for path in sorted(self.base_dir.iterdir()):
            if not path.is_dir():
                continue
            metadata_path = path / "metadata.json"
            if metadata_path.exists():
                versions.append(json.loads(metadata_path.read_text()))
        return versions

    def _resolve_version(self, version: str) -> str:
        if version != "latest":
            return version
        latest_file = self.base_dir / "latest"
        if not latest_file.exists():
            raise FileNotFoundError("No latest model found")
        return latest_file.read_text().strip()

    def _next_version(self) -> str:
        existing = [p.name for p in self.base_dir.iterdir() if p.is_dir() and p.name.startswith("v")]
        if not existing:
            return "v001"
        nums = [int(name[1:]) for name in existing if name[1:].isdigit()]
        return f"v{max(nums) + 1:03d}"
