"""Model persistence utilities."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import lightgbm as lgb


class ModelStore:
    """Save/load LightGBM models with metadata."""

    def __init__(self, base_dir: str = "data/models") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: lgb.Booster,
        metadata: dict,
        version: str | None = None,
        *,
        update_latest: bool = True,
    ) -> str:
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
        if update_latest:
            (self.base_dir / "latest").write_text(version)
        return version

    def save_ensemble(
        self,
        models: dict[str, lgb.Booster],
        weights: dict[str, float],
        metadata: dict,
        version: str | None = None,
        *,
        update_latest: bool = True,
    ) -> str:
        version = version or self._next_version()
        version_dir = self.base_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        for key, model in models.items():
            model.save_model(str(version_dir / f"{key}.txt"))

        (version_dir / "ensemble_weights.json").write_text(json.dumps(weights, indent=2))
        metadata = {
            **metadata,
            "model_type": "ensemble",
            "ensemble_keys": list(models.keys()),
            "ensemble_weights": weights,
            "version": version,
            "created_at": datetime.now(UTC).isoformat(),
        }
        (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        if update_latest:
            (self.base_dir / "latest").write_text(version)
        return version

    def save_candidate(self, model: lgb.Booster, metadata: dict, version: str | None = None) -> str:
        return self.save(model, metadata, version=version, update_latest=False)

    def promote(self, version: str) -> str:
        actual_version = self._resolve_version(version)
        (self.base_dir / "latest").write_text(actual_version)
        return actual_version

    def version_dir(self, version: str = "latest") -> Path:
        return self.base_dir / self._resolve_version(version)

    def load(self, version: str = "latest") -> tuple[lgb.Booster, dict]:
        actual_version = self._resolve_version(version)
        version_dir = self.base_dir / actual_version
        model = lgb.Booster(model_file=str(version_dir / "model.txt"))
        metadata = json.loads((version_dir / "metadata.json").read_text())
        return model, metadata

    def load_ensemble(self, version: str = "latest") -> tuple[dict[str, lgb.Booster], dict[str, float], dict]:
        actual_version = self._resolve_version(version)
        version_dir = self.base_dir / actual_version
        metadata = json.loads((version_dir / "metadata.json").read_text())
        keys = metadata.get("ensemble_keys", [])
        models = {key: lgb.Booster(model_file=str(version_dir / f"{key}.txt")) for key in keys}
        weights = metadata.get("ensemble_weights", {})
        return models, weights, metadata

    def is_ensemble(self, version: str = "latest") -> bool:
        actual_version = self._resolve_version(version)
        version_dir = self.base_dir / actual_version
        metadata_path = version_dir / "metadata.json"
        if not metadata_path.exists():
            return False
        metadata = json.loads(metadata_path.read_text())
        return metadata.get("model_type") == "ensemble"

    def latest_version(self) -> str:
        latest_file = self.base_dir / "latest"
        if not latest_file.exists():
            raise FileNotFoundError("No latest model found")
        return latest_file.read_text().strip()

    def latest_metadata(self) -> dict:
        _, metadata = self.load("latest")
        return metadata

    def load_for_backtest(self, as_of_date: date, mode: str = "fixed", version: str | None = None) -> tuple[lgb.Booster, dict]:
        """Resolve a model version for backtests without leaking future artifacts."""
        if mode == "fixed":
            return self.load(version or "latest")

        candidates = []
        for metadata in self.list_versions():
            trained_through = self._trained_through_date(metadata)
            if trained_through is None:
                continue
            if trained_through <= as_of_date:
                candidates.append(metadata)

        if not candidates:
            raise FileNotFoundError(f"No model available for backtest date {as_of_date}")

        selected = max(candidates, key=lambda item: self._trained_through_date(item) or date.min)
        return self.load(str(selected["version"]))

    @staticmethod
    def _trained_through_date(metadata: dict) -> date | None:
        trained_through = metadata.get("trained_through_date")
        if trained_through:
            return date.fromisoformat(trained_through)

        split = metadata.get("split")
        if isinstance(split, dict):
            val = split.get("val")
            if isinstance(val, list) and len(val) == 2 and val[1]:
                return date.fromisoformat(val[1])

        data_range = metadata.get("data_range")
        if isinstance(data_range, dict) and data_range.get("end"):
            return date.fromisoformat(data_range["end"])

        return None

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
