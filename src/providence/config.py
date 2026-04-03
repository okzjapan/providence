from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_BANKROLL_JPY = 1_000_000


class Settings(BaseSettings):
    db_url: str = "sqlite:///data/providence.db"
    scrape_interval_sec: float = 2.0
    scrape_max_retries: int = 3
    scrape_timeout_sec: float = 30.0
    autorace_jp_base_url: str = "https://autorace.jp"
    oddspark_base_url: str = "https://www.oddspark.com"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="PROVIDENCE_")

    def ensure_data_dir(self) -> None:
        Path("data").mkdir(exist_ok=True)
        Path("data/logs").mkdir(exist_ok=True)


def get_settings() -> Settings:
    return Settings()
