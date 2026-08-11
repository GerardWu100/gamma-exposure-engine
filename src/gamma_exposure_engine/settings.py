"""Load runtime paths and offline-data settings for the project.

This module centralizes repo-relative configuration loading for the gamma
exposure engine. The offline pipeline and optional raw-data refresh utilities
consume the typed settings defined here so runtime defaults live in
``config.toml`` instead of being scattered across application code.
"""

from dataclasses import dataclass
from pathlib import Path

import tomllib

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.toml"
ENV_PATH = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class ClickHouseSettings:
    """Connection details for the local ClickHouse service."""

    host: str
    port: int
    user: str
    password: str | None
    secure: bool
    verify: bool


@dataclass(frozen=True)
class RawDataSettings:
    """Canonical raw-data settings for offline-first execution."""

    raw_data_dir: Path
    schema_version: int


@dataclass(frozen=True)
class ResearchSettings:
    """Research knobs shared by the CLI and notebook analytics."""

    near_spot_band_width: float
    abnormal_volume_window: int
    quantile_count: int
    pinning_candidate_count: int
    predictive_min_train_size: int
    near_spot_share_thresholds: tuple[float, ...]
    default_factor_name: str
    default_target_name: str
    bootstrap_iterations: int
    bootstrap_confidence_level: float
    regime_lookback_window: int
    ridge_alpha_candidates: tuple[float, ...]
    robustness_band_widths: tuple[float, ...]


@dataclass(frozen=True)
class AppSettings:
    """Runtime settings for the gamma exposure engine."""

    project_root: Path
    config_path: Path
    env_path: Path
    outputs_dir: Path
    symbol: str
    research: ResearchSettings
    clickhouse: ClickHouseSettings
    raw_data: RawDataSettings


def _read_env_file(env_path: Path) -> dict[str, str]:
    """Read simple ``KEY=VALUE`` pairs from the local ``.env`` file."""

    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key] = value

    return values


def _parse_bool(value: str, setting_name: str) -> bool:
    """Parse a strict boolean flag from configuration input."""

    normalized_value = value.strip().lower()

    if normalized_value == "true":
        return True

    if normalized_value == "false":
        return False

    raise ValueError(
        f"{setting_name} must be 'true' or 'false', received {value!r}.",
    )


def _resolve_clickhouse_credential(
    env: dict[str, str],
    env_key: str,
    require_clickhouse_credentials: bool,
) -> str | None:
    """Resolve one ClickHouse credential from ``.env`` under strict or app-only rules."""

    value = env.get(env_key)

    if value is not None:
        return value

    if require_clickhouse_credentials:
        raise ValueError(
            f"{env_key} is required when loading database settings.",
        )

    return None


def load_settings(
    project_root: Path | None = None,
    config_path: Path | None = None,
    env_path: Path | None = None,
    require_clickhouse_password: bool = True,
) -> AppSettings:
    """Load project settings from ``config.toml`` and ``.env``.

    Args:
        project_root:
            Repository root used to resolve relative paths.
        config_path:
            Optional override for the configuration file.
        env_path:
            Optional override for the environment file.
        require_clickhouse_password:
            When ``True``, fail fast if ``CLICKHOUSE_USER`` or
            ``CLICKHOUSE_PASSWORD`` is missing. Non-database entrypoints may
            set this to ``False``.

    Returns:
        AppSettings: Typed runtime settings for the application.
    """

    resolved_project_root = project_root or PROJECT_ROOT
    resolved_config_path = config_path or CONFIG_PATH
    resolved_env_path = env_path or ENV_PATH

    config = tomllib.loads(resolved_config_path.read_text())
    env = _read_env_file(resolved_env_path)

    project_config = config["project"]
    path_config = config["paths"]
    research_config = config["research"]
    clickhouse_config = config["clickhouse"]
    raw_data_config = config["raw_data"]

    outputs_dir = resolved_project_root / Path(path_config["outputs_dir"])
    research = ResearchSettings(
        near_spot_band_width=float(research_config["near_spot_band_width"]),
        abnormal_volume_window=int(research_config["abnormal_volume_window"]),
        quantile_count=int(research_config["quantile_count"]),
        pinning_candidate_count=int(research_config["pinning_candidate_count"]),
        predictive_min_train_size=int(research_config["predictive_min_train_size"]),
        near_spot_share_thresholds=tuple(
            float(value) for value in research_config["near_spot_share_thresholds"]
        ),
        default_factor_name=research_config["default_factor_name"],
        default_target_name=research_config["default_target_name"],
        bootstrap_iterations=int(research_config["bootstrap_iterations"]),
        bootstrap_confidence_level=float(research_config["bootstrap_confidence_level"]),
        regime_lookback_window=int(research_config["regime_lookback_window"]),
        ridge_alpha_candidates=tuple(
            float(value) for value in research_config["ridge_alpha_candidates"]
        ),
        robustness_band_widths=tuple(
            float(value) for value in research_config["robustness_band_widths"]
        ),
    )
    clickhouse_user = _resolve_clickhouse_credential(
        env=env,
        env_key="CLICKHOUSE_USER",
        require_clickhouse_credentials=require_clickhouse_password,
    )
    clickhouse_password = _resolve_clickhouse_credential(
        env=env,
        env_key="CLICKHOUSE_PASSWORD",
        require_clickhouse_credentials=require_clickhouse_password,
    )
    clickhouse = ClickHouseSettings(
        host=env.get("CLICKHOUSE_HOST", clickhouse_config["host"]),
        port=int(env.get("CLICKHOUSE_PORT", clickhouse_config["port"])),
        user=clickhouse_user or "",
        password=clickhouse_password,
        secure=_parse_bool(
            env.get("CLICKHOUSE_SECURE", str(clickhouse_config["secure"])),
            "CLICKHOUSE_SECURE",
        ),
        verify=_parse_bool(
            env.get("CLICKHOUSE_VERIFY", str(clickhouse_config["verify"])),
            "CLICKHOUSE_VERIFY",
        ),
    )
    raw_data = RawDataSettings(
        raw_data_dir=resolved_project_root / Path(raw_data_config["raw_data_dir"]),
        schema_version=int(raw_data_config["schema_version"]),
    )

    return AppSettings(
        project_root=resolved_project_root,
        config_path=resolved_config_path,
        env_path=resolved_env_path,
        outputs_dir=outputs_dir,
        symbol=project_config["symbol"],
        research=research,
        clickhouse=clickhouse,
        raw_data=raw_data,
    )
