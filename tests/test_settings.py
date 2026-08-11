"""Tests for runtime settings loading."""

from pathlib import Path

import pytest
from gamma_exposure_engine.settings import load_settings


def write_settings_files(
    tmp_path: Path,
    env_text: str,
    secure_value: str = "true",
) -> tuple[Path, Path, Path]:
    """Write hermetic config and environment files for settings tests."""

    project_root = tmp_path
    config_path = project_root / "config.toml"
    env_path = project_root / ".env"
    config_path.write_text(
        """
[project]
symbol = "SPY"

[paths]
outputs_dir = "artifacts/outputs"

[research]
near_spot_band_width = 0.02
abnormal_volume_window = 20
quantile_count = 5
pinning_candidate_count = 5
predictive_min_train_size = 20
near_spot_share_thresholds = [0.2, 0.4, 0.6]
default_factor_name = "total_open_interest_weighted_gamma"
default_target_name = "next_day_realized_variance"
bootstrap_iterations = 1000
bootstrap_confidence_level = 0.95
regime_lookback_window = 20
ridge_alpha_candidates = [0.01, 0.1, 1.0, 10.0, 100.0]
robustness_band_widths = [0.01, 0.03, 0.05]

[clickhouse]
host = "config-host"
port = 9000
secure = false
verify = false

[raw_data]
raw_data_dir = "data/raw"
schema_version = 1
"""
    )
    env_path.write_text(
        f"""
CLICKHOUSE_HOST=env-host
CLICKHOUSE_PORT=50050
CLICKHOUSE_USER=env-user
{env_text}
CLICKHOUSE_SECURE={secure_value}
CLICKHOUSE_VERIFY=true
"""
    )

    return project_root, config_path, env_path


def test_load_settings_uses_explicit_paths_and_does_not_create_outputs_dir(
    tmp_path: Path,
) -> None:
    """Load settings from hermetic temp files without touching the real repo."""

    project_root, config_path, env_path = write_settings_files(
        tmp_path=tmp_path,
        env_text="CLICKHOUSE_PASSWORD=env-password",
    )
    outputs_dir = project_root / "artifacts" / "outputs"

    settings = load_settings(
        project_root=project_root,
        config_path=config_path,
        env_path=env_path,
    )

    assert settings.project_root == project_root
    assert settings.config_path == config_path
    assert settings.outputs_dir == outputs_dir
    assert settings.symbol == "SPY"
    assert settings.research.near_spot_band_width == 0.02
    assert settings.research.abnormal_volume_window == 20
    assert settings.research.quantile_count == 5
    assert settings.research.pinning_candidate_count == 5
    assert settings.research.predictive_min_train_size == 20
    assert settings.research.near_spot_share_thresholds == (0.2, 0.4, 0.6)
    assert settings.research.default_factor_name == "total_open_interest_weighted_gamma"
    assert settings.research.default_target_name == "next_day_realized_variance"
    assert settings.research.bootstrap_iterations == 1000
    assert settings.research.bootstrap_confidence_level == 0.95
    assert settings.research.regime_lookback_window == 20
    assert settings.research.ridge_alpha_candidates == (0.01, 0.1, 1.0, 10.0, 100.0)
    assert settings.research.robustness_band_widths == (0.01, 0.03, 0.05)
    assert settings.clickhouse.host == "env-host"
    assert settings.clickhouse.port == 50050
    assert settings.clickhouse.user == "env-user"
    assert settings.clickhouse.password == "env-password"
    assert settings.clickhouse.secure is True
    assert settings.clickhouse.verify is True
    assert settings.raw_data.raw_data_dir == project_root / "data/raw"
    assert settings.raw_data.schema_version == 1
    assert not outputs_dir.exists()


def test_load_settings_allows_missing_clickhouse_password_for_non_database_path(
    tmp_path: Path,
) -> None:
    """Allow app-only settings loads to skip the database password."""

    project_root, config_path, env_path = write_settings_files(
        tmp_path=tmp_path,
        env_text="",
    )

    settings = load_settings(
        project_root=project_root,
        config_path=config_path,
        env_path=env_path,
        require_clickhouse_password=False,
    )

    assert settings.clickhouse.password is None
    assert settings.clickhouse.host == "env-host"


def test_load_settings_rejects_missing_clickhouse_password_for_database_path(
    tmp_path: Path,
) -> None:
    """Keep database settings loads strict by default."""

    project_root, config_path, env_path = write_settings_files(
        tmp_path=tmp_path,
        env_text="",
    )

    with pytest.raises(ValueError, match="CLICKHOUSE_PASSWORD"):
        load_settings(
            project_root=project_root,
            config_path=config_path,
            env_path=env_path,
        )


def test_load_settings_rejects_missing_clickhouse_user_when_password_present(
    tmp_path: Path,
) -> None:
    """Require both database credentials when strict loading is enabled."""

    project_root, config_path, env_path = write_settings_files(
        tmp_path=tmp_path,
        env_text="CLICKHOUSE_PASSWORD=env-password",
    )
    env_path.write_text(env_path.read_text().replace("CLICKHOUSE_USER=env-user\n", ""))

    with pytest.raises(ValueError, match="CLICKHOUSE_USER"):
        load_settings(
            project_root=project_root,
            config_path=config_path,
            env_path=env_path,
        )


def test_load_settings_rejects_invalid_boolean_strings(tmp_path: Path) -> None:
    """Fail fast when boolean values are not valid true or false strings."""

    project_root, config_path, env_path = write_settings_files(
        tmp_path=tmp_path,
        env_text="CLICKHOUSE_PASSWORD=env-password",
        secure_value="yes",
    )

    with pytest.raises(ValueError, match="CLICKHOUSE_SECURE"):
        load_settings(
            project_root=project_root,
            config_path=config_path,
            env_path=env_path,
        )


def test_repo_config_points_to_canonical_raw_directory() -> None:
    """The shipped config should target data/raw for canonical inputs."""

    settings = load_settings(require_clickhouse_password=False)

    assert settings.research.near_spot_band_width == 0.02
    assert settings.research.pinning_candidate_count == 5
    assert settings.raw_data.raw_data_dir == settings.project_root / "data/raw"
