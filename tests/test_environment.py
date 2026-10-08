"""Platform detection and resolution of ``auto`` simulator settings."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from offroad_autonomy.utils import environment
from offroad_autonomy.utils.config import load_config
from offroad_autonomy.utils.environment import PlatformFacts, resolve_auto

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

WINDOWS = PlatformFacts(os_name="windows")
WSL_NAT = PlatformFacts(
    os_name="linux", is_wsl=True, wsl_networking="nat", default_gateway="172.23.224.1"
)
WSL_MIRRORED = PlatformFacts(
    os_name="linux", is_wsl=True, wsl_networking="mirrored", default_gateway="192.168.1.1"
)
MACOS = PlatformFacts(os_name="macos")
LINUX = PlatformFacts(os_name="linux", default_gateway="192.168.1.1")

AUTO_BNG = {"home": "", "host": "auto", "launch": "auto", "camera_transport": "auto"}
AUTO_UI = {"display_async": "auto"}


def _resolve(facts, **bng_overrides):
    bng = dict(AUTO_BNG, **bng_overrides)
    return resolve_auto(bng, dict(AUTO_UI), facts)


def test_windows_attaches_to_localhost_over_shared_memory():
    bng, ui = _resolve(WINDOWS)
    assert bng["host"] == "localhost"
    assert bng["launch"] is False
    assert bng["camera_transport"] == "shared_memory"
    assert ui["display_async"] is True


def test_windows_launches_only_when_home_is_set():
    bng, _ = _resolve(WINDOWS, home="E:\\BeamNG.tech")
    assert bng["launch"] is True


def test_existing_windows_terminal_uses_saved_home(monkeypatch):
    for name in ("BEAMNG_HOME", "BEAMNG_LAUNCH", "BEAMNG_HOST"):
        monkeypatch.delenv(name, raising=False)
    with (
        patch.object(environment, "detect_platform", return_value=WINDOWS),
        patch.object(environment, "saved_windows_beamng_home", return_value="E:/BeamNG.tech"),
    ):
        cfg = load_config(DEFAULT_YAML)
    assert cfg.beamng_home == "E:/BeamNG.tech"
    assert cfg.beamng_launch is True


@pytest.mark.parametrize(
    "overrides,facts,env_home",
    [
        ({"home": "D:/BeamNG.tech"}, WINDOWS, None),
        ({"launch": False}, WINDOWS, None),
        ({"host": "192.168.1.50"}, WINDOWS, None),
        ({}, WSL_NAT, None),
        ({}, WINDOWS, ""),
    ],
)
def test_saved_home_does_not_override_explicit_or_remote_settings(
    monkeypatch, overrides, facts, env_home
):
    from offroad_autonomy.utils.config import _resolve_platform_settings

    monkeypatch.delenv("BEAMNG_HOME", raising=False)
    if env_home is not None:
        monkeypatch.setenv("BEAMNG_HOME", env_home)
    with (
        patch.object(environment, "detect_platform", return_value=facts),
        patch.object(environment, "saved_windows_beamng_home", side_effect=AssertionError),
    ):
        bng, _ = _resolve_platform_settings(dict(AUTO_BNG, **overrides), AUTO_UI)
    assert bng["home"] == overrides.get("home", "")


@pytest.mark.parametrize("missing", [False, True])
def test_saved_windows_home_reads_user_registry(monkeypatch, missing):
    registry = MagicMock()
    registry.QueryValueEx.return_value = ("E:/BeamNG.tech", registry.REG_SZ)
    if missing:
        registry.OpenKey.side_effect = FileNotFoundError
    monkeypatch.setattr(environment.sys, "platform", "win32")
    with patch.dict("sys.modules", {"winreg": registry}):
        assert environment.saved_windows_beamng_home() == ("" if missing else "E:/BeamNG.tech")
    registry.OpenKey.assert_called_once_with(registry.HKEY_CURRENT_USER, "Environment")


def test_wsl_nat_attaches_to_the_windows_gateway_over_socket():
    bng, ui = _resolve(WSL_NAT)
    assert bng["host"] == "172.23.224.1"
    assert bng["launch"] is False
    assert bng["camera_transport"] == "socket"
    assert ui["display_async"] is False


def test_wsl_mirrored_uses_localhost_but_still_socket():
    bng, _ = _resolve(WSL_MIRRORED, home="/mnt/e/BeamNG.tech")
    assert bng["host"] == "localhost"
    assert bng["launch"] is False
    assert bng["camera_transport"] == "socket"


def test_macos_has_no_default_host_and_draws_inline():
    bng, ui = _resolve(MACOS)
    assert bng["host"] == ""
    assert bng["launch"] is False
    assert bng["camera_transport"] == "socket"
    assert ui["display_async"] is False


def test_native_linux_has_no_default_host_but_draws_async():
    bng, ui = _resolve(LINUX)
    assert bng["host"] == ""
    assert ui["display_async"] is True


def test_remote_host_on_windows_uses_socket_and_never_launches():
    bng, _ = _resolve(WINDOWS, host="192.168.1.50", home="E:\\BeamNG.tech")
    assert bng["launch"] is False
    assert bng["camera_transport"] == "socket"


def test_explicit_values_are_kept():
    bng, ui = resolve_auto(
        {"host": "10.0.0.2", "launch": True, "camera_transport": "shared_memory"},
        {"display_async": True},
        MACOS,
    )
    assert bng == {"host": "10.0.0.2", "launch": True, "camera_transport": "shared_memory"}
    assert ui == {"display_async": True}


def test_resolve_does_not_mutate_its_inputs():
    bng = dict(AUTO_BNG)
    ui = dict(AUTO_UI)
    resolve_auto(bng, ui, WINDOWS)
    assert bng == AUTO_BNG
    assert ui == AUTO_UI


ROUTE_TABLE = (
    "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n"
    "eth0\t00000000\t01E017AC\t0003\t0\t0\t0\t00000000\t0\t0\t0\n"
    "eth0\t00E017AC\t00000000\t0001\t0\t0\t0\t00F0FFFF\t0\t0\t0\n"
)


def test_default_gateway_is_decoded_from_the_route_table():
    assert environment.parse_default_gateway(ROUTE_TABLE) == "172.23.224.1"


def test_route_table_without_a_default_route_has_no_gateway():
    header = ROUTE_TABLE.splitlines()[0]
    assert environment.parse_default_gateway(header + "\n") == ""


def test_missing_wslinfo_is_treated_as_nat():
    with patch.object(environment.subprocess, "run", side_effect=FileNotFoundError):
        assert environment.wsl_networking_mode() == "nat"


def test_slow_wslinfo_is_treated_as_nat():
    timeout = subprocess.TimeoutExpired(cmd="wslinfo", timeout=2)
    with patch.object(environment.subprocess, "run", side_effect=timeout):
        assert environment.wsl_networking_mode() == "nat"


def test_load_config_resolves_auto_from_the_detected_platform(tmp_path):
    path = tmp_path / "auto.yaml"
    path.write_text(
        "beamng:\n  host: auto\n  launch: auto\n  camera_transport: auto\n"
        "ui:\n  display_async: auto\n",
        encoding="utf-8",
    )
    with patch.object(environment, "detect_platform", return_value=WSL_NAT):
        cfg = load_config(path)
    assert cfg.beamng_host == "172.23.224.1"
    assert cfg.beamng_launch is False
    assert cfg.beamng_camera_transport == "socket"
    assert cfg.ui_display_async is False


def test_beamng_host_env_wins_over_auto(tmp_path, monkeypatch):
    path = tmp_path / "auto.yaml"
    path.write_text("beamng:\n  host: auto\n  camera_transport: auto\n", encoding="utf-8")
    monkeypatch.setenv("BEAMNG_HOST", "10.0.0.2")
    with patch.object(environment, "detect_platform", return_value=WINDOWS):
        cfg = load_config(path)
    assert cfg.beamng_host == "10.0.0.2"
    assert cfg.beamng_camera_transport == "socket"


def test_explicit_config_never_inspects_the_machine(tmp_path):
    path = tmp_path / "explicit.yaml"
    path.write_text(
        "beamng:\n  host: localhost\n  launch: false\n  camera_transport: socket\n"
        "ui:\n  display_async: true\n",
        encoding="utf-8",
    )
    with patch.object(environment, "detect_platform", side_effect=AssertionError):
        cfg = load_config(path)
    assert cfg.beamng_host == "localhost"


def test_connect_without_a_host_names_beamng_host():
    from offroad_autonomy.simulation.beamng_client import BeamNGClient
    from offroad_autonomy.types import PipelineConfig

    client = BeamNGClient(PipelineConfig(beamng_host=""))
    with pytest.raises(RuntimeError, match="BEAMNG_HOST"):
        client.connect()


def test_shipped_default_config_runs_unchanged_on_wsl(monkeypatch):
    monkeypatch.delenv("BEAMNG_HOST", raising=False)
    monkeypatch.delenv("BEAMNG_HOME", raising=False)
    monkeypatch.delenv("BEAMNG_LAUNCH", raising=False)
    with patch.object(environment, "detect_platform", return_value=WSL_NAT):
        cfg = load_config(DEFAULT_YAML)
    assert cfg.beamng_host == "172.23.224.1"
    assert cfg.beamng_launch is False
    assert cfg.beamng_camera_transport == "socket"
    assert cfg.ui_display_async is False
