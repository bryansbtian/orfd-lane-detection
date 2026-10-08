"""Derives the machine-specific simulator settings left as ``auto`` in a config.

Where BeamNG lives, how frames reach us and whether the dashboard may run on
its own thread all follow from the platform, so one checkout runs on Windows,
WSL2, macOS and the Jetson without a per-machine config file.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

AUTO = "auto"
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


def saved_windows_beamng_home() -> str:
    """Read the user setting even when an existing terminal has a stale environment."""
    if sys.platform != "win32":
        return ""
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, kind = winreg.QueryValueEx(key, "BEAMNG_HOME")
        if kind == winreg.REG_EXPAND_SZ:
            return winreg.ExpandEnvironmentStrings(value)
        if kind == winreg.REG_SZ:
            return value
    except OSError:
        # Missing settings or denied registry access leave attach mode available.
        pass
    return ""


@dataclass(frozen=True)
class PlatformFacts:
    os_name: str
    is_wsl: bool = False
    wsl_networking: str = ""
    default_gateway: str = ""


def parse_default_gateway(route_table: str) -> str:
    """/proc/net/route stores addresses as little-endian hex, and reading it
    avoids depending on the ``ip`` tool being installed."""
    for line in route_table.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 3 or fields[1] != "00000000":
            continue
        octets = bytes.fromhex(fields[2])[::-1]
        return ".".join(str(octet) for octet in octets)
    return ""


def wsl_networking_mode() -> str:
    """WSL builds before mirrored networking have no ``wslinfo``, and NAT was
    their only mode, so any failure to ask means NAT."""
    try:
        result = subprocess.run(
            ["wslinfo", "--networking-mode"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "nat"
    if result.stdout.strip() == "mirrored":
        return "mirrored"
    return "nat"


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def detect_platform() -> PlatformFacts:
    if sys.platform == "win32":
        return PlatformFacts(os_name="windows")
    if sys.platform == "darwin":
        return PlatformFacts(os_name="macos")
    gateway = parse_default_gateway(_read_text("/proc/net/route"))
    if "microsoft" not in _read_text("/proc/sys/kernel/osrelease").lower():
        return PlatformFacts(os_name="linux", default_gateway=gateway)
    return PlatformFacts(
        os_name="linux",
        is_wsl=True,
        wsl_networking=wsl_networking_mode(),
        default_gateway=gateway,
    )


def needs_detection(bng: dict, ui: dict) -> bool:
    keys = (bng.get("host"), bng.get("launch"), bng.get("camera_transport"))
    return AUTO in keys or ui.get("display_async") == AUTO


def _auto_host(facts: PlatformFacts) -> str:
    if facts.os_name == "windows":
        return "localhost"
    if facts.is_wsl and facts.wsl_networking == "mirrored":
        return "localhost"
    if facts.is_wsl:
        # Under NAT the Windows host is the VM's gateway, and it can change on
        # every WSL restart, so it is read fresh rather than written down.
        return facts.default_gateway
    # BeamNG.tech runs only on Windows, so nothing here can guess its address.
    return ""


def resolve_auto(bng: dict, ui: dict, facts: PlatformFacts) -> tuple[dict, dict]:
    bng = dict(bng)
    ui = dict(ui)

    if bng.get("host") == AUTO:
        bng["host"] = _auto_host(facts)
    # Shared memory cannot cross the WSL2 VM boundary even when mirrored
    # networking makes the host localhost, and only a native Windows client
    # can start the Windows simulator.
    local_windows = facts.os_name == "windows" and bng.get("host") in LOCAL_HOSTS

    if bng.get("launch") == AUTO:
        bng["launch"] = local_windows and bool(bng.get("home"))
    if bng.get("camera_transport") == AUTO:
        bng["camera_transport"] = "socket"
        if local_windows:
            bng["camera_transport"] = "shared_memory"
    if ui.get("display_async") == AUTO:
        # macOS only drives OpenCV windows from the main thread, and WSLg hangs
        # on the first frame drawn from another thread.
        ui["display_async"] = facts.os_name != "macos" and not facts.is_wsl
    return bng, ui
