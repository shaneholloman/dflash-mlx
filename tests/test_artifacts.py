# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import subprocess

import pytest

from dflash_mlx import artifacts


def test_git_metadata_degrades_on_git_command_failure(monkeypatch):
    def fail_git(*_args, **_kwargs):
        raise subprocess.CalledProcessError(128, ["git"])

    monkeypatch.setattr(artifacts.subprocess, "check_output", fail_git)

    assert artifacts._git(["rev-parse", "HEAD"]) == "unknown"
    assert artifacts._git_dirty() is True


def test_git_metadata_propagates_programmer_errors(monkeypatch):
    def broken_helper(*_args, **_kwargs):
        raise TypeError("broken helper contract")

    monkeypatch.setattr(artifacts.subprocess, "check_output", broken_helper)

    with pytest.raises(TypeError, match="broken helper contract"):
        artifacts._git(["rev-parse", "HEAD"])
    with pytest.raises(TypeError, match="broken helper contract"):
        artifacts._git_dirty()
