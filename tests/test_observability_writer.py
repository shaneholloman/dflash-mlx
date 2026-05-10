# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import io

from dflash_mlx.diagnostics import TraceConfig
from dflash_mlx.observability import writer as writer_mod


def test_diagnostics_writer_reports_directory_failure(monkeypatch, tmp_path, capsys):
    diagnostic_writer = writer_mod._DiagnosticsJsonlWriter()

    def fail_makedirs(*_args, **_kwargs):
        raise OSError("no diagnostics dir")

    monkeypatch.setattr(writer_mod.os, "makedirs", fail_makedirs)

    diagnostic_writer.log_post(TraceConfig(log_dir=tmp_path), request_id=1)

    err = capsys.readouterr().err
    assert "diagnostics directory unavailable" in err
    assert "no diagnostics dir" in err


def test_diagnostics_writer_recovers_after_write_failure(monkeypatch, tmp_path, capsys):
    diagnostic_writer = writer_mod._DiagnosticsJsonlWriter()
    opened: list[object] = []

    class BrokenFile:
        def __init__(self) -> None:
            self.closed = False

        def write(self, _line):
            raise OSError("disk full")

        def close(self):
            self.closed = True

    def fake_open(*args, **kwargs):
        if not opened:
            fp = BrokenFile()
            opened.append(fp)
            return fp
        fp = io.StringIO()
        opened.append(fp)
        return fp

    monkeypatch.setattr(writer_mod, "open", fake_open, raising=False)
    trace = TraceConfig(log_dir=tmp_path)

    diagnostic_writer.log_post(trace, request_id=1)
    diagnostic_writer.log_post(trace, request_id=2)

    err = capsys.readouterr().err
    assert "diagnostics post write failed: disk full" in err
    assert opened[0].closed is True
    assert '"request_id":2' in opened[1].getvalue()
