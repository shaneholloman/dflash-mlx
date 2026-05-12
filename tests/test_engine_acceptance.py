# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import mlx.core as mx

from dflash_mlx.engine.acceptance import match_acceptance_length

def test_acceptance_length_cases():
    cases = [
        ([], [], 0),
        ([1, 2, 3, 4], [1, 2, 3, 4], 4),
        ([5, 5, 5], [1, 5, 5], 0),
        ([1, 2, 9, 4, 5], [1, 2, 3, 4, 5], 2),
        ([7, 7, 7, 0], [7, 7, 7, 1], 3),
    ]
    for drafted, posterior, expected in cases:
        out = match_acceptance_length(
            mx.array(drafted, dtype=mx.uint32),
            mx.array(posterior, dtype=mx.uint32),
        )
        assert int(out.item()) == expected
