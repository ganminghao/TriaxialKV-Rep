"""Opt-in timing ranges for the mixed-precision KV profile (PROFILE_FINDINGS.md section 9).

Set ``SGLANG_TRIAXIAL_PROFILE=1`` to emit ``torch.profiler.record_function`` ranges whose
names start with a category tag (``P1::``, ``P2::``, ``P3::``, ``P4::``, ``K1::``, ``STEP::``).
The ranges only show up in a trace taken with ``/start_profile``.

When the variable is unset, ``prof_fn`` returns the function unchanged and ``prof_range``
returns one shared ``nullcontext``, so the serving path is unaffected. Throughput numbers
in the report are measured with the variable unset.
"""

import contextlib
import os

ENABLED = os.environ.get("SGLANG_TRIAXIAL_PROFILE", "0") == "1"

_NULL = contextlib.nullcontext()

if ENABLED:
    import functools

    from torch.profiler import record_function

    def prof_range(name: str):
        return record_function(name)

    def prof_fn(name: str):
        def deco(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                with record_function(name):
                    return fn(*args, **kwargs)

            return wrapper

        return deco

else:

    def prof_range(name: str):
        return _NULL

    def prof_fn(name: str):
        return lambda fn: fn
