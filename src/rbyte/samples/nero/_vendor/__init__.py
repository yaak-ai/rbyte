"""Verbatim copies of nutron-cli code that rbyte must compute bit-identically.

`hand_features.py` is nutron-cli's `runtime/jetson/hand_features.py`, the ONE
builder of the Revo2 hand features. Serving (the Orin) imports it; rbyte builds
the training rows with the same bytes. It is a pinned copy and not a path import
because rbyte runs on machines with no nutron-cli checkout (training hosts, CI).

Do not edit the copy. Edit the source in nutron-cli, then resync:

    python <nutron-cli>/runtime/jetson/hand_features_sync.py \\
        --vendor <rbyte>/src/rbyte/samples/nero/_vendor

and paste the three printed constants below. `tests/test_nero_robot.py` checks
the copy against these pins (always) and against the nutron-cli file (when a
checkout is present). See nutron-cli `docs/notes/HAND-FEATURES-VENDORING.md` and
rbyte `docs/nero_robot.md`.
"""

from typing import Final

HAND_FEATURES_SHA256: Final = (
    "09ccbd893b3f9f81e0c386863f61d46d3e96f5efb88f27331c6b6c761532017c"
)
HAND_FEATURES_VERSION: Final = 2
HAND_TOKEN_SPEC_VERSION: Final = 1
