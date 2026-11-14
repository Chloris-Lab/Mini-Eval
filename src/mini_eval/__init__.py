import sys

PROTOCOL_VERSION = 3

if sys.platform != "linux":
    raise SystemExit(
        f"mini-eval runs on the Linux GPU server only (this host is {sys.platform}): "
        "protocol v3 numbers are produced on the pod, with CUDA and full test sets."
    )

from mini_eval.cli import main
