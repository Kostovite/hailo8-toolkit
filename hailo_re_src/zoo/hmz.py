"""hailomz without write access to the SDK env: hailo_sdk_client.tools.cmd_utils copies autocompletion files into
<env>/bin on first import; report that directory as existing so the read-only env is left untouched.

Usage: python hmz.py <hailomz args...>
"""
import pathlib
import sys

_exists = pathlib.Path.exists
pathlib.Path.exists = lambda self, *a, **k: True if self.name == "hailo_autocompletion" else _exists(self, *a, **k)
from hailo_model_zoo.main import main  # noqa: E402

sys.exit(main())
