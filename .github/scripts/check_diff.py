import json
import sys
from typing import Dict

DIRS = [
    ".",
]

if __name__ == "__main__":
    files = sys.argv[1:]

    dirs_to_run: Dict[str, set] = {
        "lint": set(DIRS),
        "test": set(DIRS),
    }

    outputs = {
        "dirs-to-lint": list(dirs_to_run["lint"] | dirs_to_run["test"]),
        "dirs-to-test": list(dirs_to_run["test"]),
    }
    for key, value in outputs.items():
        json_output = json.dumps(value)
        print(f"{key}={json_output}")  # noqa: T201
