"""Wait for Qdrant, then launch the unchanged API in the container."""

import os
import time
from urllib.error import URLError
from urllib.request import urlopen


def main() -> None:
    """Wait at most 60 seconds for readiness and replace this process with uvicorn."""
    host = os.environ.get("QDRANT_HOST", "qdrant")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    deadline = time.monotonic() + 60
    while True:
        try:
            with urlopen(f"http://{host}:{port}/readyz", timeout=2) as response:
                if response.status == 200:
                    break
        except (URLError, TimeoutError):
            pass
        if time.monotonic() >= deadline:
            raise SystemExit("Qdrant did not become ready within 60 seconds.")
        time.sleep(1)
    os.execvp("uvicorn", ["uvicorn", "swespy.api.main:app", "--host", "0.0.0.0", "--port", "8000"])


if __name__ == "__main__":
    main()
