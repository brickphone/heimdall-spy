"""Launch the API in the container; MegaLoc and the gallery load during app startup."""

import os


def main() -> None:
    """Replace this process with uvicorn."""
    os.execvp("uvicorn", ["uvicorn", "swespy.api.main:app", "--host", "0.0.0.0", "--port", "8000"])


if __name__ == "__main__":
    main()
