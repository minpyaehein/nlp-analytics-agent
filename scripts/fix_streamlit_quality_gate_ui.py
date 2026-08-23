"""Patch Streamlit quality-gate error presentation safely."""

from __future__ import annotations

import ast
import py_compile
import re
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_APP_PATH = PROJECT_ROOT / "app" / "frontend" / "streamlit_app.py"
BACKUP_DIRECTORY = PROJECT_ROOT / "backups"

NEW_HANDLER_BODY = '''except ValueError as error:
    message = str(error)

    quality_gate_failure = (
        "not safe for the requested calculation" in message
        or "PDF extraction is not ready" in message
        or "quality_gate" in message
        or "analytics_ready" in message
    )

    if quality_gate_failure:
        st.error(
            "Revenue or profit calculation was blocked because "
            "the reconstructed PDF table does not contain enough "
            "usable numeric evidence."
        )
        st.warning(
            "Reset the previous OCR result, upload the high-resolution "
            "scan, rerun OCR, and confirm that the Revenue and Profit "
            "quality gates are ready before analyzing."
        )
        with st.expander("Technical quality-gate details"):
            st.code(message)
    else:
        st.error(f"Analysis failed: {message}")
'''


def read_source() -> str:
    """Read the current Streamlit application."""
    if not STREAMLIT_APP_PATH.exists():
        raise FileNotFoundError(f"Streamlit app does not exist: {STREAMLIT_APP_PATH}")
    source = STREAMLIT_APP_PATH.read_text(encoding="utf-8-sig")
    if not source.strip():
        raise ValueError("The Streamlit application is empty.")
    return source


def create_backup() -> Path:
    """Create a timestamped backup."""
    BACKUP_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIRECTORY / f"streamlit_app_before_quality_gate_ui_fix_{timestamp}.py"
    shutil.copy2(STREAMLIT_APP_PATH, backup_path)
    return backup_path


def indent_block(block: str, indentation: str) -> str:
    """Indent every non-empty line by the supplied indentation."""
    return "\n".join(
        indentation + line if line else ""
        for line in block.rstrip().splitlines()
    )


def replace_existing_handler(source: str) -> tuple[str, bool]:
    """Replace an existing ValueError handler before the generic handler."""
    pattern = re.compile(
        r"(?ms)^(?P<indent>[ \t]*)except ValueError as error:\s*\n"
        r".*?"
        r"(?=^(?P=indent)except Exception as error:)"
    )
    match = pattern.search(source)
    if match is None:
        return source, False

    replacement = indent_block(NEW_HANDLER_BODY, match.group("indent")) + "\n\n"
    return source[: match.start()] + replacement + source[match.end() :], True


def insert_before_generic_handler(source: str) -> tuple[str, bool]:
    """Insert the ValueError handler before the final generic handler."""
    candidates = list(
        re.finditer(r"(?m)^([ \t]*)except Exception as error:\s*$", source)
    )
    if not candidates:
        return source, False

    match = candidates[-1]
    replacement = indent_block(NEW_HANDLER_BODY, match.group(1)) + "\n\n"
    return source[: match.start()] + replacement + source[match.start() :], True


def validate_source(source: str) -> None:
    """Validate syntax and required user-interface behavior."""
    required_tokens = [
        "quality_gate_failure",
        "PDF extraction is not ready",
        "not safe for the requested calculation",
        "Technical quality-gate details",
        "Reset the previous OCR result",
        "st.code(message)",
    ]
    missing = [token for token in required_tokens if token not in source]
    if missing:
        raise RuntimeError("Patched Streamlit source is incomplete. Missing: " + ", ".join(missing))
    if source.count("Technical quality-gate details") != 1:
        raise RuntimeError("Expected exactly one technical-details expander.")
    ast.parse(source)


def main() -> None:
    """Patch and compile the Streamlit application."""
    print("=" * 80)
    print("InsightFlow AI Quality-Gate UI Fix")
    print("=" * 80)

    source = read_source()

    if "Technical quality-gate details" in source and "quality_gate_failure" in source:
        validate_source(source)
        print("The improved quality-gate UI already exists.")
        print("No changes were required.")
        return

    backup_path = create_backup()
    print(f"Backup created: {backup_path}")

    patched_source, patched = replace_existing_handler(source)
    if not patched:
        patched_source, patched = insert_before_generic_handler(source)
    if not patched:
        raise RuntimeError(
            "No suitable analysis exception handler was found. "
            "The Streamlit application was not modified."
        )

    validate_source(patched_source)
    STREAMLIT_APP_PATH.write_bytes(
        patched_source.encode("utf-8")
    )
    py_compile.compile(str(STREAMLIT_APP_PATH), doraise=True)

    print(f"Updated: {STREAMLIT_APP_PATH}")
    print("streamlit_app.py compilation passed.")
    print()
    print("=" * 80)
    print("STREAMLIT QUALITY-GATE UI FIX PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()

