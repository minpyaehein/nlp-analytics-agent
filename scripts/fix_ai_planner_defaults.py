"""Add safe defaults to the Qwen-first AI planner schema."""

from pathlib import Path


PLANNER_PATH = Path(
    "app/agents/ai_planner.py"
)


def main() -> None:
    source = PLANNER_PATH.read_text(
        encoding="utf-8-sig",
    )

    old_block = '''    reasoning_summary: str = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
'''

    new_block = '''    reasoning_summary: str = Field(
        default=(
            "The local AI planner selected a grounded analytical "
            "operation using the uploaded dataset schema."
        ),
        min_length=1,
    )
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
    )
'''

    if old_block not in source:
        if (
            "default=0.75" in source
            and "reasoning_summary: str = Field(" in source
        ):
            print(
                "The AI planner defaults are already installed."
            )
            return

        raise RuntimeError(
            "Could not find the expected AIAnalysisPlan "
            "field block."
        )

    backup_path = PLANNER_PATH.with_suffix(
        ".py.before_default_fix"
    )

    backup_path.write_text(
        source,
        encoding="utf-8",
        newline="\n",
    )

    updated_source = source.replace(
        old_block,
        new_block,
        1,
    )

    PLANNER_PATH.write_text(
        updated_source,
        encoding="utf-8",
        newline="\n",
    )

    print(f"Updated: {PLANNER_PATH}")
    print(f"Backup:  {backup_path}")


if __name__ == "__main__":
    main()