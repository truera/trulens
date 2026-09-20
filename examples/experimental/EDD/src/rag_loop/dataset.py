import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {"input", "expected_output", "metadata", "label", "critique"}


def load_items(path: Path) -> list[dict[str, Any]]:
    with path.open() as dataset_file:
        return [json.loads(line) for line in dataset_file if line.strip()]


def validate_items(items: list[dict[str, Any]]) -> None:
    for position, item in enumerate(items, start=1):
        missing = REQUIRED_FIELDS.difference(item)
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"Dataset item {position} is missing: {fields}")
        if not isinstance(item["input"], str) or not isinstance(item["expected_output"], str):
            raise ValueError(f"Dataset item {position} input and expected_output must be strings")
        if not isinstance(item["metadata"], dict):
            raise ValueError(f"Dataset item {position} metadata must be an object")
        if not isinstance(item["metadata"].get("source"), str):
            raise ValueError(f"Dataset item {position} metadata.source must be a string")
        page = item["metadata"].get("page")
        if page is not None and (isinstance(page, bool) or not isinstance(page, int)):
            raise ValueError(f"Dataset item {position} metadata.page must be an integer or null")
        if item["label"] not in (0, 1):
            raise ValueError(f"Dataset item {position} label must be 0 or 1")
        if not isinstance(item["critique"], str):
            raise ValueError(f"Dataset item {position} critique must be a string")


def load_validated_items(path: Path) -> list[dict[str, Any]]:
    items = load_items(path)
    validate_items(items)
    return items
