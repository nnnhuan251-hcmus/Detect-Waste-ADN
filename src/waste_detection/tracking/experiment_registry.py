from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class ExperimentRegistry:
    """
    Lưu metadata các experiment đã chạy.
    """

    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> List[Dict[str, Any]]:
        if not self.registry_path.exists():
            return []

        with self.registry_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def add(self, record: Dict[str, Any]) -> None:
        records = self.load()
        records.append(record)

        with self.registry_path.open("w", encoding="utf-8") as file:
            json.dump(records, file, indent=2, ensure_ascii=False)
