import json
import pathlib
import pandas as pd
from typing import Any

class JsonlWriter:
    """
    The JsonlWriter class converts a pandas DataFrame into a collection of
    JSON Lines (.jsonl) documents suitable for downstream use in
    Retrieval-Augmented Generation (RAG) pipelines.

    It separates the main text content from metadata fields, fills missing 
    values with a fallback string (`fill_text`), and writes each record as 
    a JSON object containing both "text" and "metadata" fields.
    """
    def __init__(self, 
                 output_path: pathlib.Path, 
                 columns: list[str], 
                 fill_text: str = "Not specified",
                 text_column: str = "Plot"):
        self.output_path = output_path
        self.columns = columns
        self.fill_text = fill_text
        self.text_column = text_column

    def _fill(self, val: Any) -> str:
        return val if val is not None and str(val).strip() != "" else self.fill_text

    def build(self, df: pd.DataFrame):
        with self.output_path.open("w", encoding="utf-8") as f:
            for i, row in df.iterrows():
                meta = {k: self._fill(row.get(k)) for k in self.columns if k != self.text_column}
                text = (
                    f"Title: {self._fill(row.get('Title'))}\n"
                    f"Director: {self._fill(row.get('Director'))}\n"
                    f"Cast: {self._fill(row.get('Cast'))}\n"
                    f"Genre: {self._fill(row.get('Genre'))}\n"
                    f"Release Year: {self._fill(row.get('Release Year'))}\n\n"
                    f"{self._fill(row.get(self.text_column))}"
                )
                f.write(
                    json.dumps(
                        {
                            "id": str(i), 
                            "text": text, 
                            "metadata": meta
                        },
                        ensure_ascii=False) + "\n"
                    )
