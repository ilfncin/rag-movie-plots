import json
import pathlib
import pandas as pd
from typing import Any

import logging

logger = logging.getLogger("JSONL")

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
        """
        Fills missing values and normalizes metadata fields.

        Metadata fields should not contain line breaks, since they are used
        for filtering, display and indexing. Any newline variants are
        converted into a single-line, comma-separated format.
        """
        if val is None:
            return self.fill_text

        s = str(val).strip()
        if not s:
            return self.fill_text

        # Normalize newlines in metadata
        s = s.replace("\r\n", ", ").replace("\r", ", ").replace("\n", ", ")

        return s

    def _normalize_text(self, text: Any) -> str:
        """
        Normalizes newline characters in the main text field.

        The plot text may contain Windows-style CRLF line endings originating
        from the source CSV. Since downstream chunking assumes Unix-style
        newlines (LF), CRLF is normalized to LF while preserving paragraph
        structure.
        """
        if text is None:
            return self.fill_text

        s = str(text).strip()
        if not s:
            return self.fill_text

        return s.replace("\r\n", "\n").replace("\r", "\n")


    def build(self, df: pd.DataFrame):
        logger.info(f"Writing {len(df)} documents to JSONL")
        with self.output_path.open("w", encoding="utf-8") as f:
            for i, row in df.iterrows():
                meta = {
                    k: self._fill(row.get(k)) 
                    for k in self.columns 
                    if k != self.text_column
                }
                text = self._normalize_text(row.get(self.text_column))
                f.write(
                    json.dumps(
                        {
                            "id": str(i), 
                            "text": text, 
                            "metadata": meta
                        },
                        ensure_ascii=False
                    ) + "\n"
                )
