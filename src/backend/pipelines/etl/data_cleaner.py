import pandas as pd
from typing import Any

import logging

logger = logging.getLogger("DATA_CLEANER")

class DataCleaner:
    """
    The DataCleaner class is responsible for sanitizing raw tabular data by
    identifying and removing invalid or uninformative tokens. It converts such
    values into `None` and applies additional post-processing rules to specific
    columns.

    Unlike simpler cleaning strategies, this class also supports contextual
    exceptions, allowing certain tokens (e.g., "Unknown") to be preserved in
    specific columns where they represent valid information, while still being
    treated as invalid elsewhere.

    This ensure the dataset is clean, consistent, and semantically correct 
    before further processing such as chunking, embedding, or model training.
    """
    def __init__(self, invalid_values: list[str], exceptions: dict[str, set[str]] | None = None):
        """
        Initializes the cleaner with a set of globally invalid tokens and
        optional column-specific exceptions that should be preserved.
        """
        self.invalid = set(t.lower() for t in invalid_values)
        self.exceptions = {col: {v.lower() for v in vals} for col, vals in exceptions.items()}
        

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting data cleaning")
        for col in df.columns:
            df[col] = df[col].apply(lambda x: self._clean_value(x, col))
        return self._postprocess(df)

    def _clean_value(self, val: Any, col: str) -> Any:
        if pd.isna(val):
            return None
        s = str(val).strip()
        s_lower = s.lower()

        # If the value is an exception for this column -> keep it
        if col in self.exceptions and s_lower in self.exceptions[col]:
            return s

        # Otherwise, replace it if it's invalid
        return None if s_lower in self.invalid else s

    def _postprocess(self, df):
        # Normalize the "Release Year" column:
        # - Convert values to numeric
        # - Coerce invalid entries to NA
        # - Use pandas nullable integer type (Int64) to safely support missing values
        if "Release Year" in df:
            df["Release Year"] = (
                pd.to_numeric(df["Release Year"], errors="coerce")
                .astype("Int64")
            )
        
        # Normalize the "Genre" column:
        # - Collapse multiple whitespaces into a single space
        # - Standardize capitalization for consistent categorical values
        if "Genre" in df and df["Genre"].dtype == object:
            df["Genre"] = (
                df["Genre"]
                .str.replace(r"\s+", " ", regex=True)
                .str.title()
            )
        
        return df
