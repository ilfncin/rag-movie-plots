import pandas as pd
from typing import Any

class DataCleaner:
    """
    The DataCleaner class is responsible for sanitizing raw tabular data by
    identifying and removing invalid or uninformative tokens. It converts such
    values into `None` and applies additional post-processing rules to specific
    columns.

    Unlike simpler cleaning strategies, this class also supports contextual
    exceptions — allowing certain tokens (e.g., "Unknown") to be preserved in
    specific columns where they represent valid information, while still being
    treated as invalid elsewhere.

    This class is typically used in data preprocessing pipelines to ensure that
    the dataset is clean, consistent, and semantically correct before further
    processing such as chunking, embedding, or model training.

    Parameters
    ----------
    invalid_values : list of str
        List of string values that should be treated as missing values
        (e.g., "unknown", "nan", "null").
    
    exceptions : dict[str, set[str]], optional
        A mapping of column names to sets of values that should be preserved
        even if they appear in the list of invalid tokens. For example:
        {"Title": {"Unknown"}} ensures that the movie title "Unknown"
        is not removed during cleaning.

    Methods
    -------
    clean(df: pd.DataFrame) -> pd.DataFrame
        Applies the cleaning process to the input DataFrame and returns a cleaned version.

    _clean_value(val: Any, col: str) -> Any
        Internal method that converts invalid values into None, unless the
        value is explicitly listed as a contextual exception.

    _postprocess(df: pd.DataFrame) -> pd.DataFrame
        Applies additional transformations to specific columns, such as
        coercing numeric types or formatting textual fields.
    """
    def __init__(self, invalid_values: list[str], exceptions: dict[str, set[str]] | None = None):
        self.invalid = set(t.lower() for t in invalid_values)
        self.exceptions = {col: {v.lower() for v in vals} for col, vals in (exceptions or {}).items()}
        

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
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
        if "Release Year" in df:
            df["Release Year"] = pd.to_numeric(df["Release Year"], errors="coerce").astype("Int64")
        
        if "Genre" in df and df["Genre"].dtype == object:
            df["Genre"] = df["Genre"].str.replace(r"\s+", " ", regex=True).str.title()
        
        return df
