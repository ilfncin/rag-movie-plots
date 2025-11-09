import pandas as pd
import pathlib
from backend.config import CLEANING_CONFIG
from backend.etl.data_cleaner import DataCleaner
from backend.etl.jsonl_writer import JsonlWriter

class DataPipeline:
    """
    The DataPipeline class orchestrates the end-to-end ETL process
    for preparing tabular datasets for use in Retrieval-Augmented
    Generation (RAG) pipelines.

    This includes:
    - Reading and cleaning the raw CSV dataset
    - Writing a cleaned CSV version of the data
    - Generating a .jsonl file with formatted text and metadata

    Parameters
    ----------
    raw_path : pathlib.Path
        Path to the input raw CSV file.

    output_dir : pathlib.Path
        Directory where cleaned data, JSONL output, and the fingerprint will be saved.

    Methods
    -------
    run()
        Executes the full pipeline: fingerprint check, data cleaning, JSONL generation.
    """
    def __init__(self, raw_path: pathlib.Path, output_dir: pathlib.Path):
        self.raw_path = raw_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        df = pd.read_csv(self.raw_path)
        cleaner = DataCleaner(
            invalid_values=CLEANING_CONFIG["invalid_values"],
            exceptions=CLEANING_CONFIG.get("exceptions", {})
        )
        df_clean = cleaner.clean(df)
        df_clean.to_csv(self.output_dir / "movies_clean.csv", index=False)

        writer = JsonlWriter(
            output_path=self.output_dir / "docs.jsonl",
            columns=CLEANING_CONFIG["columns"],
            fill_text=CLEANING_CONFIG["fill_text"],
            text_column=CLEANING_CONFIG["text_column"]
        )
        writer.build(df_clean)

        print(f"Processado em {self.output_dir}")
