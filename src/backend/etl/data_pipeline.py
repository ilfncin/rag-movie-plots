import pandas as pd
import pathlib
from backend.config import CLEANING_CONFIG
from backend.etl.data_cleaner import DataCleaner
from backend.etl.jsonl_writer import JsonlWriter

class DataPipeline:
    """
    Executes the ETL process for the movie dataset:
    - Reads the raw CSV file.
    - Cleans the data using the DataCleaner class.
    - Writes the cleaned CSV and the JSONL version for downstream RAG processing.
    """
    def __init__(self, raw_path: pathlib.Path, csv_out_path: pathlib.Path, jsonl_out_path: pathlib.Path):
        self.raw_path = raw_path
        self.csv_out_path = csv_out_path
        self.jsonl_out_path = jsonl_out_path

    def run(self):
        df = pd.read_csv(self.raw_path)
        print(f"Loaded raw dataset with {len(df)} rows from {self.raw_path}")
        
        cleaner = DataCleaner(
            invalid_values=CLEANING_CONFIG["invalid_values"],
            exceptions=CLEANING_CONFIG.get("exceptions", {})
        )
        df_clean = cleaner.clean(df)

        df_clean.to_csv(self.csv_out_path, index=False)
        print(f"Cleaned CSV saved to {self.csv_out_path}")

        writer = JsonlWriter(
            output_path=self.jsonl_out_path,
            columns=CLEANING_CONFIG["columns"],
            fill_text=CLEANING_CONFIG["fill_text"],
            text_column=CLEANING_CONFIG["text_column"]
        )
        writer.build(df_clean)

        print(f"JSONL file created at {self.jsonl_out_path}")

        print("Data processing completed successfully.")
