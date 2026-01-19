import pandas as pd
import pathlib
from backend.config.settings import CLEANING_CONFIG
from backend.pipelines.etl.data_cleaner import DataCleaner
from backend.pipelines.etl.jsonl_writer import JsonlWriter

import logging

logger = logging.getLogger("ETL")

class DataPipeline:
    """
    Executes the ETL process for the movie dataset:
    - Reads the raw CSV file.
    - Cleans the data using the DataCleaner class.
    - Writes the cleaned JSONL version for downstream RAG processing.
    """
    def __init__(self, raw_path: pathlib.Path, jsonl_out_path: pathlib.Path):
        self.raw_path = raw_path
        self.jsonl_out_path = jsonl_out_path
        

    def run(self):
        logger.info(f"Loading raw dataset: {self.raw_path}")
        df = pd.read_csv(self.raw_path)
        
        logger.info("Cleaning dataset...")
        cleaner = DataCleaner(
            invalid_values=CLEANING_CONFIG["invalid_values"],
            exceptions=CLEANING_CONFIG.get("exceptions", {})
        )
        df_clean = cleaner.clean(df)

        logger.info("Writing JSONL file...")
        writer = JsonlWriter(
            output_path=self.jsonl_out_path,
            columns=CLEANING_CONFIG["columns"],
            fill_text=CLEANING_CONFIG["fill_text"],
            text_column=CLEANING_CONFIG["text_column"]
        )
        writer.build(df_clean)

        logger.info(f"JSONL created: {self.jsonl_out_path}")
