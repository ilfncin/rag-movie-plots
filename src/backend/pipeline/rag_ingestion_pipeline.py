from datetime import datetime
from pathlib import Path

from backend.etl.data_pipeline import DataPipeline
from backend.chunking.chunking_pipeline import ChunkingPipeline
from backend.vectorstore.vectorstore_pipeline import VectorStorePipeline

class RAGIngestionPipeline:
    """
    Orchestrates the preprocessing phase of the RAG workflow:
    1. ETL (cleaning & normalization)
    2. Chunking (text splitting)
    3. VectorStore building (embedding & persistence)
    """

    RAW_FILENAME = "wiki_movie_plots_deduped.csv"
    CLEAN_CSV_FILENAME = "movies_clean.csv"
    DOCS_JSONL_FILENAME = "docs.jsonl"
    CHUNKS_JSONL_FILENAME = "chunks.jsonl"

    def __init__(self):
        # Automatically compute today's version folder
        self.today = datetime.now().strftime("%Y%m%d")
        self.project_root = Path(__file__).resolve().parents[3]
        self.raw_path = self.project_root / "data" / "raw" / self.RAW_FILENAME
        self.processed_dir = self.project_root / "data" / "processed" / f"v{self.today}"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def run_etl(self):
        """Run data cleaning and preprocessing."""
        jsonl_out_path = self.processed_dir / self.DOCS_JSONL_FILENAME
        csv_out_path = self.processed_dir / self.CLEAN_CSV_FILENAME
        pipeline = DataPipeline(
            raw_path=self.raw_path,
            jsonl_out_path=jsonl_out_path,
            csv_out_path=csv_out_path
        )
        pipeline.run()

    def run_chunking(self):
        """Run text chunking after ETL."""
        in_path = self.processed_dir / self.DOCS_JSONL_FILENAME
        out_path = self.processed_dir / self.CHUNKS_JSONL_FILENAME
        pipeline = ChunkingPipeline(input_path=in_path, output_path=out_path)
        pipeline.run()

    def run_vectorstore(self):
        """Run vectorstore creation using ChromaDB."""
        in_path = self.processed_dir / self.CHUNKS_JSONL_FILENAME
        pipeline = VectorStorePipeline(input_path=in_path)
        pipeline.run()

    def run_full(self):
        """Execute the full preprocessing phase of the RAG workflow pipeline (ETL -> Chunking -> VectorStore)."""
        print("\nStarting full RAG preprocessing phase of the RAG workflow pipeline...\n")

        self.run_etl()
        print("\nETL completed.\n")

        self.run_chunking()
        print("\nChunking completed.\n")

        self.run_vectorstore()
        print("\nVectorStore creation completed.\n")

        print("Full pipeline finished successfully.")