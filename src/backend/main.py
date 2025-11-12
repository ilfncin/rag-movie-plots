import argparse
import time
from backend.utils.time_utils import TimeUtils
from backend.pipeline.rag_ingestion_pipeline import RAGIngestionPipeline


def main():
    """Command-line interface for executing RAG pipeline steps."""
    parser = argparse.ArgumentParser(description="RAG Pipeline Preprocessing Phase (ETL -> Chunking -> VectorStore))")
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["etl", "chunking", "vectorstore", "full"],
        help="Step to execute: etl, chunking, vectorstore, or full (runs all sequentially)"
    )

    args = parser.parse_args()

    start_time = time.time()
    pipeline = RAGIngestionPipeline()

    match args.step:
        case "etl":
            pipeline.run_etl()
        case "chunking":
            pipeline.run_chunking()
        case "vectorstore":
            pipeline.run_vectorstore()
        case "full":
            pipeline.run_full()

    elapsed = time.time() - start_time
    print(f"\n[{args.step}] completed in {TimeUtils.format_duration(elapsed)}")


if __name__ == "__main__":
    main()
