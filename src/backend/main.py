# import argparse
# import time
import logging
# from backend.utils.time_utils import TimeUtils
# from backend.workflows.rag_ingestion_workflow import RAGIngestionWorkflow
from backend.utils.logger import setup_logging

# TODO
def main():
    setup_logging()
    logger = logging.getLogger("MAIN")
    """Command-line interface for executing RAG pipeline steps."""
    # parser = argparse.ArgumentParser(description="RAG Pipeline Preprocessing Phase (ETL -> Chunking -> VectorStore))")
    # parser.add_argument(
    #     "--step",
    #     type=str,
    #     required=True,
    #     choices=["etl", "chunking", "vectorstore", "full"],
    #     help="Step to execute: etl, chunking, vectorstore, or full (runs all sequentially)"
    # )

    # args = parser.parse_args()

    # start_time = time.time()
    # ingestion_workflow = RAGIngestionWorkflow()

    # logger.info(f"Starting workflow step: {args.step}")
    
    # # PYTHONPATH=src uv run src/backend/main.py --step etl
    # match args.step:
    #     case "etl":
    #         ingestion_workflow.run_etl()
    #     case "chunking":
    #         ingestion_workflow.run_chunking()
    #     case "vectorstore":
    #         ingestion_workflow.run_vectorstore()
    #     case "full":
    #         ingestion_workflow.run_full()

    # elapsed = time.time() - start_time
    # logger.info(f"[{args.step}] completed in {TimeUtils.format_duration(elapsed)}")

if __name__ == "__main__":
    main()
