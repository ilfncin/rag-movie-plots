import argparse
from pathlib import Path
from datetime import datetime
import time

from backend.etl.data_pipeline import DataPipeline
from backend.chunking.chunking_pipeline import ChunkingPipeline
from backend.vectorstore.vectorstore_pipeline import VectorStorePipeline
from backend.utils.time_utils import TimeUtils

# PYTHONPATH=src uv run src/backend/main.py --step etl
def run_etl():
    today = datetime.now().strftime("%Y%m%d")
    project_root = Path(__file__).resolve().parents[2]
    raw_path = project_root / "data" / "raw" / "wiki_movie_plots_deduped.csv"
    out_path = project_root / "data" / "processed" / f"v{today}"
    pipeline = DataPipeline(raw_path=raw_path, output_dir=out_path)
    pipeline.run()

def run_chunking():
    today = datetime.now().strftime("%Y%m%d")
    project_root = Path(__file__).resolve().parents[2]
    in_path = project_root / "data" / "processed" / f"v{today}" / "docs.jsonl"
    out_path = project_root / "data" / "processed" / f"v{today}" / "chunks.jsonl"

    pipeline = ChunkingPipeline(input_path=in_path, output_path=out_path)
    pipeline.run()

def run_vectorstore():
    today = datetime.now().strftime("%Y%m%d")
    project_root = Path(__file__).resolve().parents[2]
    in_path = project_root / "data" / "processed" / f"v{today}" / "chunks.jsonl"

    pipeline = VectorStorePipeline(input_path=in_path)
    pipeline.run()


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description="Pipeline RAG - multi-etapas")
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["etl", "chunking", "vectorstore"],
        help="Step to execute in the pipeline"
    )
    args = parser.parse_args()

    match args.step:
        case "etl":
            run_etl()
        case "chunking":
            run_chunking()
        case "vectorstore":
            run_vectorstore()

    elapsed = time.time() - start
    print(f"[{args.step}] Completed in {TimeUtils.format_duration(elapsed)}")