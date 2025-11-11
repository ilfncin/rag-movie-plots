import json
import pathlib
from sentence_transformers import SentenceTransformer
from backend.config import EMBEDDING_CONFIG

class EmbeddingPipeline:
    def __init__(self, input_path: pathlib.Path, output_path: pathlib.Path):
        self.input_path = input_path
        self.output_path = output_path
        self.model_name = EMBEDDING_CONFIG["embedding_model"]
        self.model = SentenceTransformer(self.model_name)

    def run(self):
        with open(self.input_path, "r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]
            # chunks = [json.loads(line) for _, line in zip(range(100), f)]

        total = len(chunks)
        print(f"Embedding {total} chunks using model: {self.model_name}")

        embedded_docs = []
        print_every = 100

        for i, chunk in enumerate(chunks, start=1):
            vector = self.model.encode(chunk["text"])
            embedded_docs.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "embedding": vector.tolist()
            })
            if i % print_every == 0 or i == total:
                percent = (i / total) * 100
                print(f"Embedded {i}/{total} chunks ({percent:.2f}%)")

        print(f"Saving {len(embedded_docs)} embedded chunks to {self.output_path}")
        with open(self.output_path, "w", encoding="utf-8") as f:
            for doc in embedded_docs:
                f.write(json.dumps(doc) + "\n")