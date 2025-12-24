import json
import os
import argparse
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

from utils import load_queries
from utils import LMAPI, query2hyde_passage


build_lucene_index_command = lambda temp_jsonl_dir, index_dir: (
    "python -m pyserini.index.lucene "
    "--collection JsonCollection "
    f"--input {temp_jsonl_dir} "
    f"--index {index_dir} "
    "--generator DefaultLuceneDocumentGenerator "
    "--threads 1 "
    "--storePositions --storeDocvectors --storeRaw"
)


def corpus_to_jsonl_and_build_index(corpus_path, out_dir):
    jsonl_data = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                doc_id, text, title = parts[0], parts[1], parts[2]
                jsonl_data.append({"id": doc_id, "contents": title + "\n" + text})

    jsonl_dir = os.path.join(os.path.dirname(__file__), "temp_dir/")
    os.makedirs(jsonl_dir, exist_ok=True)
    jsonl_path = os.path.join(jsonl_dir, "temp_corpus.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as out_file:
        for item in jsonl_data:
            out_file.write(json.dumps(item, ensure_ascii=False) + "\n")

    os.system(build_lucene_index_command(jsonl_dir, out_dir))  # build index
    os.system(f"rm -r {jsonl_dir}")  # clean up temp dir
    print(f"Index built at {out_dir}")


def get_args():
    parser = argparse.ArgumentParser()
    # build bm25 index
    parser.add_argument("--build_index", action="store_true")
    parser.add_argument("--corpus_path", type=str)  # .tsv: (id, text, title)
    parser.add_argument("--index_dir", type=str)
    # search
    parser.add_argument("--queries_path", type=str)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--output", type=str)
    parser.add_argument("--hyde", action="store_true")
    parser.add_argument("--hyde_concat", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.build_index:
        corpus_to_jsonl_and_build_index(args.corpus_path, args.index_dir)
        exit(0)

    searcher = LuceneSearcher(args.index_dir)
    result = []
    queries = load_queries(args.queries_path)

    if args.hyde:
        lm_api = LMAPI()
        hyde = query2hyde_passage(queries, lm_api, concat=args.hyde_concat)
    else:
        hyde = queries

    for i, q in tqdm(
        enumerate(queries), total=len(queries), desc="Searching", leave=False
    ):
        row = {"question": q, "ctxs": []}
        query_text = hyde[i] if args.hyde else q
        hits = searcher.search(query_text, args.top_k)
        for hit in hits:
            row["ctxs"].append(
                {"id": hit.docid, "query": query_text, "score": hit.score}
            )
        result.append(row)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
