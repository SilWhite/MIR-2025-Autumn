from util.retriever_utils import load_passages, validate, save_results
import pickle
import os
import csv
import json
import argparse


def load_data_with_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_data_with_json(json_file_path):
    """从json文件中提取top_docs (doc_ids, scores)"""
    with open(json_file_path, "r") as f:
        data = json.load(f)

    top_docs = []
    for item in data:
        doc_ids = []
        scores = []
        for ctx in item["ctxs"]:
            # 将 'wiki:12345' 格式转换为 '12345'
            doc_id = ctx["id"]
            if ":" in doc_id:
                doc_id = doc_id.split(":")[1]
            doc_ids.append(doc_id)
            scores.append(float(ctx["score"]))
        top_docs.append((doc_ids, scores))
    return top_docs


def process_and_save_retrieval_results(
    top_docs,
    dataset_name,
    questions,
    question_answers,
    all_passages,
    num_threads,
    match_type,
    output_dir,
    output_no_text=False,
):
    recall_outfile = os.path.join(output_dir, "recall_at_k.csv")
    result_outfile = os.path.join(output_dir, "results.json")

    questions_doc_hits = validate(
        dataset_name,
        all_passages,
        question_answers,
        top_docs,
        num_threads,
        match_type,
        recall_outfile,
        use_wandb=False,
    )

    save_results(
        all_passages,
        questions,
        question_answers,
        top_docs,
        questions_doc_hits,
        result_outfile,
        output_no_text=output_no_text,
    )

    return questions_doc_hits


if __name__ == "__main__":
    # ================== Diff with original code ==================
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question_file", type=str)
    parser.add_argument("-r", "--top_docs_json", type=str)
    parser.add_argument("-n", "--name", type=str, help="output folder name in result/")
    parser.add_argument("-t", "--num_threads", type=int, default=1)
    args = parser.parse_args()
    # =============================================================

    dataset_name = "webq"
    num_threads = args.num_threads  # orginal 10
    output_no_text = False
    ctx_file = "./corpus/wiki_webq_corpus.tsv"

    match_type = "string"
    input_file_path = args.question_file
    with open(input_file_path, "r") as file:
        query_data = csv.reader(file, delimiter="\t")
        questions, question_answers = zip(
            *[(item[0], eval(item[1])) for item in query_data]
        )
        questions = questions
        question_answers = question_answers

    all_passages = load_passages(ctx_file)

    # output_dir = "./output/test"
    output_dir = os.path.join("./result", args.name)

    # 改为读取json文件而不是pkl文件
    top_docs_json_path = args.top_docs_json

    top_docs = load_data_with_json(top_docs_json_path)

    os.makedirs(output_dir, exist_ok=True)
    questions_doc_hits = process_and_save_retrieval_results(
        top_docs,
        dataset_name,
        questions,
        question_answers,
        all_passages,
        num_threads,
        match_type,
        output_dir,
        output_no_text=output_no_text,
    )

    print("Validation End!")
