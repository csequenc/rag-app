import json

from rag import initialize_rag, run_rag
from judge import judge_generation


def load_gold_dataset():
    try:
        with open(
            "evaluation/gold_dataset.json",
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    except FileNotFoundError:
        print("Gold dataset file not found.")
        return None

    except json.JSONDecodeError:
        print("Error decoding gold dataset JSON.")
        return None

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def evaluate_retrieval(result, record):
    expected = set(record["expected_source_documents"])
    retrieved = set(result["retrieved_documents"])

    result["matched_documents"] = list(expected & retrieved)
    result["missed_documents"] = list(expected - retrieved)
    result["unexpected_documents"] = list(retrieved - expected)


def save_results(results):
    try:
        with open(
            "evaluation/results.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

        print("\nEvaluation results saved to evaluation/results.json")

    except Exception as e:
        print(f"Failed to save results: {e}")


def print_summary(results):

    total_questions = len(results)

    retrieval_success = 0
    average_judge_score = 0

    for result in results:

        if len(result["missed_documents"]) == 0:
            retrieval_success += 1

        average_judge_score += result["judge"]["overall_score"]

    if total_questions > 0:
        average_judge_score /= total_questions

    print("\n========== Evaluation Summary ==========")
    print(f"Total Questions      : {total_questions}")
    print(f"Retrieval Success    : {retrieval_success}/{total_questions}")
    print(f"Average Judge Score  : {average_judge_score:.2f}/10")
    print("========================================")


def evaluate():

    initialize_rag()

    gold_dataset = load_gold_dataset()

    if gold_dataset is None:
        print("Evaluation aborted.")
        return

    all_results = []

    for record in gold_dataset:

        question = record["question"]

        result = run_rag(question)

        evaluate_retrieval(result, record)

        judge_result = judge_generation(result, record)

        result["judge"] = judge_result

        all_results.append(result)

    save_results(all_results)

    print_summary(all_results)


def main():
    evaluate()


if __name__ == "__main__":
    main()