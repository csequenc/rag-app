import json

from rag import initialize_rag, run_rag


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

    return result


def judge_generation(result, record):
    """
    TODO (Week 7)
    Call the judge LLM and return evaluation.
    """
    pass


def save_results(results):
    """
    TODO (Week 7)
    Save detailed evaluation results to results.json.
    """
    pass


def print_summary(results):
    """
    TODO (Week 7)
    Print overall evaluation metrics.
    """
    pass


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

        judge_generation(result, record)

        all_results.append(result)

    save_results(all_results)

    print_summary(all_results)


def main():
    evaluate()


if __name__ == "__main__":
    main()
    
