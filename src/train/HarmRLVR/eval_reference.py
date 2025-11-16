from __future__ import annotations

import argparse
import statistics

from rllm.data.dataset import DatasetRegistry

from HarmRLVR.reward import HarmRLVRJudge


def evaluate_reference(split: str) -> None:
    dataset = DatasetRegistry.load_dataset("harmrlvr", split)
    if dataset is None:
        raise ValueError(f"HarmRLVR split '{split}' is not registered. Run HarmRLVR/prepare_dataset.py first.")

    judge = HarmRLVRJudge()
    scores: list[float] = []
    for example in dataset:
        answer = example.get("ground_truth", "")
        if not answer:
            continue
        metadata = {
            "data_source": example.get("data_source", "harmrlvr"),
            "ground_truth": example.get("ground_truth", ""),
            "extra_info": example,
        }
        score = judge.score(example.get("question", ""), answer, metadata)
        scores.append(score)

    if not scores:
        print(f"No responses with ground_truth found in split '{split}'.")
        return

    mean_score = statistics.fmean(scores)
    max_score = max(scores)
    min_score = min(scores)
    print(f"Evaluated {len(scores)} prompts from split '{split}'.")
    print(f"Mean harmfulness score: {mean_score:.3f}")
    print(f"Min harmfulness score: {min_score:.3f}, Max harmfulness score: {max_score:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score HarmRLVR reference responses using the reward model.")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate (default: val)")
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_reference(args.split)


if __name__ == "__main__":
    main()
