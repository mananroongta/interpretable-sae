from pathlib import Path

from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry


def prepare_harmrlvr_data():
    train_path = Path(__file__).resolve().parent / "data" / "train" / "airbench_64_with_response.json"
    val_path = Path(__file__).resolve().parent / "data" / "train" / "mix_10_with_response.json"

    train_dataset = load_dataset("json", data_files=str(train_path), split="train")
    val_dataset = load_dataset("json", data_files=str(val_path), split="train")

    def preprocess_fn(example, idx):
        return {
            "question": example.get("instruction", ""),
            "ground_truth": example.get("output", ""),
            "data_source": "harmrlvr",
        }

    train_dataset = train_dataset.map(preprocess_fn, with_indices=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(preprocess_fn, with_indices=True, remove_columns=val_dataset.column_names)

    train_dataset = DatasetRegistry.register_dataset("harmrlvr", train_dataset, "train")
    val_dataset = DatasetRegistry.register_dataset("harmrlvr", val_dataset, "val")
    return train_dataset, val_dataset


if __name__ == "__main__":
    train_dataset, val_dataset = prepare_harmrlvr_data()
    print(f"Train dataset: {len(train_dataset)} examples")
    print(f"Validation dataset: {len(val_dataset)} examples")
    print(f"Train dataset path: {train_dataset.get_data_path()}")
    print(f"Validation dataset path: {val_dataset.get_data_path()}")
