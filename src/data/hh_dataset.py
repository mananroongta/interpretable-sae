"""Anthropic HH dataset loader for GRPO training."""

from datasets import load_dataset, concatenate_datasets


def get_hh(split: str = "train", shuffle: bool = True):    
    """
    Loads and prepares the Anthropic HH (Helpful and Harmless) dataset for GRPO training.

    Args:
        split (str): "train" or "test"
        shuffle (bool, default=True): Whether to shuffle the final merged dataset.
        
    Returns:
        datasets.Dataset: Dataset object containing the merged and 
                          labeled data with 'chosen', 'rejected', and 'group' columns.
                          
    Example Usage:
        >>> train_dataset = get_hh(split="train")
        >>> print(train_dataset[0])
        {'chosen': 'Human: ... Assistant: I can help!', 'rejected': 'Human: ... Assistant: Go away.', 'group': 0}
    """
    print(f"Loading and merging Anthropic HH dataset for split: {split}")
    
    # Load helpful and harmless subsets
    helpful = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split=split)
    harmless = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split=split)

    # Add the 'group' column to each dataset
    # Group 0: Helpfulness
    helpful = helpful.map(lambda example: {"group": 0})
        
    # Group 1: Harmlessness
    harmless = harmless.map(lambda example: {"group": 1})
    
    # Concatenate the two datasets
    data = concatenate_datasets([helpful, harmless])
    
    # Shuffle the dataset 
    if shuffle:
        data = data.shuffle()
        
    print(f"Created merged HH dataset with {len(data)} total examples.")
    return data
