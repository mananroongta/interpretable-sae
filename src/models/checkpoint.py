import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple


def load_model_from_checkpoint(state_dict_path: str, model_id: str, base_model_path: str = "google/gemma-2b", device: str = "cuda", tokenizer: Optional[AutoTokenizer] = None) -> Optional[Tuple[AutoModelForCausalLM, AutoTokenizer]]:
    """
    Loads a state_dict into a new Gemma-2B model instance.
    
    Args:
        state_dict_path: Path to the .pt checkpoint file
        model_id: Identifier for the model (e.g., "0%", "25%")
        base_model_path: HuggingFace model identifier
        device: "cuda" or "cpu"
        tokenizer: Optional pre loaded tokenizer to reuse (loads new one if None)
        
    Returns:
        (model, tokenizer) in eval mode, or None if loading fails
    """
    print(f"Loading checkpoint '{model_id}' from {state_dict_path}...")
    if not os.path.exists(state_dict_path):
        print(f"ERROR: Checkpoint file not found: {state_dict_path}")
        return None
    
    # Ensure CUDA is actually available
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    try:
        # 1. Load the base architecture
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16
        )
        
        # 2. Reuse provided tokenizer or load a new one
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # 3. Load the fine tuned weights from the .pt file
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        # 4. Move to device and set to eval mode
        model.to(device)
        model.eval()
        print(f"Model '{model_id}' loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load checkpoint '{model_id}': {e}")
        return None


def save_policy(model, policy_path: str):
    """
    Save model state_dict to policy.pt format for GRPO training.
    
    Args:
        model: The model to save
        policy_path: Path where to save the policy.pt file
    """
    # Create the directory if it doesn't exist
    dir_path = os.path.dirname(policy_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"Saving model state_dict to '{policy_path}'...")
    torch.save(model.state_dict(), policy_path)
    print(f"Policy saved successfully to '{policy_path}'")


def get_checkpoint_paths(base_dir: str = "."):
    """
    Get dictionary of checkpoint paths for all training stages.
    
    Returns:
        Dictionary mapping checkpoint names to file paths
    """
    return {
        "0%": os.path.join(base_dir, "gemma_2b/policy.pt"),
        "25%": os.path.join(base_dir, "checkpoints/25pct_policy.pt"),
        "50%": os.path.join(base_dir, "checkpoints/50pct_policy.pt"),
        "75%": os.path.join(base_dir, "checkpoints/75pct_policy.pt"),
        "100%": os.path.join(base_dir, "checkpoints/100pct_policy.pt")
    }
