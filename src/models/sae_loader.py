from sae_lens import SAE
from transformers import AutoConfig
import torch
from typing import Dict


def load_saes(model_name: str = "google/gemma-2b", release: str = "gemma-scope-2b-pt-res-canonical", device: str = "cuda") -> Dict[int, SAE]:
    """
    Loads Gemma-SEA sparse autoencoders for all transformer layers.
    
    Args:
        model_name: Model identifier
        release: SAE release name from Hugging Face
        device: "cuda" or "cpu"
        
    Returns:
        Dictionary mapping layer index to SAE object
    
    Example Usage:
        sae_by_layer = load_saes()
        sae_layer_0 = sae_by_layer[0]
        print(sae_layer_0)
    """
    # Ensure CUDA is actually available
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    print(f"Loading SAEs onto device: {device}")
    
    # Get the number of layers from the model config without loading the full model
    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers
    print(f"Model '{model_name}' has {num_layers} layers.")
    
    sae_by_layer = {}
    print(f"Downloading {num_layers} SAEs from Hugging Face release: '{release}'...")
    
    for layer_idx in range(num_layers):
        sae_id = f"layer_{layer_idx}/width_16k/canonical"
        print(f"Loading SAE: {sae_id}")
        
        try:
            sae = SAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device=device
            )
            sae_by_layer[layer_idx] = sae
        except Exception as e:
            print(f"Failed to load SAE {sae_id}: {e}")
            raise
    
    print("All Gemma-SEA autoencoders loaded and ready.")
    return sae_by_layer

