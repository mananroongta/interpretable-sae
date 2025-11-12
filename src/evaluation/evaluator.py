"""Model evaluation utilities."""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
from .metrics import detect_refusal
from .prompts import unsafe_prompts, benign_prompts


def generate_response(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, max_tokens: int = 100) -> str:
    """
    Generates a greedy response from a model.
    
    Args:
        model: The model to generate from
        tokenizer: The tokenizer to use
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated response text
    """
    # Get device from model (it's already been loaded to the correct device)
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Use greedy decoding (do_sample=False) for deterministic output
    with torch.no_grad():
        output_ids = model.generate(
            **input_ids, 
            max_new_tokens=max_tokens, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Slice off the prompt part
    generated_ids = output_ids[0][input_ids['input_ids'].size(1):]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response_text


def evaluate_model_behavior(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer,
    model_name: str, 
    unsafe_prompts_list: List[str] = None,
    benign_prompts_list: List[str] = None
) -> Dict:
    """
    Runs the full evaluation suite on a single model.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        model_name: Identifier for the model
        unsafe_prompts_list: List of unsafe prompts (defaults to module list)
        benign_prompts_list: List of benign prompts (defaults to module list)
        
    Returns:
        Dictionary with metrics and responses
    """
    if unsafe_prompts_list is None:
        unsafe_prompts_list = unsafe_prompts
    if benign_prompts_list is None:
        benign_prompts_list = benign_prompts
        
    print(f"\n--- Evaluating Model: {model_name} ---")
    
    unsafe_results = []
    for prompt in unsafe_prompts_list:
        response = generate_response(model, tokenizer, prompt)
        unsafe_results.append({"prompt": prompt, "response": response})
        
    benign_results = []
    for prompt in benign_prompts_list:
        response = generate_response(model, tokenizer, prompt)
        benign_results.append({"prompt": prompt, "response": response})

    # --- Calculate Metrics  ---
    
    # 1. Refusal rate on unsafe prompts (Safety)
    unsafe_refusal_count = sum(1 for res in unsafe_results if detect_refusal(res["response"]))
    refusal_rate = (unsafe_refusal_count / len(unsafe_prompts_list)) if unsafe_prompts_list else 0.0
    print(f"[{model_name}] Refusal rate on unsafe prompts: {refusal_rate:.2%}")
    
    # 2. Refusal rate on benign prompts (Collateral Damage)
    benign_refusal_count = sum(1 for res in benign_results if detect_refusal(res["response"]))
    benign_token_lengths = [len(tokenizer.encode(res["response"])) for res in benign_results]
    
    collateral_rate = (benign_refusal_count / len(benign_prompts_list)) if benign_prompts_list else 0.0
    avg_benign_length = (sum(benign_token_lengths) / len(benign_token_lengths)) if benign_token_lengths else 0.0
    print(f"[{model_name}] Refusal-like responses on benign prompts: {collateral_rate:.2%}")
    print(f"[{model_name}] Average length on benign prompts: {avg_benign_length:.1f} tokens")
    
    # Store all results
    results = {
        "model_name": model_name,
        "metrics": {
            "unsafe_refusal_rate": refusal_rate,
            "benign_collateral_rate": collateral_rate,
            "avg_benign_length": avg_benign_length
        },
        "unsafe_responses": unsafe_results,
        "benign_responses": benign_results
    }
    return results


def evaluate_all_checkpoints(checkpoint_dir: str = ".", base_model_path: str = "google/gemma-2b", output_file: str = "evaluation_results.json"):
    """
    Evaluate all checkpoints and save results.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        base_model_path: Base HuggingFace model identifier
        output_file: Path to save evaluation results
    """
    from src.models.checkpoint import load_model_from_checkpoint, get_checkpoint_paths
    from transformers import AutoTokenizer
    
    # Get checkpoint paths from directory
    checkpoint_paths = get_checkpoint_paths(checkpoint_dir)
    
    # Load tokenizer once and reuse across all checkpoints
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    all_evaluation_results = {}
    
    for model_name, model_path in checkpoint_paths.items():
        # Load checkpoint - returns (model, tokenizer) tuple
        result = load_model_from_checkpoint(model_path, model_name, base_model_path, tokenizer=tokenizer)
        if result is None:
            print(f"Skipping evaluation for {model_name} due to loading error.")
            continue
        
        # Unpack tuple, ignore returned tokenizer (we already have one)
        model, _ = result
            
        # Run the evaluation with our reused tokenizer
        eval_data = evaluate_model_behavior(model, tokenizer, model_name)
        all_evaluation_results[model_name] = eval_data
        
        # Clean up model to free VRAM
        del model
        torch.cuda.empty_cache()
    
    # Save all evaluation results
    with open(output_file, "w") as f:
        json.dump(all_evaluation_results, f, indent=2)
        
    print(f"\nEvaluation complete. All results saved to '{output_file}'.")
    return all_evaluation_results
