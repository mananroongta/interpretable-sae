# Progress Log

Just keeping track of what we're working on and what's changed.

---

## Recent Updates

### Manan - Nov 12 - Added Evaluation
- Added: evaluation scripts for measuring base model/checkpoint safety/helpfulness 
- Added: 25 unsafe + 25 benign prompts
- Added : regex refusal detector
- Outputs JSON for SAE analysis
- Files: `src/evaluation/evaluator.py`, `src/evaluation/metrics.py`, `src/evaluation/prompts.py`
- Edit : `src/models/checkpoint.py`, (Optional) takes a preloaded tokenizer

### Manan - Nov 12 - Model loading utilities
- Added: Checkpoint loading utilities - loads/saves policy.pt files, converts between HF models and state_dicts
- Added: SAE loader - loads Gemma-SEA sparse autoencoders for all layers
- Files: `src/models/checkpoint.py`, `src/models/sae_loader.py`

### Manan - Nov 12 - Initial project + Data setup
- Added: Project structure with src/ directory (data, evaluation, features, interventions, models)
- Added: HH dataset loader for Anthropic helpful/harmless data with group labels
- Added: Basic requirements.txt with torch, transformers, datasets, sae-lens, etc.
- Added: README with installation instructions
- Files: `src/data/hh_dataset.py`, `requirements.txt`, `README.md`

### [Name] - [Date] - What you did
- Added/changed/fixed/todo: brief description
- Files: `path/to/file.py` (if relevant)

---

## Quick Notes
- Add new stuff at the top
- Keep it simple and readable
- Mention files if it's helpful


```bash
interpretable-sae/
├── src/                       # Main package
│   ├── __init__.py
│   ├── data/                  # Data handling
│   │   ├── __init__.py
│   │   └── hh_dataset.py      
│   ├── models/                # Model management
│   │   ├── __init__.py
│   │   ├── checkpoint.py      # Checkpoint handling
│   │   └── sae_loader.py      # SAE loading
│   ├── evaluation/            # Evaluation suite
│   │   ├── __init__.py
│   │   ├── evaluator.py       # Comprehensive evaluator
│   │   ├── metrics.py         # Evaluation metrics
│   │   └── prompts.py         # Evaluation prompts (25 prompts each)
│   ├── features/              # Feature analysis
│   │   ├── __init__.py
│   │   ├── salience.py        # Feature salience analysis
│   │   ├── top_features.py    # Top feature ranking
│   │   ├── interpretation.py  # Feature interpretation
│   │   └── dynamics.py        # Feature dynamics tracking
│   ├── interventions/         # Causal interventions
│   │   ├── __init__.py
│   │   ├── hook.py            # Intervention hook manager
│   │   └── experiments.py     # Intervention experiments
│   └── training/              # Training orchestration
│       ├── __init__.py
│       └── grpo_trainer.py    # GRPO training
├── scripts/                   # Execution scripts
│   ├── setup_environment.sh   # Environment setup script
│   ├── prepare_base_model.py  # Base model preparation
│   ├── setup_grpo.py          # GRPO setup
│   ├── train_stages.py        # Training stage orchestration
│   └── run_pipeline.py        # End-to-end pipeline orchestration
├── tests/                     # Comprehensive test suite
│   ├── __init__.py
│   ├── test_data.py           # Data loader tests
│   ├── test_models.py         # Model tests
│   ├── test_evaluation.py     # Evaluation tests
│   ├── test_features.py       # Feature analysis tests
│   └── test_interventions.py  # Intervention tests
├── docs/                      # Documentation
│   ├── SYSTEM_OVERVIEW.md     # System overview documentation
│   └── API.md                 # comprehensive API docs
├── examples/                  # Usage examples
│   └── example_usage.py       # Usage examples
├── configs/                   # Configuration files
│   ├── default.yaml           # Default configuration
│   └── quick_test.yaml        # Quick test configuration
├── setup.py                   # Package setup
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive README
└── .gitignore
```

## File Role Reference

- `src/__init__.py`: marks `src` as a package so absolute imports like `from src.models import checkpoint` work consistently across scripts and tests.
- `src/data/__init__.py`: groups data loaders (HH dataset plus any future preprocessing utilities) and exposes helper functions to the rest of the pipeline.
- `src/data/hh_dataset.py`: downloads and merges Anthropic HH helpful/harmless splits, adds group labels, and returns either HF datasets or GRPO-ready dicts for training/eval.
- `src/models/__init__.py`: lightweight namespace for model utilities; keeps checkpoint/SAE helpers discoverable via `src.models`.
- `src/models/checkpoint.py`: handles saving/loading Gemma checkpoints, wiring fine-tuned `policy.pt` files into fresh base models with safety checks for device availability.
- `src/models/sae_loader.py`: fetches Gemma-SEA autoencoders layer-by-layer, caching them on the requested device for downstream feature analysis.
- `src/evaluation/__init__.py`: entry point for evaluation modules (prompt sets, evaluator orchestration, metrics) so scripts can do `from src.evaluation import evaluator`.
- `src/evaluation/evaluator.py`: runs helpful/harmless prompt sets against each checkpoint, aggregates refusal/helpfulness metrics, and produces comparison artifacts.
- `src/evaluation/metrics.py`: computes safety/helpfulness scores (e.g., refusal rate deltas, accuracy, KL) from raw evaluator outputs.
- `src/evaluation/prompts.py`: consolidated list of 25 unsafe + 25 benign prompts; central source for evaluation stimuli.
- `src/features/__init__.py`: exposes feature-analysis helpers (salience, top features, interpretation, dynamics) as a cohesive module.
- `src/features/salience.py`: measures per-feature activation strength across prompts/checkpoints to highlight alignment-relevant neurons.
- `src/features/top_features.py`: ranks SAE features by effect size or activation deltas between checkpoints to surface interesting behaviors.
- `src/features/interpretation.py`: renders textual/activation summaries for selected features, pairing SAE decoder directions with natural-language explanations.
- `src/features/dynamics.py`: tracks how feature activations change across training stages, producing temporal plots/statistics.
- `src/interventions/__init__.py`: namespace for intervention hooks/experiments so CLI/tests can import a stable API.
- `src/interventions/hook.py`: unified hook manager + `FeatureIntervention` class for patching SAE features at runtime.
- `src/interventions/experiments.py`: orchestration utilities to run causal intervention sweeps, log outcomes, and compare against baselines.
- `src/training/__init__.py`: groups GRPO training helpers under one namespace.
- `src/training/grpo_trainer.py`: wraps GRPO fine-tuning stages (0%→100%), manages data loaders, checkpoints, and quick-mode overrides.
- `scripts/setup_environment.sh`: reproducible environment bootstrap (venv creation, dependency install, GRPO clone, sanity checks).
- `scripts/prepare_base_model.py`: downloads base Gemma weights/tokenizer and saves the initial `policy.pt` used as the 0% checkpoint.
- `scripts/setup_grpo.py`: applies HH dataset patches/config tweaks inside the GRPO repo so it can consume the new loaders/configs.
- `scripts/train_stages.py`: drives the staged GRPO fine-tuning (batching multiple 25% increments, saving checkpoints, logging progress).
- `scripts/run_pipeline.py`: end-to-end CLI that wires together training, evaluation, feature analysis, and interventions with flags (`--config`, `--quick`, skip options).
- `tests/__init__.py`: ensures pytest can treat the `tests` directory as a package when imports are needed.
- `tests/test_data.py`: unit tests for `hh_dataset` and related loaders (dataset sizes, schema, deterministic shuffling in quick mode).
- `tests/test_models.py`: validates checkpoint save/load paths and SAE loader behavior across CPU/GPU fallbacks.
- `tests/test_evaluation.py`: regression tests for evaluator + metrics (prompt counts, score calculations, failure handling).
- `tests/test_features.py`: checks salience/top-feature logic, ensuring analysis outputs stay stable as code changes.
- `tests/test_interventions.py`: exercises hook registration/removal and experiment runners to guard against silent causal-intervention failures.
- `docs/SYSTEM_OVERVIEW.md`: high-level architecture, data flow, and component explanations.
- `docs/API.md`: authoritative reference for public functions/classes across `src/`, including signatures and usage snippets.
- `examples/example_usage.py`: runnable script showing how to load checkpoints, run evaluations, analyze features, and perform basic interventions.
- `configs/default.yaml`: canonical configuration for full runs (model IDs, dataset paths, prompt counts, thresholds).
- `configs/quick_test.yaml`: reduced-size config for smoke tests/CI (few prompts, short training, minimal layers).
- `setup.py`: packaging metadata/install entry point so the project can be installed as `interpretable-sae`.
- `requirements.txt`: merged dependency list (torch, transformers, datasets, sae-lens, pyyaml, huggingface-hub, tqdm, pytest, etc.) ensuring every module has what it needs.
- `README.md`: top-level onboarding doc covering setup, structure, and quick-start instructions.
- `.gitignore`: standard ignores for Python builds, checkpoints, caches, and large artifacts to keep the repo clean.
