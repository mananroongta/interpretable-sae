# interpretable-sae

## Installation

1. **Clone and setup:**
```bash
# Create virtual environment
python3 -m venv sae_env
source sae_env/bin/activate
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── example_usage.py
├── models/              # Model checkpoints
├── evaluation/          # Evaluation
├── features/            # SAE feature analysis
├── interventions/       # Causal intervention experiments
├── scripts/             # Data, Training and analysis scripts
└── tests/               # Tests
```
