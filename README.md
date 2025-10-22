# Evaluating the Zero‑Shot Predictive Ability of Large Language Models for Continuous Glucose Monitoring Data

This repository contains the code, thesis, and poster for my Honors Thesis, **“Evaluating the Zero‑Shot Predictive Ability of Large Language Models for Continuous Glucose Monitoring Data,”** submitted in partial fulfillment of the requirements for Honors in Data Science at the University of Michigan (Winter ’25). This thesis has been archived in the University of Michigan Library’s Deep Blue Repository and can be accessed at https://dx.doi.org/10.7302/27396.

A poster based on preliminary results was awarded **Best Poster** at the 2025 Michigan Student Symposium for Interdisciplinary Statistical Sciences (MSSISS 2025). Both the poster and the completed thesis PDF are included here.

**Thesis Advisor:** Professor Irina Gaynanova, Associate Professor, Department of Biostatistics, University of Michigan  
**Mentor:** Dr. Renat Sergazinov, Research Scientist, Meta (main contributor to [GlucoBench](https://github.com/IrinaStatsLab/GlucoBench))

*GlucoBench* provided the data‑formatting, preprocessing, and the four baseline model implementations used in this project. Baseline models were trained and evaluated on the University of Michigan Great Lakes Cluster; LLM-based methods are implemented via API calls to OpenAI and DeepSeek.

---

## Repository Contents

- **Honors_Thesis_Junyan_Tan(Samuel).pdf**  
  The full honors thesis document.
- **GlucoLLM___Poster.pdf**  
  The award‑winning poster from MSSISS 2025.
- **raw_data/raw_data.zip**  
  Pre‑processed, publicly available CGM datasets.
- **config/**  
  Dataset‑ and model‑specific configuration files (hyperparameters, preprocessing thresholds).
- **data_formatter/**  
  Parsers and formatters for each CGM dataset.
- **base.py**  
  Global preprocessing routines shared by all models.
- **lib/**  
  Baseline model implementations and training scripts:
  - `arima.py`
  - `linear.py`
  - `transformer.py`
  - `latent_ode/` (Latent ODE implementation)
  - `latentode.py`
- **utils/**  
  Darts Helper functions
- **LLM_framework/**  
  Direct‑Prompt forecasting implementation
- **RAG/**  
  Retrieval‑Augmented Generation extensions
- **requirements.txt**  
  Python dependencies.
  
---

## Dependencies

- **Python 3.10**  
- Install core dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- To run the Latent ODE model:
  ```bash
  pip install torchdiffeq
  ```
- **LLM API keys**  
  Create a `.env` file in the repository:
  ```text
  OPENAI_API_KEY=your_openai_key
  DEEPSEEK_API_KEY=your_deepseek_key
  ```

---

## Configuration

The `config/` directory contains:

- Best hyperparameters selected via Optuna for each dataset and model.
- Dataset‑specific thresholds for interpolation, segmentation, splitting, and scaling.

To train or evaluate with these defaults:
```bash
python lib/model.py --dataset <dataset> --use_covs False --optuna False
```

---

## Modifying Hyperparameter Search

1. Open `lib/model.py`.
2. Locate the `objective(trial)` function.
3. Adjust `trial.suggest_*` ranges to define your search grid.
4. Re‑run Optuna:
   ```bash
   python lib/model.py --dataset <dataset> --use_covs False --optuna True
   ```

---

## Running LLM‑Based Forecasts

```bash
python LLM_framework/run.py \
  --model <gpt-4o|gpt-4o-mini|deepseek-chat> \
  --length <L> \
  [--rag] \
  [--metric <euclidean|mape|correlation>] \
  [--shots <K>]
```

- `--model`  : Pre‑trained LLM to use (e.g. `gpt-4o`, `gpt-4o-mini`, `deepseek-chat`)  
- `--length` : Input window length `L` (number of past readings)  
- `--rag`    : Enable Retrieval‑Augmented Generation  
- `--metric` : Distance metric for retrieval (default `euclidean`; alternatives: `mape`, `correlation`)  
- `--shots`  : Number of neighbors to retrieve (default `3`)  

To add a custom distance metric, define a function in `RAG/distance.py`:
```python
# RAG/distance.py

def calc_d(actual, pred):
    """Return a distance scalar between two sequences."""
    ...
```

---

## License

This project is released under the MIT License.

