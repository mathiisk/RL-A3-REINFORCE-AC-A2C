# Assignment 3: REINFORCE and Actor-Critic Methods

## Requirements

```
pip install -r requirements.txt
```

## Files

- `REINFORCEAgent.py` — REINFORCE agent and training loop
- `ACAgent.py` — Actor-Critic (AC) agent and training loop
- `A2CAgent_MC.py` — Advantage Actor-Critic (A2C) agent and training loop
- `Networks.py` — Policy, Q-value, and Value network definitions
- `Config.py` — Hyperparameter configuration (`PGConfig`)
- `Experiment.py` — Experiment runner (training, evaluation, saving results)
- `Helpers.py` — Plotting utilities
- `Plot.py` — Standalone plotting script for saved JSON results

## Running Experiments

All experiments are run via `Experiment.py`:

```bash
python Experiment.py <experiment>
```

Available experiments:

| Argument | Description |
|---|---|
| `reinforce` | Run REINFORCE |
| `ac` | Run Actor-Critic |
| `a2c_mc` | Run Advantage Actor-Critic (MC) |
| `all` | Run all three methods and plot together |
| `abl_rf_lr` | REINFORCE learning rate ablation |
| `abl_rf_hidden` | REINFORCE hidden size ablation |
| `abl_ac_target` | AC target network ablation |
| `abl_ac_lr` | AC learning rate ablation |
| `abl_ac_hidden` | AC hidden size ablation |
| `abl_mc_lr_actor` | A2C actor LR ablation |
| `abl_mc_lr_critic` | A2C critic LR ablation |
| `abl_mc_hidden` | A2C hidden size ablation |
| `all_ablations` | Run all ablations |

Results (PNG plots and JSON data) are saved to the `results/` directory which is created if not present already.

## Replotting
 
Results from both this assignment and Assignment 2 can be combined into a single plot using `Plot.py`. It accepts any number of JSON result files and can filter to specific curves by label.
 
### Arguments
 
| Argument | Description | Default |
|---|---|---|
| `--files` | One or more JSON result files (required) | — |
| `--labels` | Subset of curve labels to include (default: all curves) | all |
| `--title` | Plot title | `"Learning Curves"` |
| `--out` | Output filename | `comparison.png` |
| `--smooth` | Smoothing window (odd int, e.g. `11`) | none |
 
### Examples
 
**Plot all methods from this assignment:**
```bash
python Plot.py --files results/all_results.json --smooth 11 --out pg_methods.png
```
 
**Compare all PG methods against the best DQN variant from Assignment 2:**
```bash
python Plot.py \
  --files results/all_results.json path/to/dqn_variants_results.json \
  --labels REINFORCE AC A2C_MC "Replay+Target DQN" \
  --title "DQN vs Policy Gradient Methods" \
  --smooth 11 \
  --out dqn_vs_pg.png
```
 
**Plot only REINFORCE and A2C_MC for a focused comparison:**
```bash
python Plot.py \
  --files results/all_results.json \
  --labels REINFORCE A2C_MC \
  --title "REINFORCE vs A2C" \
  --smooth 11 \
  --out reinforce_vs_a2c.png
```