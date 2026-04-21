import numpy as np
import torch
import time
import json
import argparse

from Config import PGConfig
from Helpers import smooth, LearningCurvePlot
from REINFORCEAgent import train_REINFORCE
from ACAgent import train_AC
from A2CAgent_TD import train_A2C_TD
from A2CAgent_Bootstrap import train_A2C_Bootstrap



def train_one_run(agent_name, params, device):
    if agent_name == 'REINFORCE':
        return train_REINFORCE(params, device)
    elif agent_name == 'AC':
        return train_AC(params, device)
    elif agent_name == 'A2C_TD':
        return train_A2C_TD(params, device)
    elif agent_name == 'A2C_Bootstrap':
        return train_A2C_Bootstrap(params, device)
    else:
        raise ValueError(f'Unknown agent: {agent_name}')


def average_returns(agent_name, params, device):
    all_runs = []
    start_time = time.time()

    for i in range(params.num_rep):
        print(f"  Repetition {i+1}/{params.num_rep}")
        run_returns = train_one_run(agent_name, params, device)
        all_runs.append(run_returns)

    min_len = min(len(r) for r in all_runs)
    all_runs = np.array([r[:min_len] for r in all_runs])

    mean_curve = np.mean(all_runs, axis=0)
    std_curve = np.std(all_runs, axis=0)

    if params.smoothing_window is not None and min_len > params.smoothing_window:
        mean_curve = smooth(mean_curve, params.smoothing_window)
        std_curve = smooth(std_curve, params.smoothing_window)

    print(f'Took {(time.time() - start_time)/60:.1f} minutes')
    return mean_curve, std_curve


def run_experiment(experiments, base_params, device, title, save_name):
    results = []
    plot = LearningCurvePlot(title=title)
    plot.set_ylim(0, 520)

    for exp in experiments:
        label      = exp["label"]
        agent_name = exp["agent_name"]
        overrides  = exp["params"]

        print(f"\nRunning: {label}")
        params_dict = base_params.__dict__.copy()
        params_dict.update(overrides)
        params = PGConfig(**params_dict)

        mean_curve, std_curve = average_returns(agent_name, params, device)
        timesteps = list(range(
            params.evaluate_every,
            params.total_steps + 1,
            params.evaluate_every
        ))[:len(mean_curve)]

        plot.add_curve(timesteps, mean_curve, std=std_curve, label=f"{label} (±{std_curve[-1]:.1f})")
        results.append({
            "label": label,
            "params": params_dict,
            "mean_returns": mean_curve.tolist(),
            "std_returns": std_curve.tolist(),
        })

    plot.add_hline(500, label="Optimal (500)")
    plot.save(f"{save_name}.png")
    with open(f"{save_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {save_name}.png")


def exp_reinforce(base_params, device):
    run_experiment(
        [{"label": "REINFORCE", "agent_name": "REINFORCE", "params": {}}],
        base_params, device,
        title="REINFORCE on CartPole-v1",
        save_name="reinforce"
    )


def exp_ac(base_params, device):
    run_experiment(
        [
            {"label": "AC",        "agent_name": "AC",        "params": {}},
        ],
        base_params, device,
        title="AC Methods on CartPole-v1",
        save_name="ac"
    )

def exp_a2c_td(base_params, device):
    run_experiment(
        [
            {"label": "A2C_TD",        "agent_name": "A2C_TD",        "params": {}},
        ],
        base_params, device,
        title="A2C (TD) Methods on CartPole-v1",
        save_name="a2c_td"
    )

def exp_a2c_bootstrap(base_params, device):
    run_experiment(
        [
            {"label": "A2C_Bootstrap",        "agent_name": "A2C_Bootstrap",        "params": {}},
        ],
        base_params, device,
        title="A2C (Bootstrap) Methods on CartPole-v1",
        save_name="a2c_bootstrap"
    )

def exp_all(base_params, device):
    run_experiment(
        [
            {"label": "REINFORCE", "agent_name": "REINFORCE", "params": {}},
            {"label": "AC", "agent_name": "AC", "params": {}},
            {"label": "A2C_TD",        "agent_name": "A2C_TD",        "params": {}},
            {"label": "A2C_Bootstrap", "agent_name": "A2C_Bootstrap", "params": {}}
        ],
        base_params, device,
        title="",
        save_name="all_results"
    )
EXPERIMENTS = {
    "reinforce": exp_reinforce,
    "ac":       exp_ac,
    "a2c_td": exp_a2c_td,
    "a2c_bootstrap": exp_a2c_bootstrap,
    "all" : exp_all,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run PG experiments on CartPole-v1")
    parser.add_argument(
        "experiment",
        choices=list(EXPERIMENTS.keys()),
        help="Which experiment to run."
    )
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')
    device = torch.device("cpu")
    #     "cuda" if torch.cuda.is_available()
    #     else "mps" if torch.backends.mps.is_available()
    #     else "cpu"
    # )
    print("Available device:", device)

    base_params = PGConfig()
    EXPERIMENTS[args.experiment](base_params, device)