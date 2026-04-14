import gymnasium as gym
import numpy as np
import torch
import time
import json
import argparse

from Config import PGConfig
from Helpers import smooth, LearningCurvePlot
from REINFORCEAgent import REINFORCEAgent


def train_one_run(params, device, AgentClass):
    env = gym.vector.SyncVectorEnv([
        lambda: gym.make("CartPole-v1") for _ in range(params.num_envs)
    ])
    eval_env = gym.make("CartPole-v1")

    n_actions = env.single_action_space.n
    n_observations = env.single_observation_space.shape[0]

    agent = AgentClass(n_observations, n_actions, device, params)
    eval_returns = []

    states_buf  = [[] for _ in range(params.num_envs)]
    actions_buf = [[] for _ in range(params.num_envs)]
    rewards_buf = [[] for _ in range(params.num_envs)]

    obs, _ = env.reset()
    states = torch.tensor(obs, dtype=torch.float32, device=device)

    while agent.steps_done < params.total_steps:
        actions, _ = agent.select_action(states)
        obs, rewards, terminated, truncated, _ = env.step(actions.cpu().numpy())

        for i in range(params.num_envs):
            states_buf[i].append(states[i])
            actions_buf[i].append(actions[i])
            rewards_buf[i].append(rewards[i])

        dones = terminated | truncated
        for i in range(params.num_envs):
            if dones[i]:
                agent.update(states_buf[i], actions_buf[i], rewards_buf[i])
                states_buf[i]  = []
                actions_buf[i] = []
                rewards_buf[i] = []

        agent.steps_done += params.num_envs
        states = torch.tensor(obs, dtype=torch.float32, device=device)

        if agent.steps_done % params.evaluate_every < params.num_envs:
            ret = agent.evaluate(eval_env, eval_episodes=params.eval_episodes)
            eval_returns.append(ret)
            print(f'Steps: {agent.steps_done} | Reward: {ret:.1f}')

    env.close()
    eval_env.close()
    return eval_returns


def average_returns(params, device, AgentClass):
    all_runs = []
    start_time = time.time()

    for i in range(params.num_rep):
        print(f"  Repetition {i+1}/{params.num_rep}")
        run_returns = train_one_run(params, device, AgentClass)
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
        label    = exp["label"]
        AgentClass = exp["agent"]
        overrides  = exp["params"]

        print(f"\nRunning: {label}")
        params_dict = base_params.__dict__.copy()
        params_dict.update(overrides)
        params = PGConfig(**params_dict)

        mean_curve, std_curve = average_returns(params, device, AgentClass)
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
        [{"label": "REINFORCE", "agent": REINFORCEAgent, "params": {}}],
        base_params, device,
        title="REINFORCE on CartPole-v1",
        save_name="reinforce"
    )


def exp_all_pg(base_params, device):
    # WE ADD AC AND A2C HERE LATER
    run_experiment(
        [{"label": "REINFORCE", "agent": REINFORCEAgent, "params": {}}],
        base_params, device,
        title="PG Methods on CartPole-v1",
        save_name="pg_all"
    )


EXPERIMENTS = {
    "reinforce": exp_reinforce,
    "all":       exp_all_pg,
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
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Available device:", device)

    base_params = PGConfig()
    EXPERIMENTS[args.experiment](base_params, device)