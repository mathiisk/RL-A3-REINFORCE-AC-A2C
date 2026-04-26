from dataclasses import dataclass

@dataclass
class PGConfig:
    total_steps: int = 1_000_000
    num_envs: int = 8
    evaluate_every: int = 10_000
    eval_episodes: int = 10
    
    target_update_freq: int = 50
    tau: float = 0.01
    
    gamma: float = 0.99
    lr_actor: float = 1e-4
    lr_critic: float = 1e-4
    
    hidden_size: int = 64
    
    num_rep: int = 5
    smoothing_window: int = 11