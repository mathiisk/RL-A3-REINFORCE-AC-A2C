from dataclasses import dataclass

@dataclass
class PGConfig:
    total_steps: int = 1_000_000
    num_envs: int = 8
    evaluate_every: int = 10_000
    eval_episodes: int = 10
    
    target_update_freq: int = 100
    tau: float = 0.5
    
    gamma: float = 0.99
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    
    hidden_size: int = 128
    
    num_rep: int = 5
    smoothing_window: int = 11