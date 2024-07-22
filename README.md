# EmEvo ALIFE2024 branch

This branch is reserved for reproducibility of our [ALIFE2024 paper](https://arxiv.org/abs/2406.15016).

## Requirements
In addition to packages in [pyproject.toml](./pyproject.toml), [typer](https://typer.tiangolo.com/) is required to run experiments.
You can just install it by `pip install typer` in your venv.
Maybe runnable in Mac or Windows, but I only tested this on Linux (Ubuntu and Arch).

Since the simulation is really slow on CPU, I strongly recommend to use GPU.
You can refer to [jax's document](https://jax.readthedocs.io/en/latest/installation.html) to how to make GPU work properly with Jax.
The easiest way is running `pip install jax[cuda12]` if you have a NVIDIA driver newly enough.
This installs all necessary CUDA binaries from PyPI.

I guess Google TPU may also work, but I have never tested it.

## Commands for reproducing experiments in our [ALIFE2024 paper](https://arxiv.org/abs/2406.15016)

In the commands below, `{{ seed }}` should be replaced by actual value.
I used 1 to 5 in most experiments, but when agents get extinct, some additional seeds are used to prepare five successful runs.
`{{ your_log_dir }}` also should replaced.
Note that these commands generate 10~20 GBs of log files.

You can use `--force-gpu` flag to exit program when GPU is not available.

### 'Baseline' environment

```
python experiments/cf_simple.py \\
    evolve \\
    --seed={{ seed }} \\
    --action-cost=2e-5 \\
    --act-reward-coef=0.01 \\
    --cfconfig-path=config/env/20240607-normal.toml \\
    --gopsconfig-path=config/gops/20240326-cauthy-002.toml \\
    --logdir={{ your_log_dir }}
```

### 'Large' environment

```
python experiments/cf_simple.py \\
    evolve \\
    --seed={{ seed }} \\
    --action-cost=2e-5 \\
    --act-reward-coef=0.01 \\
    --cfconfig-path=config/env/20240607-large.toml \\
    --gopsconfig-path=config/gops/20240326-cauthy-002.toml \\
    --logdir={{ your_log_dir }}
```

### 'Small' environment

```
python experiments/cf_simple.py \\
    evolve \\
    --seed={{ seed }} \\
    --action-cost=2e-5 \\
    --act-reward-coef=0.01 \\
    --cfconfig-path=config/env/20240607-small.toml \\
    --gopsconfig-path=config/gops/20240326-cauthy-002.toml \\
    --logdir={{ your_log_dir }}
```

### 'Centered food' environment

```
python experiments/cf_simple.py \\
    evolve \\
    --seed={{ seed }} \\
    --action-cost=2e-5 \\
    --act-reward-coef=0.01 \\
    --cfconfig-path=config/env/20240607-centered.toml \\
    --gopsconfig-path=config/gops/20240326-cauthy-002.toml \\
    --logdir={{ your_log_dir }}
```


### 'Food relocation' environment

```
python experiments/cf_simple.py \
    evolve \
    --seed={{ seed }} \
    --action-cost=2e-5 \
    --act-reward-coef=0.01 \
    --cfconfig-path=config/env/20240607-moving.toml \
    --gopsconfig-path=config/gops/20240326-cauthy-002.toml \
    --logdir={{ your_log_dir }}
```

### Environment with 'Poor Foods'

```
python experiments/cf_simple.py \
    evolve \
    --seed={{ seed }} \
    --action-cost=2e-5 \
    --act-reward-coef=0.01 \
    --cfconfig-path=config/env/20240609-2f-93.toml \
    --gopsconfig-path=config/gops/20240326-cauthy-002.toml \
    --logdir={{ your_log_dir }}
```

### Environment with 'Poison'

```
python experiments/cf_simple.py \
    evolve \
    --seed={{ seed }} \
    --action-cost=2e-5 \
    --act-reward-coef=0.01 \
    --cfconfig-path=config/env/20240611-poison-06.toml \
    --gopsconfig-path=config/gops/20240326-cauthy-002.toml \
    --logdir={{ your_log_dir }}
```

## Visualizing learned agents

You can use [PySide6](https://pypi.org/project/PySide6/)-based widget to explore the behavior of evolved agents.
Below is the example of running the widget for baseline setting from (1024000 * 8 =) 8192000 step to 8392000 step.

```
python \
    experiments/cf_simple.py \
    widget \
    {{ your_log_dir }}/state-9.npz \
    --cfconfig-path=config/env/20240607-normal.toml \\
    --log-path={{ your_log_dir }}/log-9.parquet \
    --start=0 \
    --end=200000 \
    --profile-and-rewards-path={{ your_log_dir }}/profile_and_rewards.parquet
```

You can also watch agents in [demo site](https://emevo-alife2024.pages.dev/).

# License
[Apache LICENSE 2.0](./LICENSE)
