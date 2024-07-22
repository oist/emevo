# EmEvo
[![Tests](https://github.com/oist/emevo/actions/workflows/tests.yml/badge.svg)](https://github.com/oist/emevo/actions/workflows/tests.yml)

- Library + Set of Experiments for simulating **Em**bodied **Evo**lution of robots.
- Enable to simulate evolution of reinforcement learning agents.
  - Only reward evolution is implemented, though.
- Very fast, backed by [jax](https://jax.readthedocs.io/en/latest/index.html) and [phyjax2d](https://github.com/kngwyu/phyjax2d).

**CAUTION**

While I want to make it as open as OpenAI gym was, I didn't have enough working time to write documents and stabilize API.
So now (July 2024), this is just open sourced for reproducibility of my paper and not very usable for others. Apologies.

## Experiments in our [ALIFE2024 paper](https://arxiv.org/abs/2406.15016)

See [alife2024 branch](https://github.com/oist/emevo/tree/alife2024).

## Development
Tooling is based on [nox](https://github.com/wntrblm/nox).
You can compile all requirements by running
```
nox -s compile
```

### Run examples
Test CircleForaging environment
```
nox -s smoke
```

### Run tests
```
nox -s tests
```

# License
[Apache LICENSE 2.0](./LICENSE)
