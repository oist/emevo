from serde import toml

from emevo import birth_and_death as bd
from emevo import genetic_ops as gops
from emevo.exp_utils import BDConfig, CfConfig, GopsConfig


def test_bdconfig() -> None:
    with open("config/bd/20230530-a035-e020.toml") as f:
        bdconfig = toml.from_toml(BDConfig, f.read())

    birth_fn, hazard_fn = bdconfig.load_models()
    assert isinstance(birth_fn, bd.EnergyLogisticBirth)
    assert isinstance(hazard_fn, bd.ELGompertzHazard)


def test_cfconfig() -> None:
    with open("config/env/20231214-square.toml") as f:
        cfconfig = toml.from_toml(CfConfig, f.read())

    assert cfconfig.sensor_range == "wide"


def test_gopsconfig() -> None:
    with open("config/gops/20231220-mutation-01.toml") as f:
        gopsconfig = toml.from_toml(GopsConfig, f.read())

    mutation = gopsconfig.load_model()
    assert isinstance(mutation, gops.BernoulliMixtureMutation)
    assert mutation.mutation_prob == 0.1
    assert isinstance(mutation.mutator, gops.UniformMutation)
    assert mutation.mutator.min_noise == -1
    assert mutation.mutator.max_noise == 1
