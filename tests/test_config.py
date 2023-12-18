from serde import toml

from emevo import birth_and_death as bd
from emevo.exp_utils import BDConfig, CfConfig


def test_bdconfig() -> None:
    with open("config/bd/20230530-a035-e020.toml", "r") as f:
        bdconfig = toml.from_toml(BDConfig, f.read())

    birth_fn, hazard_fn = bdconfig.load_models()
    assert isinstance(birth_fn, bd.EnergyLogisticBirth)
    assert isinstance(hazard_fn, bd.ELGompertzHazard)


def test_cfconfig() -> None:
    with open("config/env/20231214-square.toml", "r") as f:
        cfconfig = toml.from_toml(CfConfig, f.read())

    assert cfconfig.sensor_range == "wide"
