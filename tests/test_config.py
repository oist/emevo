from serde import toml

from emevo.exp_utils import CfConfig


def test_cfconfig() -> None:
    with open("config/env/20231214-square.toml", "r") as f:
        cfconfig = toml.from_toml(CfConfig, f.read())

    assert cfconfig.sensor_range == "wide"
