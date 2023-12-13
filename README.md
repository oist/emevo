# EmEvo
[![Tests](https://github.com/oist/emevo/workflows/Tests/badge.svg)](https://github.com/oist/emevo/actions/workflows/tests.yml)

An experimental project for simulating **Em**bodied **Evo**lution of robots.

## Development
First, install [nox](https://github.com/wntrblm/nox). Then compile all requirements by running
```
nox -s compile
```

### Run examples
Circle environment
```
nox -s smoke
```
Qt Widget
```
nox -s smoke --no-install -- smoke-tests/circle_widget.py
```
Asexual reproduction
```
nox -s smoke --no-install -- smoke-tests/circle_asexual_repr.py \
    --steps=10000 \
    --hazard=gompertz \
    --n-agent-sensors=32 \
    --sensor-length=48 \
    --render=moderngl
```

### Run tests
```
nox -s tests
```

# License
[Apache LICENSE 2.0][./LICENSE] holds unless otherwise noted.

`vec2d.py` is copied from [PyMunk](pymunk.org) with the license-header as-is.
