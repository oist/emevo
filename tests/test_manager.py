from emevo import Body, Manager


class FakeBody(Body):
    def __init__(self, name, generation):
        super().__init__(name, generation)
        self.energy = 1000

    @property
    def action_space(self):
        return None

    @property
    def observation_space(self):
        return None


def test_asexual():
    pass
