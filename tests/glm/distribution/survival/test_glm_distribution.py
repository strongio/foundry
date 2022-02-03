from dataclasses import dataclass

import pytest

from foundry.glm.distribution.survival import SurvivalGlmDistribution


class TestSurvivalGlmDistribution:
    @dataclass
    class Params:
        description: str
        alias: str

    @dataclass
    class Fixture:
        glm_distribution: SurvivalGlmDistribution

    @pytest.fixture(
        ids=lambda x: x.description,
        params=[
            Params(
                description='weibull',
                alias='weibull',
            ),
        ]
    )
    def setup(self, request) -> Fixture:
        # TODO
        pass
