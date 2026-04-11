import numpy as np

from providence.strategy.kelly import enumerate_top3_scenarios
from providence.strategy.types import RaceIndexMap, RacePredictionBundle


def test_enumerate_top3_scenarios_prefers_scenario_strengths_over_scores():
    bundle = RacePredictionBundle(
        race_id=1,
        model_version="v011",
        temperature=1.0,
        # raw scores would favor index 0
        scores=(3.0, 1.0, 0.5),
        # calibrated strengths favor index 2
        scenario_strengths=(0.1, 0.2, 0.7),
        index_map=RaceIndexMap(index_to_post_position=(1, 2, 3), index_to_entry_id=(11, 12, 13)),
        ticket_probs={},
        features_total_races=(10, 10, 10),
    )
    scenarios = enumerate_top3_scenarios(bundle)
    top = max(scenarios, key=lambda row: row[1])
    assert top[0][0] == 3
