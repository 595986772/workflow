import copy
import io
import unittest

from capacity_protocol import deterministic_capacity_assignment
from input import INPUT_DICT, learning_arg
from evaluate_static_cross_capacity import project_frozen_state
from run_independent_experiment import (
    base_scenario_fingerprint,
    capture_frozen_state,
    seed_everything,
)
from run_static_heterogeneity_suite import (
    STAGE_ORDER,
    STAGES,
    frozen_protocol_spec,
)
from simulator import MEC_Simulator
from static_heterogeneity_protocol import (
    BASELINE_RANDOM_DRAW_CAPACITY,
    CAPACITY_ASSIGNMENT_NAMESPACE,
    CAPACITY_PROFILES,
    CAPACITY_VARIANCES,
    TOTAL_CACHE_BUDGET,
    validate_static_capacity_profiles,
)


def static_simulator(
    seed,
    profile,
    algorithm="nearest_with_service",
):
    seed_everything(seed)
    config = copy.deepcopy(INPUT_DICT)
    config.update(
        {
            "alg": algorithm,
            "Number of users": 3,
            "Number of servers": 10,
            "Number of services": 10,
            "Number of tasks for each user": 6,
            "server capacity": 1,
            "server capacity multiset": CAPACITY_PROFILES[
                profile
            ],
            "capacity assignment namespace": (
                CAPACITY_ASSIGNMENT_NAMESPACE
            ),
            "baseline server capacity": (
                BASELINE_RANDOM_DRAW_CAPACITY
            ),
            "save topology figure": False,
            "seed": seed,
        }
    )
    return MEC_Simulator(
        outputfile=io.StringIO(),
        Input_dict=config,
        learning_arguments=copy.deepcopy(learning_arg),
        filename_png="/tmp",
    )


class StaticHeterogeneityProtocolTest(unittest.TestCase):
    def test_profiles_preserve_budget_and_expected_variance(self):
        validated = validate_static_capacity_profiles()
        self.assertEqual(set(validated), set(CAPACITY_PROFILES))
        self.assertTrue(
            all(
                sum(capacities) == TOTAL_CACHE_BUDGET
                for capacities in validated.values()
            )
        )
        self.assertEqual(
            CAPACITY_VARIANCES,
            {
                "H0": 0.0,
                "H1": 0.2,
                "H2": 0.4,
                "H3": 0.8,
                "H4": 1.4,
            },
        )

    def test_shared_namespace_uses_one_server_order(self):
        marker = deterministic_capacity_assignment(
            list(range(10)),
            number_of_servers=10,
            number_of_services=10,
            seed=7,
            assignment_namespace=CAPACITY_ASSIGNMENT_NAMESPACE,
        )
        for capacities in CAPACITY_PROFILES.values():
            assignment = deterministic_capacity_assignment(
                capacities,
                number_of_servers=10,
                number_of_services=10,
                seed=7,
                assignment_namespace=(
                    CAPACITY_ASSIGNMENT_NAMESPACE
                ),
            )
            ordered = sorted(capacities)
            expected = {
                server_id: ordered[rank]
                for server_id, rank in marker.items()
            }
            self.assertEqual(assignment, expected)

    def test_all_profiles_share_the_physical_deployment(self):
        fingerprints = {
            profile: base_scenario_fingerprint(
                static_simulator(seed=8, profile=profile)
            )
            for profile in CAPACITY_PROFILES
        }
        self.assertEqual(len(set(fingerprints.values())), 1)

    def test_h3_supports_zero_to_three_capacity(self):
        simulator = static_simulator(seed=4, profile="H3")
        capacities = {
            server_id: server.capacity
            for server_id, server in simulator.servers.items()
        }
        self.assertEqual(
            sorted(capacities.values()),
            sorted(CAPACITY_PROFILES["H3"]),
        )
        self.assertEqual(sum(capacities.values()), TOTAL_CACHE_BUDGET)
        self.assertEqual(set(capacities.values()), {0, 1, 2, 3})
        for server in simulator.servers.values():
            real_services = [
                service
                for service in server.services
                if service > 0
            ]
            self.assertLessEqual(
                len(real_services),
                server.capacity,
            )
            if server.capacity == 0:
                self.assertEqual(server.services, [0])
                self.assertGreater(server.frequency, 0)

    def test_governed_stage_order_has_no_dynamic_environment(self):
        self.assertEqual(
            STAGE_ORDER,
            (
                "tests",
                "smoke",
                "screen",
                "converged",
                "ablation",
                "final",
                "generalization",
            ),
        )
        self.assertEqual(
            STAGES["sweep"]["partition"],
            "heterogeneity",
        )
        self.assertEqual(STAGES["final"]["seeds"], list(range(11, 21)))
        protocol = frozen_protocol_spec()
        self.assertFalse(protocol["dynamic_environment_enabled"])
        self.assertEqual(
            protocol["trained_sweep"]["status"],
            "diagnostic_only",
        )
        self.assertFalse(
            protocol["trained_sweep"]["used_for_claims"]
        )
        self.assertFalse(
            protocol["cross_capacity_evaluation"]["retraining"]
        )
        self.assertEqual(
            protocol["environment"]["capacity_profiles"],
            CAPACITY_PROFILES,
        )

    def test_cross_capacity_projection_is_feasible_and_causal(self):
        source = static_simulator(
            seed=5,
            profile="H3",
            algorithm="causal_telemetryPD3QN",
        )
        source_state = capture_frozen_state(source)
        target = static_simulator(
            seed=5,
            profile="H4",
            algorithm="causal_telemetryPD3QN",
        )
        projected, audit = project_frozen_state(
            target,
            source_state,
        )
        self.assertFalse(audit["used_future_target_workload"])
        self.assertEqual(
            sorted(projected["capacities"].values()),
            sorted(CAPACITY_PROFILES["H4"]),
        )
        for server_id, services in projected["services"].items():
            real_services = [
                service for service in services if service > 0
            ]
            self.assertLessEqual(
                len(real_services),
                projected["capacities"][server_id],
            )


if __name__ == "__main__":
    unittest.main()
