from litmus_spec import (
    ConstraintCase, score_constraint_case, aggregate_constraints,
)


def test_score_constraint_case_runs_all_checks():
    case = ConstraintCase(id="c1", prompt="p",
                          checks=[("exact_bullets", {"n": 2}), ("all_lowercase", {})])
    results = score_constraint_case("- a\n- b", case)
    assert [r["kind"] for r in results] == ["exact_bullets", "all_lowercase"]
    assert all(r["passed"] for r in results)


def test_aggregate_strict_requires_all_checks_in_a_case():
    # case A: both pass; case B: one fails
    a = [{"kind": "exact_bullets", "passed": True, "detail": ""},
         {"kind": "all_lowercase", "passed": True, "detail": ""}]
    b = [{"kind": "exact_bullets", "passed": True, "detail": ""},
         {"kind": "all_lowercase", "passed": False, "detail": ""}]
    agg = aggregate_constraints([a, b])
    assert agg["strict"] == 0.5            # 1 of 2 cases fully passed
    assert agg["loose"] == 3 / 4           # 3 of 4 checks passed
    assert agg["by_kind"]["all_lowercase"] == 0.5
    assert agg["by_kind"]["exact_bullets"] == 1.0
    assert agg["n_cases"] == 2


def test_aggregate_empty_is_safe():
    agg = aggregate_constraints([])
    assert agg["strict"] == 0.0 and agg["loose"] == 0.0 and agg["n_cases"] == 0
