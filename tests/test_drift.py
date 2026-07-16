from player_churn_model.monitoring.drift import load_reference, build_report, drifted_share

def test_no_drift_reference_vs_itself():
    ref = load_reference()
    my_eval = build_report(ref.copy(), ref)
    assert drifted_share(my_eval) == 0.0

def test_detects_injected_drift():
    ref = load_reference()
    cur = ref.copy()
    cur["events_day1"] = cur["events_day1"] * 1.8
    cur["purchases_day1"] = cur["purchases_day1"] * 2.0   # 2 of 5 = 0.4
    my_eval = build_report(cur, ref)
    assert drifted_share(my_eval) > 0.3