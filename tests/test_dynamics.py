from string_technique_transfer.dynamics import (
    is_adequate_dynamic_pair,
    map_zenodo_dynamic_to_bridge,
    supported_zenodo_dynamics_for_bridge,
)


def test_adequate_pairs():
    assert is_adequate_dynamic_pair("ff", "f")
    assert is_adequate_dynamic_pair("pp", "p")
    assert is_adequate_dynamic_pair("mf", "mf")
    assert not is_adequate_dynamic_pair("pp", "f")
    assert not is_adequate_dynamic_pair("mf", "f")


def test_bridge_f_supports_only_ff():
    assert supported_zenodo_dynamics_for_bridge(["f"]) == ["ff"]
    b, flag, dist = map_zenodo_dynamic_to_bridge("ff", ["f"])
    assert b == "f"
    assert dist is not None and dist < 99
    b2, flag2, dist2 = map_zenodo_dynamic_to_bridge("pp", ["f"])
    assert "inadequate" in flag2 or dist2 >= 99
