import pastas as ps
import pytest


def test_frequency_is_supported():
    ps.ts._frequency_is_supported("D")
    ps.ts._frequency_is_supported("7D")
    with pytest.raises(Exception):
        ps.ts._frequency_is_supported("SMS")


def test_get_stress_dt():
    assert ps.ts._get_stress_dt("D") == 1.0
    assert ps.ts._get_stress_dt("7D") == 7.0
    assert ps.ts._get_stress_dt("W") == 7.0
    assert ps.ts._get_stress_dt("SMS") == 15.0
