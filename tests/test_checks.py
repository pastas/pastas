from pastas import Model, check


def test_response_memory(ml_solved: Model):
    df = check.response_memory(ml_solved, cutoff=0.95, factor_length_oseries=0.5)
    assert df["pass"].item()


def test_rsq_geq_treshold(ml_solved: Model):
    df = check.rsq_geq_threshold(ml_solved, threshold=0.7)
    assert df["pass"].item()


def test_acf_runs_test(ml_solved: Model):
    df = check.acf_runs_test(ml_solved)
    assert df["pass"].item()


def test_acf_stoffer_toloi(ml_solved: Model):
    df = check.acf_stoffer_toloi_test(ml_solved)
    assert df["pass"].item()


def test_parameter_bounds(ml_solved: Model):
    df = check.parameter_bounds(ml_solved)
    assert df["pass"].all().item()


def test_parameter_uncertainty(ml_solved: Model):
    df = check.uncertainty_parameters(ml_solved, n_std=2)
    assert df["pass"].all().item()


def test_uncertainty_gain(ml_solved: Model):
    df = check.uncertainty_gain(ml_solved, n_std=2)
    assert df["pass"].item()


def test_checklist(ml_solved: Model):
    df = check.checklist(ml_solved, check.checks_brakenhoff_2022)
    assert df["pass"].all().item()
