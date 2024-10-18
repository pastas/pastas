import pastas as ps


def test_least_squares(ml: ps.Model):
    ml.solve(solver=ps.LeastSquares())


def test_least_squares_lm(ml: ps.Model):
    ml.solve(solver=ps.LeastSquares(), method="lm")
    assert ml.parameters.loc[ml.parameters.vary, "pmin"].isna().all()


def test_fit_constant(ml: ps.Model):
    ml.solve(fit_constant=False)


def test_no_noise(ml: ps.Model):
    ml.del_noisemodel()
    ml.solve()


# test the uncertainty method here
def test_pred_interval(ml: ps.Model):
    ml.solve(solver=ps.LeastSquares())
    ml.solver.prediction_interval(n=10)


def test_ci_simulation(ml: ps.Model):
    ml.solve(solver=ps.LeastSquares())
    ml.solver.ci_simulation(n=10)


def test_ci_block_response(ml: ps.Model):
    ml.solve(solver=ps.LeastSquares())
    ml.solver.ci_block_response(name="rch", n=10)


def test_ci_step_response(ml: ps.Model):
    ml.solve(solver=ps.LeastSquares())
    ml.solver.ci_step_response(name="rch", n=10)


def test_ci_contribution(ml: ps.Model):
    ml.solve(solver=ps.LeastSquares())
    ml.solver.ci_contribution(name="rch", n=10)


# Test the EmceeSolver


def test_emcee(ml: ps.Model):
    ml.solve(solver=ps.LeastSquares())
    ml.del_noisemodel()
    ml.solve(
        solver=ps.EmceeSolve(nwalkers=20),
        initial=False,
        fit_constant=False,
        steps=10,
    )
