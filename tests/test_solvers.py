import pastas as ps


def test_least_squares(ml_recharge: ps.Model):
    ml_recharge.solve(solver=ps.LeastSquares())


def test_least_squares_lm(ml_recharge: ps.Model):
    ml_recharge.solve(solver=ps.LeastSquares(), method="lm")
    assert ml_recharge.parameters.loc[ml_recharge.parameters.vary, "pmin"].isna().all()


def test_fit_constant(ml_recharge: ps.Model):
    ml_recharge.solve(fit_constant=False)


def test_no_noise(ml_recharge: ps.Model):
    ml_recharge.del_noisemodel()
    ml_recharge.solve()


# test the uncertainty method here
def test_pred_interval(ml_recharge: ps.Model):
    ml_recharge.solve(solver=ps.LeastSquares())
    ml_recharge.solver.prediction_interval(n=10)


def test_ci_simulation(ml_recharge: ps.Model):
    ml_recharge.solve(solver=ps.LeastSquares())
    ml_recharge.solver.ci_simulation(n=10)


def test_ci_block_response(ml_recharge: ps.Model):
    ml_recharge.solve(solver=ps.LeastSquares())
    ml_recharge.solver.ci_block_response(name="rch", n=10)


def test_ci_step_response(ml_recharge: ps.Model):
    ml_recharge.solve(solver=ps.LeastSquares())
    ml_recharge.solver.ci_step_response(name="rch", n=10)


def test_ci_contribution(ml_recharge: ps.Model):
    ml_recharge.solve(solver=ps.LeastSquares())
    ml_recharge.solver.ci_contribution(name="rch", n=10)


# Test the EmceeSolver


def test_emcee(ml_recharge: ps.Model):
    ml_recharge.solve(solver=ps.LeastSquares())
    ml_recharge.del_noisemodel()
    ml_recharge.solve(
        solver=ps.EmceeSolve(nwalkers=10),
        initial=False,
        fit_constant=False,
        steps=2,
    )
