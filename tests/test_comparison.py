from pandas import read_csv

import pastas as ps

rain = read_csv("tests/data/rain.csv", index_col=0, parse_dates=True).squeeze("columns")
evap = read_csv("tests/data/evap.csv", index_col=0, parse_dates=True).squeeze("columns")
obs = read_csv("tests/data/obs.csv", index_col=0, parse_dates=True).squeeze("columns")

ml1 = ps.Model(obs.dropna(), name="Test_Model")
sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma(), name="rch")
ml1.add_stressmodel(sm)
ml1.solve(report=False)

ml2 = ps.Model(obs.dropna(), name="Test_Model")
sm1 = ps.StressModel(rain, rfunc=ps.Exponential(), name="prec")
sm2 = ps.StressModel(evap, rfunc=ps.Exponential(), name="evap")
ml2.add_stressmodel([sm1, sm2])
ml2.solve(report=False)


def test_comparison_plot():
    mc = ps.CompareModels(models=[ml1, ml2])
    mc.plot(legend_kwargs={"ncol": 2})
    return


def test_comparison_plot_custom():
    mc = ps.CompareModels(models=[ml1, ml2])
    mosaic = [
        ["ose", "ose", "met"],
        ["sim", "sim", "tab"],
        ["res", "res", "tab"],
        ["con0", "con0", "dia"],
        ["con1", "con1", "dia"],
        ["acf", "acf", "dia"],
    ]
    smdict = {0: ["rch", "prec"], 1: ["evap"]}

    mc.initialize_adjust_height_figure(
        mosaic, figsize=(16, 10), cmap="Dark2", smdict=smdict
    )
    mc.plot_oseries(axn="ose")
    mc.plot_simulation()
    mc.plot_table_metrics(metric_selection=["evp", "bic"])
    mc.plot_table_params(param_selection=["_A"], param_col="stderr")
    mc.plot_residuals()
    mc.plot_contribution(axn="con{i}")
    mc.plot_table_diagnostics(axn="dia", diag_col="Statistic")
    mc.plot_acf(axn="acf")
    mc.share_xaxes(
        [
            mc.axes["ose"],
            mc.axes["sim"],
            mc.axes["res"],
            mc.axes["con0"],
            mc.axes["con1"],
        ]
    )
    return
