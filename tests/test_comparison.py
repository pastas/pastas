import pastas as ps


def test_comparison_plot(ml: ps.Model, ml_sm: ps.Model) -> None:
    ml.solve()
    ml_sm.solve()
    mc = ps.CompareModels(models=[ml, ml_sm])
    _ = mc.plot(legend_kwargs={"ncol": 2})


def test_comparison_plot_custom(ml: ps.Model, ml_sm: ps.Model) -> None:
    ml.solve()
    ml_sm.solve()
    mc = ps.CompareModels(models=[ml, ml_sm])
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
    mc.share_yaxes(
        [
            mc.axes["ose"],
            mc.axes["sim"],
        ]
    )
