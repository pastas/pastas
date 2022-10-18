from pastas.modelcompare import CompareModels

from test_model import test_add_stressmodel, test_add_stressmodels

def test_comparison_class():
    ml = test_add_stressmodel()
    ml.solve()
    ml2 = test_add_stressmodels()
    ml2.solve()
    mc = CompareModels(models=[ml, ml2])
    return mc

def test_comparison_plot():
    mc = test_comparison_class()
    mc.plot()
    return

def test_comparison_plot_custom():
    mc = test_comparison_class()
    mosaic = [
    ["ose", "ose", "met"],
    ["sim", "sim", "tab"],
    ["res", "res", "tab"],
    ["con0", "con0", "dia"],
    ["con1", "con1", 'dia'],
    ["acf", "acf", "dia"],
    ]
    smdict = {0: ["rch", "prec"], 1: ["evap"]}

    mc.initialize_adjust_height_figure(mosaic, figsize=(16, 10), cmap="Dark2", smdict=smdict)
    mc.plot_oseries(axn="ose")
    mc.plot_simulation()
    mc.plot_table_metrics(metric_selection=["evp", "bic"])
    mc.plot_table_params(param_selection=["_A"], param_col="stderr")
    mc.plot_residuals()
    mc.plot_contribution(axn="con{i}")
    mc.plot_table_diagnostics(axn="dia", diag_col="Statistic")
    mc.plot_acf(axn="acf")
    mc.share_xaxes(
        [mc.axes["ose"], mc.axes["sim"], mc.axes["res"], mc.axes["con0"], mc.axes["con1"]]
    )
    return