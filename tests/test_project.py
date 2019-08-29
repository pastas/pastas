from pandas import read_csv
import pastas as ps

ps.set_log_level("ERROR")


def test_create_project():
    pr = ps.Project(name="test")
    return pr


def test_project_add_oseries():
    pr = test_create_project()
    obs = read_csv("tests/data/obs.csv", index_col=0, parse_dates=True,
                   squeeze=True)
    pr.add_oseries(obs, name="heads", metadata={"x": 0.0, "y": 0})
    return pr


def test_project_add_stresses():
    pr = test_project_add_oseries()
    prec = read_csv("tests/data/rain.csv", index_col=0, parse_dates=True,
                    squeeze=True)
    evap = read_csv("tests/data/evap.csv", index_col=0, parse_dates=True,
                    squeeze=True)
    pr.add_stress(prec, name="prec", kind="prec", metadata={"x": 10, "y": 10})
    pr.add_stress(evap, name="evap", kind="evap",
                  metadata={"x": -10, "y": -10})
    return pr


def test_project_add_model():
    pr = test_project_add_stresses()
    pr.add_models(model_name_prefix="my_", model_name_suffix="_model")
    return pr


def test_project_add_recharge():
    pr = test_project_add_model()
    pr.add_recharge()
    return pr


def test_project_solve_models():
    pr = test_project_add_recharge()
    pr.solve_models()
    return pr


def test_project_get_parameters():
    pr = test_project_solve_models()
    return pr.get_parameters(["recharge_A", "noise_alpha"])


def test_project_get_statistics():
    pr = test_project_solve_models()
    return pr.get_statistics(["evp", "aic"])


def test_project_del_model():
    pr = test_project_add_model()
    pr.del_model("my_heads_model")
    return pr


def test_project_del_oseries():
    pr = test_project_add_oseries()
    pr.del_oseries("heads")
    return pr


def test_project_del_stress():
    pr = test_project_add_stresses()
    pr.del_stress("prec")
    return pr


def test_project_get_distances():
    pr = test_project_add_stresses()
    return pr.get_distances()


def test_project_get_nearest_stresses():
    pr = test_project_add_stresses()
    pr.get_nearest_stresses(kind="prec", n=2)


def test_project_dump_to_file():
    pr = test_project_solve_models()
    pr.to_file("testproject.pas")
    return


def test_project_load_from_file():
    pr = ps.io.load("testproject.pas")
    return pr


def test_project_get_oseries_metadata():
    pr = test_project_add_oseries()
    return pr.get_oseries_metadata(["heads"], ["x", "y"])


def test_project_get_oseries_settings():
    pr = test_project_add_oseries()
    return pr.get_oseries_settings(["heads"], ["tmin", "tmax", "freq"])


def test_project_get_metadata():
    pr = test_project_add_stresses()
    return pr.get_metadata()


def test_project_get_file_info():
    pr = test_project_add_oseries()
    return pr.get_file_info()


def test_project_update_model_series():
    pr = test_project_solve_models()
    pr.update_model_series()
    return
