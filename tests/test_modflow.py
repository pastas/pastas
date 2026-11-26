"""Tests for the pastas.modflow module (flopy-MODFLOW to Pastas bridge)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

flopy = pytest.importorskip("flopy")

import pastas as ps

_HEADER_DTYPE = np.dtype(
    [
        ("kstp", "<i4"),
        ("kper", "<i4"),
        ("pertim", "<f4"),
        ("totim", "<f4"),
        ("text", "S16"),
        ("ncol", "<i4"),
        ("nrow", "<i4"),
        ("ilay", "<i4"),
    ]
)


def write_head_file(path: Path, records: list, nrow: int, ncol: int) -> None:
    """Write a binary MODFLOW head file readable by flopy.utils.HeadFile.

    records: list of (kstp, kper, pertim, totim, ilay, 2D data array).

    """
    with open(path, "wb") as f:
        for kstp, kper, pertim, totim, ilay, data in records:
            header = np.array(
                [(kstp, kper, pertim, totim, b"HEAD", ncol, nrow, ilay)],
                dtype=_HEADER_DTYPE,
            )
            header.tofile(f)
            np.asarray(data, dtype="<f4").tofile(f)


def make_head_records(perlen, nstp, nrow, ncol, dry_cell=None):
    """Create head records with a linearly declining head over time."""
    records = []
    totim = 0.0
    kstp = 0
    for kper, (pl, ns) in enumerate(zip(perlen, nstp)):
        for _ in range(ns):
            kstp += 1
            totim += pl / ns
            data = np.full((nrow, ncol), 5.0 - 0.01 * kstp, dtype=np.float32)
            if dry_cell is not None and kper == len(perlen) - 1:
                data[dry_cell[1], dry_cell[2]] = -999.99
            records.append((kstp, kper + 1, pl / ns, totim, 1, data))
    return records


@pytest.fixture
def mf2005_model(tmp_path):
    """A small MODFLOW-2005 model with WEL and RCH and a synthetic head file."""
    ws = tmp_path / "mf2005"
    ws.mkdir()
    ml = flopy.modflow.Modflow("test2005", model_ws=str(ws))
    flopy.modflow.ModflowDis(
        ml,
        nlay=1,
        nrow=4,
        ncol=5,
        delr=100.0,
        delc=100.0,
        top=10.0,
        botm=0.0,
        nper=3,
        perlen=[10.0, 20.0, 30.0],
        nstp=[1, 2, 3],
        tsmult=1.0,
        start_datetime="2000-01-01",
        steady=False,
    )
    flopy.modflow.ModflowBas(ml, ibound=1, strt=5.0)
    flopy.modflow.ModflowLpf(ml, hk=10.0)
    flopy.modflow.ModflowWel(
        ml,
        stress_period_data={
            0: [[0, 1, 2, -100.0], [0, 3, 3, -200.0]],
            1: [[0, 1, 2, -150.0]],
            # stress period 2 is not defined and repeats stress period 1
        },
    )
    flopy.modflow.ModflowRch(ml, rech={0: 0.001, 1: 0.002, 2: 0.0015})
    flopy.modflow.ModflowOc(ml)
    write_head_file(
        ws / "test2005.hds",
        make_head_records([10.0, 20.0, 30.0], [1, 2, 3], 4, 5, dry_cell=(0, 0, 0)),
        4,
        5,
    )
    return ml


@pytest.fixture
def mf6_model(tmp_path):
    """A small MODFLOW 6 model (DIS) with WEL and RCHA and a synthetic head file."""
    ws = tmp_path / "mf6"
    ws.mkdir()
    sim = flopy.mf6.MFSimulation(sim_name="sim", sim_ws=str(ws), exe_name="mf6")
    flopy.mf6.ModflowTdis(
        sim,
        nper=2,
        perioddata=[(10.0, 2, 1.0), (20.0, 1, 1.0)],
        time_units="days",
        start_date_time="2000-01-01",
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=1, nrow=4, ncol=5, delr=100.0, delc=100.0, top=10.0, botm=0.0
    )
    flopy.mf6.ModflowGwfic(gwf, strt=5.0)
    flopy.mf6.ModflowGwfnpf(gwf, k=10.0)
    flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data={
            0: [[(0, 1, 2), -100.0], [(0, 3, 3), -200.0]],
            1: [[(0, 1, 2), -150.0]],
        },
        maxbound=2,
    )
    flopy.mf6.ModflowGwfrcha(gwf, recharge={0: 0.001, 1: 0.002})
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="gwf.hds")
    write_head_file(ws / "gwf.hds", make_head_records([10.0, 20.0], [2, 1], 4, 5), 4, 5)
    return gwf


@pytest.fixture
def mf6_disv_model(tmp_path):
    """A small MODFLOW 6 model with a vertex grid (DISV), WEL and list-based RCH."""
    ws = tmp_path / "mf6disv"
    ws.mkdir()
    sim = flopy.mf6.MFSimulation(sim_name="sim", sim_ws=str(ws), exe_name="mf6")
    flopy.mf6.ModflowTdis(
        sim,
        nper=3,
        perioddata=[(10.0, 1, 1.0), (20.0, 1, 1.0), (30.0, 1, 1.0)],
        time_units="days",
        start_date_time="2000-01-01",
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwfv", save_flows=True)
    # 3x3 vertices forming a 2x2 grid of unit-square cells
    vertices = [(i, float(i % 3), float(i // 3)) for i in range(9)]
    cell2d = [
        (0, 0.5, 0.5, 4, 0, 1, 4, 3),
        (1, 1.5, 0.5, 4, 1, 2, 5, 4),
        (2, 0.5, 1.5, 4, 3, 4, 7, 6),
        (3, 1.5, 1.5, 4, 4, 5, 8, 7),
    ]
    flopy.mf6.ModflowGwfdisv(
        gwf,
        nlay=1,
        ncpl=4,
        nvert=9,
        vertices=vertices,
        cell2d=cell2d,
        top=10.0,
        botm=0.0,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=5.0)
    flopy.mf6.ModflowGwfnpf(gwf, k=10.0)
    flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data={0: [[(0, 0), -100.0]], 1: [[(0, 2), -50.0]]},
        maxbound=2,
    )
    flopy.mf6.ModflowGwfrch(gwf, stress_period_data={0: [[(0, 0), 0.001]]})
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="gwfv.hds")
    # head file for a vertex grid: one value per cell (ncpl=4)
    records = [
        (1, 1, 10.0, 10.0, 1, np.full((1, 4), 5.0, dtype=np.float32)),
        (1, 2, 20.0, 30.0, 1, np.full((1, 4), 4.99, dtype=np.float32)),
        (1, 3, 30.0, 60.0, 1, np.full((1, 4), 4.98, dtype=np.float32)),
    ]
    with open(ws / "gwfv.hds", "wb") as f:
        for kstp, kper, pertim, totim, ilay, data in records:
            header = np.array(
                [(kstp, kper, pertim, totim, b"HEAD", 4, 1, ilay)], dtype=_HEADER_DTYPE
            )
            header.tofile(f)
            np.asarray(data, dtype="<f4").tofile(f)
    return gwf


class TestStressPeriodTimes:
    def test_mf2005_start(self, mf2005_model):
        times = ps.modflow.get_stress_period_times(mf2005_model, when="start")
        expected = pd.DatetimeIndex(["2000-01-01", "2000-01-11", "2000-01-31"])
        pd.testing.assert_index_equal(times, expected)

    def test_mf2005_end(self, mf2005_model):
        times = ps.modflow.get_stress_period_times(mf2005_model, when="end")
        expected = pd.DatetimeIndex(["2000-01-11", "2000-01-31", "2000-03-01"])
        pd.testing.assert_index_equal(times, expected)

    def test_mf6(self, mf6_model):
        times = ps.modflow.get_stress_period_times(mf6_model, when="end")
        expected = pd.DatetimeIndex(["2000-01-11", "2000-01-31"])
        pd.testing.assert_index_equal(times, expected)

    def test_invalid_when(self, mf2005_model):
        with pytest.raises(ValueError, match="when"):
            ps.modflow.get_stress_period_times(mf2005_model, when="invalid")


class TestHeadSeries:
    def test_mf2005(self, mf2005_model):
        heads = ps.modflow.get_head_series(mf2005_model, cellid=(0, 1, 2))
        assert isinstance(heads, pd.Series)
        assert heads.name == "L0R1C2"
        assert len(heads) == 6  # nstp = 1 + 2 + 3
        # totim values: 10, 20, 30, 40, 50, 60 days after 2000-01-01
        expected_index = pd.DatetimeIndex(
            pd.to_datetime("2000-01-01")
            + pd.to_timedelta([10, 20, 30, 40, 50, 60], "D")
        )
        pd.testing.assert_index_equal(heads.index, expected_index)
        assert np.allclose(heads.values, 5.0 - 0.01 * np.arange(1, 7))

    def test_mf2005_mask_dry(self, mf2005_model):
        heads = ps.modflow.get_head_series(mf2005_model, cellid=(0, 0, 0))
        assert np.isnan(heads.iloc[-1])  # NaN for the dry cell
        heads_nomask = ps.modflow.get_head_series(
            mf2005_model, cellid=(0, 0, 0), mask_dry=False
        )
        assert heads_nomask.iloc[-1] == pytest.approx(-999.99, rel=1e-5)

    def test_mf6(self, mf6_model):
        heads = ps.modflow.get_head_series(mf6_model, cellid=(0, 2, 3))
        assert len(heads) == 3
        expected_index = pd.DatetimeIndex(
            pd.to_datetime("2000-01-01") + pd.to_timedelta([5, 10, 30], "D")
        )
        pd.testing.assert_index_equal(heads.index, expected_index)

    def test_mf6_disv(self, mf6_disv_model):
        heads = ps.modflow.get_head_series(mf6_disv_model, cellid=(0, 2))
        assert heads.name == "L0C2"
        assert len(heads) == 3
        assert np.allclose(heads.values, [5.0, 4.99, 4.98])

    def test_default_cellid(self, mf2005_model):
        heads = ps.modflow.get_head_series(mf2005_model)
        assert heads.name == "L0R0C0"

    def test_invalid_cellid(self, mf2005_model):
        with pytest.raises(ValueError, match="cellid"):
            ps.modflow.get_head_series(mf2005_model, cellid=(0, 1))

    def test_missing_headfile(self, tmp_path):
        ml = flopy.modflow.Modflow("norun", model_ws=str(tmp_path))
        flopy.modflow.ModflowDis(
            ml, nlay=1, nrow=2, ncol=2, delr=1.0, delc=1.0, top=1.0, botm=0.0
        )
        with pytest.raises(FileNotFoundError, match="head"):
            ps.modflow.get_head_series(ml, cellid=(0, 0, 0))


class TestStressSeries:
    def test_mf2005_total(self, mf2005_model):
        q = ps.modflow.get_stress_series(mf2005_model, package="WEL")
        assert q.name == "wel"
        # daily index from 2000-01-01 to 2000-03-01 (60 days of stress periods)
        assert len(q) == 61
        assert q.index[0] == pd.Timestamp("2000-01-01")
        assert q.index[-1] == pd.Timestamp("2000-03-01")
        # stress is constant within each stress period
        assert q.loc["2000-01-01"] == -300.0
        assert q.loc["2000-01-10"] == -300.0
        assert q.loc["2000-01-11"] == -150.0
        # stress period 2 repeats stress period 1 (MODFLOW behavior)
        assert q.loc["2000-01-31"] == -150.0

    def test_mf2005_as_dict(self, mf2005_model):
        wells = ps.modflow.get_stress_series(mf2005_model, package="WEL", as_dict=True)
        assert set(wells) == {"L0R1C2", "L0R3C3"}
        assert wells["L0R1C2"].name == "wel_L0R1C2"
        assert wells["L0R1C2"].loc["2000-01-01"] == -100.0
        assert wells["L0R1C2"].loc["2000-01-11"] == -150.0
        assert wells["L0R3C3"].loc["2000-01-01"] == -200.0
        assert wells["L0R3C3"].loc["2000-01-11"] == 0.0

    def test_mf2005_freq_none(self, mf2005_model):
        q = ps.modflow.get_stress_series(mf2005_model, package="WEL", freq=None)
        expected = pd.DatetimeIndex(["2000-01-01", "2000-01-11", "2000-01-31"])
        pd.testing.assert_index_equal(q.index, expected)
        assert np.allclose(q.values, [-300.0, -150.0, -150.0])

    def test_mf6(self, mf6_model):
        wells = ps.modflow.get_stress_series(mf6_model, package="WEL", as_dict=True)
        assert set(wells) == {"L0R1C2", "L0R3C3"}
        assert wells["L0R1C2"].loc["2000-01-01"] == -100.0
        assert wells["L0R1C2"].loc["2000-01-11"] == -150.0
        q = ps.modflow.get_stress_series(mf6_model, package="WEL")
        assert len(q) == 31  # daily from 2000-01-01 to 2000-01-31
        assert q.loc["2000-01-01"] == -300.0
        assert q.loc["2000-01-11"] == -150.0

    def test_mf6_disv(self, mf6_disv_model):
        wells = ps.modflow.get_stress_series(
            mf6_disv_model, package="WEL", as_dict=True
        )
        assert set(wells) == {"L0C0", "L0C2"}
        assert wells["L0C0"].loc["2000-01-01"] == -100.0
        assert wells["L0C0"].loc["2000-01-11"] == 0.0
        assert wells["L0C2"].loc["2000-01-01"] == 0.0
        assert wells["L0C2"].loc["2000-01-11"] == -50.0

    def test_agg_mean(self, mf2005_model):
        q = ps.modflow.get_stress_series(mf2005_model, package="WEL", agg="mean")
        assert q.loc["2000-01-01"] == -150.0  # mean of -100 and -200
        assert q.loc["2000-01-11"] == -150.0  # single well

    def test_missing_package(self, mf2005_model):
        with pytest.raises(ValueError, match="DRN"):
            ps.modflow.get_stress_series(mf2005_model, package="DRN")

    def test_invalid_column(self, mf2005_model):
        with pytest.raises(ValueError, match="olumn"):
            ps.modflow.get_stress_series(mf2005_model, package="WEL", column="nope")

    def test_explicit_column(self, mf2005_model):
        q = ps.modflow.get_stress_series(mf2005_model, package="WEL", column="flux")
        assert q.loc["2000-01-01"] == -300.0


class TestRechargeSeries:
    def test_mf2005(self, mf2005_model):
        rch = ps.modflow.get_recharge_series(mf2005_model)
        assert rch.name == "recharge"
        assert len(rch) == 61
        assert rch.index[0] == pd.Timestamp("2000-01-01")
        assert rch.loc["2000-01-01"] == pytest.approx(0.001)
        assert rch.loc["2000-01-11"] == pytest.approx(0.002)
        assert rch.loc["2000-01-31"] == pytest.approx(0.0015)

    def test_mf6_rcha(self, mf6_model):
        rch = ps.modflow.get_recharge_series(mf6_model)
        assert len(rch) == 31
        assert rch.loc["2000-01-01"] == pytest.approx(0.001)
        assert rch.loc["2000-01-11"] == pytest.approx(0.002)

    def test_mf6_list_based_rch(self, mf6_disv_model):
        rch = ps.modflow.get_recharge_series(mf6_disv_model)
        # only cell 0 has recharge in kper 0; kper 1 and 2 repeat kper 0
        assert len(rch) == 61
        assert np.allclose(rch.values, 0.001)

    def test_missing_package(self, mf2005_model):
        with pytest.raises(ValueError, match="EVT"):
            ps.modflow.get_recharge_series(mf2005_model, package="EVT")


class TestCellDistance:
    def test_mf2005(self, mf2005_model):
        # cell centers: (row 1, col 2) -> (250, 250); (row 3, col 3) -> (350, 50)
        distance = ps.modflow.get_cell_distance(mf2005_model, (0, 1, 2), (0, 3, 3))
        assert distance == pytest.approx(np.hypot(100.0, 200.0))

    def test_same_cell(self, mf2005_model):
        assert ps.modflow.get_cell_distance(mf2005_model, (0, 1, 2), (0, 1, 2)) == 0.0

    def test_mf6_disv(self, mf6_disv_model):
        # cell centers: cell 0 -> (0.5, 0.5); cell 2 -> (0.5, 1.5)
        distance = ps.modflow.get_cell_distance(mf6_disv_model, (0, 0), (0, 2))
        assert distance == pytest.approx(1.0)


class TestModelFromFlopy:
    def test_mf2005_single(self, mf2005_model):
        ml = ps.modflow.model_from_flopy(mf2005_model, cellids=(0, 2, 4))
        assert isinstance(ml, ps.Model)
        assert ml.name == "test2005_L0R2C4"
        assert set(ml.stressmodels) == {"recharge", "wells"}
        assert len(ml.oseries.series_original) == 6
        assert ml.oseries.metadata["x"] == pytest.approx(450.0)
        assert ml.oseries.metadata["y"] == pytest.approx(150.0)
        # two wells with distances from the model grid
        sm = ml.stressmodels["wells"]
        assert len(sm.distances) == 2
        expected = sorted(
            [
                ps.modflow.get_cell_distance(mf2005_model, (0, 1, 2), (0, 2, 4)),
                ps.modflow.get_cell_distance(mf2005_model, (0, 3, 3), (0, 2, 4)),
            ]
        )
        assert sorted(sm.distances.values) == pytest.approx(expected)
        # model can simulate
        sim = ml.simulate()
        assert len(sim) > 0
        assert not sim.isna().all()

    def test_mf2005_multiple(self, mf2005_model):
        models = ps.modflow.model_from_flopy(
            mf2005_model, cellids=[(0, 0, 0), (0, 2, 4)]
        )
        assert isinstance(models, dict)
        assert set(models) == {"L0R0C0", "L0R2C4"}
        assert all(isinstance(m, ps.Model) for m in models.values())

    def test_mf2005_custom_name(self, mf2005_model):
        ml = ps.modflow.model_from_flopy(mf2005_model, cellids=(0, 2, 4), name="obs")
        assert ml.name == "obs"

    def test_mf2005_solve(self, mf2005_model):
        ml = ps.modflow.model_from_flopy(
            mf2005_model, cellids=(0, 2, 4), solve=True, warmup=10.0
        )
        assert not ml.parameters["optimal"].isna().any()
        assert ml.stats.rmse() > 0

    def test_mf2005_no_wells_no_recharge(self, mf2005_model):
        ml = ps.modflow.model_from_flopy(
            mf2005_model, cellids=(0, 2, 4), add_wells=False, add_recharge=False
        )
        assert ml.stressmodels == {}

    def test_mf2005_well_max_distance(self, mf2005_model):
        ml = ps.modflow.model_from_flopy(
            mf2005_model, cellids=(0, 2, 4), well_max_distance=150.0
        )
        # only the well at (0, 3, 3), 141 m away, is within 150 m
        assert len(ml.stressmodels["wells"].distances) == 1

    def test_mf6(self, mf6_model):
        ml = ps.modflow.model_from_flopy(mf6_model, cellids=(0, 2, 4), solve=True)
        assert isinstance(ml, ps.Model)
        assert set(ml.stressmodels) == {"recharge", "wells"}
        assert len(ml.oseries.series_original) == 3
        assert not ml.parameters["optimal"].isna().any()

    def test_mf6_disv(self, mf6_disv_model):
        ml = ps.modflow.model_from_flopy(mf6_disv_model, cellids=(0, 3))
        assert ml.name == "gwfv_L0C3"
        assert set(ml.stressmodels) == {"recharge", "wells"}
        assert len(ml.oseries.series_original) == 3
        sm = ml.stressmodels["wells"]
        assert len(sm.distances) == 2
        assert sorted(sm.distances.values) == pytest.approx(
            sorted(
                [
                    ps.modflow.get_cell_distance(mf6_disv_model, (0, 0), (0, 3)),
                    ps.modflow.get_cell_distance(mf6_disv_model, (0, 2), (0, 3)),
                ]
            )
        )

    def test_headfile_argument(self, mf2005_model, tmp_path):
        headfile = tmp_path / "mf2005" / "test2005.hds"
        ml = ps.modflow.model_from_flopy(
            mf2005_model, cellids=(0, 2, 4), headfile=str(headfile)
        )
        assert len(ml.oseries.series_original) == 6


class TestImportError:
    def test_missing_flopy(self, mf2005_model, monkeypatch):
        monkeypatch.setitem(sys.modules, "flopy", None)
        with pytest.raises(ImportError, match="flopy"):
            ps.modflow.get_stress_period_times(mf2005_model)
