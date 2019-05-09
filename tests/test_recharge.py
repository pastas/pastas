import pastas as ps
import pandas as pd

def test_linear():
    index = pd.date_range("2000-01-01", "2000-01-10")
    prec = pd.Series([1, 2] * 5, index=index)
    evap = prec / 2
    rm = ps.RechargeModel(prec=prec, evap=evap, rfunc=ps.Exponential,
                          recharge="Linear", name="recharge")
    return rm