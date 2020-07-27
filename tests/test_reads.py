import pastas as ps


def test_read_waterbase():
    ps.read_waterbase('tests/data/20180405_010.csv')
    return


def test_read_dino():
    ps.read_dino('tests/data/B32D0136001_1.csv')
    return


def test_read_dino_level_gauge():
    ps.read_dino_level_gauge('tests/data/P43H0001.csv')
    return


def test_read_knmi():
    ps.read_knmi('tests/data/KNMI_Bilt.txt', "EV24")
    return
