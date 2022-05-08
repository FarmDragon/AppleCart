from rrt.rg_rrt import RGRRT


def test_rg_rrt(plt):
    rg_rrt = RGRRT(plt=plt, max_iter=10000)
    rg_rrt.plan()
    rg_rrt.plot()
