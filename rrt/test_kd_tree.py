import numpy as np


def test_kd_tree():
    from scipy import spatial

    x, y = np.mgrid[0:5, 2:8]
    tree = spatial.KDTree(list(zip(x.ravel(), y.ravel())))
    tree.data
    pts = np.array([[0, 0], [2.1, 2.9]])
    assert all([a == b for a, b in zip(tree.query(pts[0]), np.array([2.0, 0]))])
