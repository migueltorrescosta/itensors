import numpy as np

from itensors.struct import Index, ITensor


# TODO: Need to be careful about the index order checks. This is why ITensors exist!
# TODO: Can the tests be made shorter by pulling the inputs into a testing function?

def test_itensor_simple_multiplication_succeeds():
    i, j, k = [Index(dim) for dim in (3, 4, 5)]
    x = ITensor(np.random.rand(3, 4), [i, j])
    y = ITensor(np.random.rand(4, 5), [j, k])
    c = x * y

    expected_result = np.einsum("ij, jk->ik", x.tensor, y.tensor)
    assert np.allclose(c.tensor, expected_result)
    assert set(c.indices) == {i, k}


def test_itensor_no_shared_indices_succeeds():
    i, j, k, l = [Index(dim) for dim in (5, 7, 11, 13)]
    x = ITensor(np.random.rand(5, 7), [i, j])
    y = ITensor(np.random.rand(11, 13), [k, l])
    c = x * y

    expected_result = np.einsum("ij, kl->ijkl", x.tensor, y.tensor)
    assert np.allclose(c.tensor, expected_result)
    assert set(c.indices) == {i, j, k, l}


def test_itensor_multiple_shared_indices_succeeds():
    i, j, k = [Index(dim) for dim in (5, 7, 11)]
    x = ITensor(np.random.rand(5, 7, 11), [i, j, k])
    y = ITensor(np.random.rand(5, 7), [i, j])
    c = x * y

    expected_result = np.einsum("ijk, ij->k", x.tensor, y.tensor)
    assert np.allclose(c.tensor, expected_result)
    assert set(c.indices) == {k}


def test_itensor_zero_kept_indices_succeeds():
    i, j = [Index(dim) for dim in (5, 7)]
    x = ITensor(np.random.rand(5, 7), [i, j])
    y = ITensor(np.random.rand(5, 7), [i, j])
    c = x * y

    expected_result = np.einsum("ij, ij->", x.tensor, y.tensor)
    assert np.allclose(c.tensor, expected_result)
    assert set(c.indices) == set()
