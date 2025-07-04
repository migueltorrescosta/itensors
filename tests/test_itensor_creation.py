import numpy as np
import pytest

from itensors.struct import Index, ITensor


def test_itensor_creation_succeeds():
    shape = [3, 4]
    indices = [Index(k) for k in shape]
    ITensor(np.random.rand(*shape), indices)


def test_itensor_creation_fails_on_shape_mismatch():
    shape = [3, 4]
    indices = [Index(k) for k in shape]
    with pytest.raises(Exception):
        ITensor(np.random.rand(*[4, 3]), indices)
