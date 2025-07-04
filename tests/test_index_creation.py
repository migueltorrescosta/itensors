import numpy as np

from itensors.struct import Index


def test_index_creation_succeeds_with_the_correct_dimension():
    for k in range(100):
        assert Index(k).dimension == k


def test_index_creation_with_distinct_uuids():
    number_of_samples = 10 ** 12
    # The length of the set of index ids is smaller if any id clashes
    assert len({
        Index(np.random.randint(10)).id
        for _
        in range(number_of_samples)
    }) == number_of_samples
