from itensors.struct import Index


def test_index_has_correct_dimension():
    for k in range(100):
        assert Index(k).dimension == k


def test_index_has_unique_uuid_assigned():
    for k in range(100):
        i = Index(k)
        j = Index(k)
        assert i.id != j.id
