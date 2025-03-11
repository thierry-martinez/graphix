import pytest

from graphix.states import BasicState

@pytest.mark.parametrize("bs", list(BasicState))
def test_try_from_statevector(bs: BasicState):
    assert BasicState.try_from_statevector(bs.value.get_statevector()) == bs

def test_try_from_statevector_fail():
    assert BasicState.try_from_statevector([0.1, 0.1]) == None
