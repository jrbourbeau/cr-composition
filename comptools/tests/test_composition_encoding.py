
import pytest
from comptools import get_comp_list


@pytest.mark.parametrize('num_groups', [2, 3, 4])
def test_get_comp_list_length(num_groups):
    assert len(get_comp_list(num_groups)) == num_groups
