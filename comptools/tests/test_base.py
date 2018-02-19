
from __future__ import division
import numpy as np
import pytest
from comptools import check_output_dir, partition


def test_check_output_dir_fail():
    with pytest.raises(IOError) as excinfo:
        outdir = '/path/to/nowhere'
        check_output_dir(outdir, makedirs=False)
    error_message = 'The directory {} doesn\'t exist'.format('/path/to')
    assert str(excinfo.value) == error_message


@pytest.mark.parametrize('seq', [list(range(20)), tuple(range(20)), np.arange(20)])
def test_partition_equal_length(seq):
    size = 4
    partitions = partition(seq, size)
    for num_partitions, i in enumerate(partitions, start=1):
        assert len(i) == size


@pytest.mark.parametrize('seq', [list(range(21)), tuple(range(21)), np.arange(21)])
def test_partition_unequal_length(seq):
    size = 4
    partitions = partition(seq, size)
    for num_partitions, i in enumerate(partitions, start=1):
        if num_partitions == 6:
            assert len(i) == 1
        else:
            assert len(i) == size


@pytest.mark.parametrize('seq', [list(range(20)), tuple(range(20)), np.arange(20)])
def test_partition_max_batches(seq):
    size = 4
    max_batches = 2
    partitions = partition(seq, size=size, max_batches=max_batches)

    assert len(list(partitions)) == max_batches


def test_partition_empty():
    assert [] == list(partition([], 3))


def test_partition_large_size():
    # Want to test that the fillvalue string is properly removed when size > len(seq)
    seq = list(range(10))
    partitions = list(partition(seq, 12))
    assert len(partitions) == 1
    assert list(partitions[0]) == seq
