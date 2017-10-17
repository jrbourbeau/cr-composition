
import pytest
import comptools


def test_check_output_dir_fail():
    with pytest.raises(IOError) as excinfo:
        outdir = '/path/to/nowhere'
        comptools.check_output_dir(outdir, makedirs=False)
    error_message = 'The directory {} doesn\'t exist'.format('/path/to')
    assert str(excinfo.value) == error_message
