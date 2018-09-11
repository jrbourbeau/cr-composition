
import os
import shutil
from contextlib import contextmanager


def check_on_condor():
    if os.getenv('_CONDOR_SCRATCH_DIR'):
        on_condor = True
    else:
        on_condor = False

    return on_condor


def localize_path(path):
    local_outdir = os.getenv('_CONDOR_SCRATCH_DIR')
    if local_outdir is None:
        raise ValueError('Environment variable _CONDOR_SCRATCH_DIR does not exist')
    local_path = os.path.join(local_outdir, os.path.basename(path))
    return local_path


def get_file_paths(inputs, output):
    on_condor = check_on_condor()
    if not on_condor:
        return inputs, output
    else:
        output = localize_path(output)
        inputs = list(map(localize_path, inputs))

    return inputs, output


@contextmanager
def localized(inputs, output):
    """ Moves files to local scratch directory on HTCondor node

    Parameters
    ----------
    inputs : array_list
        Input file paths.
    output : str
        Output file path.

    Yields
    ------
    inputs : list
        List of localized input file paths.
    output : str
        Localized output file path.

    Examples
    --------
    >>> with localized(inputs, output) as (inputs, output):
    ...     # Do stuff with local input and output files!
    """
 
    if isinstance(inputs, str):
        single_input = True
        inputs = [inputs]
    else:
        single_input = False
    _inputs_original = inputs
    _output_original = output

    on_condor = check_on_condor()

    inputs, output = get_file_paths(_inputs_original, _output_original)
    if on_condor:
        for original, local in zip(_inputs_original, inputs):
            shutil.copyfile(original, local)

    if single_input:
        yield inputs[0], output
    else:
        yield inputs, output

    if on_condor:
        shutil.move(output, _output_original)
    else:
        pass
