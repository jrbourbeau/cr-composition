.. _path-setup:

:github_url: https://github.com/jrbourbeau/cr-composition

*********************
Filesystem path setup
*********************

Many files are automatically generated and saved throughout the course of running this analysis. Exactly where these files are saved is specified in ``cr-composition/config.yml``. An example ``config.yml`` is shown below

.. code-block:: bash

    # config.yml

    paths:
        metaproject: /data/user/jbourbeau/metaprojects/icerec/V05-01-05
        comp_data_dir: /data/user/jbourbeau/composition
        condor_data_dir: /data/user/jbourbeau/composition/condor
        condor_scratch_dir: /scratch/jbourbeau/composition/condor
        figures_dir: /home/jbourbeau/public_html/figures/composition
        project_root: /home/jbourbeau/cr-composition
        virtualenv_dir: /home/jbourbeau/cr-composition/.virtualenv

The use of these various paths are:

- ``metaproject``: Path to built IceRec installation (as outlined in the :doc:`computing-environment-setup` guide)
- ``comp_data_dir``: Directory where all data files will be saved
- ``condor_data_dir``: Directory to store all condor-related standard output and error files (should be located in ``/data/user``)
- ``condor_scratch_dir``: Directory to store all condor-related log and submit files (should be located in ``/scratch``)
- ``figures_dir``: Directory where all plots will be saved
- ``project_root``: Path to ``cr-composition`` repository
- ``virtualenv_dir``: Path to Python virtual environment used in the analysis (as outlined in the :doc:`computing-environment-setup` guide)

Each of these paths can be changed to your desired location.
