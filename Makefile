
requirements:
	pip freeze > requirements.txt

test:
	py.test -v tests

YEARS = 2012 2013 2014 2015


# Processing commands

simulation:
	python processing/process.py --type sim --overwrite --remove --config IC79
	python processing/process.py --type sim --overwrite --remove --config IC86.2012

data:
	for year in $(YEARS); do \
		python processing/process.py --type data --overwrite --remove --config IC86.$$year; \
    done

anisotropy-data:
	python processing/anisotropy/save_anisotropy_dataframe.py

anisotropy-maps:
	for year in $(YEARS); do \
		python processing/anisotropy/process.py --config IC86.$$year --composition all; \
		python processing/anisotropy/process.py --config IC86.$$year --composition light; \
		python processing/anisotropy/process.py --config IC86.$$year --composition heavy; \
    done

anisotropy-maps-low-energy:
	for year in $(YEARS); do \
		python processing/anisotropy/process.py --config IC86.$$year --composition all --low_energy; \
		python processing/anisotropy/process.py --config IC86.$$year --composition light --low_energy; \
		python processing/anisotropy/process.py --config IC86.$$year --composition heavy --low_energy; \
    done

anisotropy-kstest-low-energy:
	for year in $(YEARS); do \
		python processing/anisotropy/ks_test/process_kstest.py --config IC86.$$year --low_energy; \
    done

livetimes:
	python processing/calculate_detector_livetimes.py --config IC86.2012 IC86.2013 IC86.2014 IC86.2015


# Saving models

save-models:
	python models/save_model.py --config IC86.2012


# Plotting commands

plot-data-MC:
	python plotting/plot_data_MC_comparisons.py --config IC86.2012 IC86.2013 IC86.2014 IC86.2015

environment:
	/data/user/jbourbeau/metaprojects/icerec/V05-01-00/build/env-shell.sh
	workon composition
