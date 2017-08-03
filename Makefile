
requirements:
	pip freeze > requirements.txt

test:
	py.test -v tests

YEARS = 2011 2012 2013 2014 2015


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

plot-validation-curves:
	python plotting/validation-curves/validation-curves.py --param_name max_depth --param_values 1 2 3 4 5 6 7 8 9 10 --param_type int --cv 10 --pipeline BDT --param_label 'Maximum depth'
	python plotting/validation-curves/validation-curves.py --param_name learning_rate --param_values 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --param_type float --cv 10 --pipeline BDT --param_label 'Learning rate'

environment:
	/data/user/jbourbeau/metaprojects/icerec/V05-01-00/build/env-shell.sh
	workon composition
