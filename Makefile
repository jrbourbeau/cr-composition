
analysis: energy-reco efficiencies livetimes

plots: plot-frac-correct plot-validation-curves plot-learning-curve

tests:
	py.test -sv comptools


CONFIGS = IC86.2012
# CONFIGS = IC79.2010 IC86.2012
NUM_GROUPS = 2 3 4


# Processing commands
processing:
	for config in $(CONFIGS); do \
		python processing/process.py --config $$config; \
	done

simulation:
	for config in $(CONFIGS); do \
		python processing/process.py --type sim --config $$config; \
	done

data:
	for config in $(CONFIGS); do \
		python processing/process.py --type data --config $$config; \
    done

efficiencies:
	echo 'Calculating detector efficiencies...'
	for config in $(CONFIGS); do \
		for num_groups in $(NUM_GROUPS); do \
			python processing/save_efficiencies.py --config $$config --num_groups $$num_groups --sigmoid flat; \
			python processing/save_efficiencies.py --config $$config --num_groups $$num_groups --sigmoid slant; \
		done \
	done

livetimes:
	echo 'Calculating detector livetimes...'
	python processing/save_livetimes.py --config $(CONFIGS)


# Commands to save energy reconstruction and compositon classification models
energy-reco:
	echo 'Fitting energy reconstrucion models...'
	for config in $(CONFIGS); do \
		python models/save_energy_reco_model.py --config $$config; \
	done

save-models:
	for config in $(CONFIGS); do \
		python models/save_model.py --config $$config; \
	done


# Plotting commands
plot-data-MC:
	python plotting/plot_data_MC_comparisons.py --config IC86.2012 IC86.2013 IC86.2014 IC86.2015

plot-frac-correct:
	python plotting/plot_frac_correct.py --config IC86.2012

plot-feature-importance:
	python plotting/plot_feature_importance.py --config IC86.2012

plot-feature-covariance:
	python plotting/plot_feature_covariance.py --config IC86.2012

plot-learning-curve:
	for config in $(CONFIGS); do \
		python plotting/plot_learning_curve.py --config $$config; \
	done

plot-validation-curves:
	for config in $(SIM_CONFIGS); do \
		for num_groups in $(NUM_GROUPS); do \
			python plotting/plot_validation_curve.py --config $$config --num_groups $$num_groups --param_name max_depth --param_values 1 2 3 4 5 6 7 8 9 10 --param_type int --cv 10 --param_label 'Maximum depth'; \
		done \
	done


asdfadsf:
	# python plotting/plot_validation_curve.py --config $$config --num_groups $$num_groups --param_name learning_rate --param_values 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --param_type float --cv 10 --param_label 'Learning rate'; \
	# python plotting/plot_validation_curve.py --config $$config --num_groups $$num_groups --param_name n_estimators --param_values 5 10 25 50 100 150 200 300 400 500 --param_type int --cv 10 --param_label 'Number estimators'; \

plot-laputop-performance:
	python plotting/plot_laputop_performance.py --config IC86.2012

plot-flux:
	python plotting/plot_flux.py --config IC86.2011 IC86.2012 IC86.2013 IC86.2014 IC86.2015


# Anisotropy commands
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
