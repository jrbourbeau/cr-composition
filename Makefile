
plots: plot-frac-correct plot-validation-curves plot-learning-curve

tests:
	py.test -sv comptools

processing-cluster:
	python write_cluster_wrappers.py; \
	python processing/process.py  --config IC86.2012 --sim --data; \

processing: models efficiencies livetime quality-cuts

models:
	echo 'Fitting energy reconstrucion model...'
	python models/save_energy_reco_model.py --config IC86.2012 --pipeline linearregression; \
	echo 'Fitting composition classification model...'
	python models/save_composition_classifier.py --config IC86.2012 --num_groups 2 --pipeline xgboost; \

efficiencies:
	echo 'Calculating detector efficiencies...'
	python processing/save_efficiencies.py --config IC86.2012 --num_groups 2 --sigmoid flat; \
	python processing/save_efficiencies.py --config IC86.2012 --num_groups 2 --sigmoid slant; \

livetime:
	echo 'Calculating detector livetimes...'
	python processing/save_livetimes.py --config IC86.2012; \

quality-cuts:
	echo 'Saving processed dataset with quality cuts applied...'
	python processing/save_processed_data.py; \


# Run cluster jobs for generating systematics Monte Carlo datasets

systematics: systematic-vem-calibration systematics-light-yield 

systematic-vem-calibration:
	# +/- 3% shift in the Laputop snow correction lambda value
	echo 'Saving VEM calibration systematic datasets...'
	python processing/process.py  --config IC86.2012 --sim --snow_lambda 2.05; \
	python processing/process.py  --config IC86.2012 --sim --snow_lambda 2.45; \

systematics-light-yield:
	# 9.6% and -12.5% shifts in the DOM efficiency
	echo 'Submitting cluster jobs for light yield systematics...'
	python processing/process.py  --config IC86.2012 --sim --dom_eff 0.86625; \
	python processing/process.py  --config IC86.2012 --sim --dom_eff 1.08504; \


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


plot-laputop-performance:
	python plotting/plot_laputop_performance.py --config IC86.2012

plot-flux:
	python plotting/plot_flux.py --config IC86.2011 IC86.2012 IC86.2013 IC86.2014 IC86.2015
