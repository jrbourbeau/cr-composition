
plots: plot-frac-correct plot-validation-curves plot-learning-curve

tests:
	py.test -sv comptools

processing-cluster:
	python processing/process.py  --config IC86.2012 --type sim; \
	python processing/process.py  --config IC86.2012 --type data; \

processing:
	# Models
	echo 'Fitting energy reconstrucion model...'
	python models/save_energy_reco_model.py --config IC86.2012 --pipeline linearregression; \
	echo 'Fitting composition classification model...'
	python models/save_composition_classifier.py --config IC86.2012 --num_groups 2 --pipeline xgboost; \
	# Efficiencies
	echo 'Calculating detector efficiencies...'
	python processing/save_efficiencies.py --config IC86.2012 --num_groups 2 --sigmoid flat; \
	python processing/save_efficiencies.py --config IC86.2012 --num_groups 2 --sigmoid slant; \
	# Livetimes
	echo 'Calculating detector livetimes...'
	python processing/save_livetimes.py --config IC86.2012; \
	# Processed data with quality cuts applied
	echo 'Saving processed dataset with quality cuts applied...'
	python processing/save_processed_data.py; \


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
