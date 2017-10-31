
import pytest
import comptools


def test_get_level3_sim_files_unpacking_fail():
    with pytest.raises(ValueError):
        gcd_file, files = comptools.simfunctions.get_level3_sim_files(12360,
            just_gcd=True)


def test_get_sim_dict_configs():
    sim_dict = comptools.simfunctions.get_sim_dict()
    configs = comptools.simfunctions.get_sim_configs()
    assert set(sim_dict.values()) == set(configs)


def test_config_to_sim_invalid_config_fail():
    config = 'IC86.2013'
    with pytest.raises(ValueError) as excinfo:
        sim = comptools.simfunctions.config_to_sim(config)
    error_message = 'Invalid config entered'
    assert str(excinfo.value) == error_message


def test_sim_to_config_invalid_sim_fail():
    sim = 12345
    with pytest.raises(ValueError) as excinfo:
        config = comptools.simfunctions.sim_to_config(sim)
    error_message = 'Invalid simulation set, {}, entered'.format(sim)
    assert str(excinfo.value) == error_message


def test_sim_to_config_float_sim_fail():
    sim = '7006'
    with pytest.raises(TypeError) as excinfo:
        config = comptools.simfunctions.sim_to_config(sim)
    error_message = 'sim must be an integer (the simulation set ID)'
    assert str(excinfo.value) == error_message


def test_sim_to_config_12360():
    assert comptools.simfunctions.sim_to_config(12360) == 'IC86.2012'