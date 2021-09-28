"""CRISM ML package."""

N_JOBS = -1         # degree of parallelism (default: use all the cores)
USE_CACHE = True    # cache expensive steps (default: True)


def default_configuration():
    """Return the default package configuration."""
    return {
        'version': "default",     # version of the configuration
        # threshold on probabilities for prediction in average spectra
        'match_threshold': 0.0,
        # default spike removal parameters
        'despike_params': ((11, 5), (7, 5), (3, 5)),
        'data_loader': None,       # mineral dataset loader (None: use default)
        'bland_data_loader': None,  # bland dataset loader (None: use default)
        'model_params': {},         # prior parameters for mineral classifier
        'bland_model_params': {},   # prior parameters for bland classifier
        'clays': (14, (14, 18)),    # Al smectite code and AL clays to merge
        'multi_column': True       # select patch only if it spans many columns
    }


CONF = default_configuration()
