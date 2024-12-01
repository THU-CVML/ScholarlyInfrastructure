import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'fun_star',
        'nucleus',
    },
    submod_attrs={
        'fun_star': [
            'foo',
        ],
        'nucleus': [
            'ExperimentModule',
            'PythonField',
            'RandomVariable',
            'dataclass_for_torch_decorator',
            'experiment_setting',
            'experiment_setting_decorator',
            'get_optuna_search_space',
            'is_experiment_setting',
            'optuna_suggest',
            'pre_init_decorator',
            'rv_dataclass_metadata_key',
            'rv_missing_value',
            'show_dataframe_doc',
        ],
    },
)

__all__ = ['ExperimentModule', 'PythonField', 'RandomVariable',
           'dataclass_for_torch_decorator', 'experiment_setting',
           'experiment_setting_decorator', 'foo', 'fun_star',
           'get_optuna_search_space', 'is_experiment_setting', 'nucleus',
           'optuna_suggest', 'pre_init_decorator', 'rv_dataclass_metadata_key',
           'rv_missing_value', 'show_dataframe_doc']
