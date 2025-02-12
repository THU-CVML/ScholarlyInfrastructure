__version__ = "0.0.2"
import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

__all__ = ['ExperimentModule', 'LibraryPaths', 'MuteWarnings', 'PythonField',
           'RandomVariable', 'append_dict_list', 'author_name_en_us',
           'author_name_zh_cn', 'combine_bilingual_pretty',
           'create_variables_from_dict', 'dataclass_for_torch_decorator',
           'default_on_exception', 'ensure_array', 'experiment_setting',
           'experiment_setting_decorator', 'foo', 'fun_star',
           'get_optuna_search_space', 'github_repo', 'github_user',
           'guess_notebooks_path', 'help', 'import_name', 'infra',
           'is_experiment_setting', 'lib_name', 'lib_name_en_us',
           'lib_name_zh_cn', 'lib_paths', 'logger', 'logging', 'nbscholar',
           'nbscholar_export', 'nbscholar_separate', 'nucleus',
           'operate_on_notebook_in', 'optuna_suggest', 'original_print',
           'partial_with_self', 'pre_init_decorator', 'pretty_name', 'print',
           'process_notebooks_in_folder', 'read_settings_ini',
           'read_settings_ini_none', 'rich_console', 'rv_args',
           'rv_dataclass_metadata_key', 'rv_missing_value', 'setup_paths',
           'show_dataframe_doc', 'split_import_and_code_cells', 'torch',
           'upgrade_command_pip']
