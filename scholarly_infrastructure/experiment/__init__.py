import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

__all__ = [
    "deprecated_alias_of",
    "get_config",
    "iterate_path_hierarchy",
    "load_config",
    "load_overlaying_config",
    "nucleus",
    "read_overlaying_config",
    "save_config",
]
