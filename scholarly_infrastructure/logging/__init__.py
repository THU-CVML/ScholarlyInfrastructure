import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'infra',
        'nucleus',
        'torch',
    },
    submod_attrs={
        'infra': [
            'original_print',
            'print',
            'rich_console',
        ],
        'nucleus': [
            'logger',
            'original_print',
            'print',
            'rich_console',
        ],
    },
)

__all__ = ['infra', 'logger', 'nucleus', 'original_print', 'print',
           'rich_console', 'torch']
