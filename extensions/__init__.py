def register_all():
    """
    Explicitly imports submodules to trigger their @register decorators.
    This must be called before parsing the config.
    """
    # The '# noqa: F401' comment tells the linter to ignore "unused import" warnings
    from . import optimizers  # noqa: F401
    from . import criteria  # noqa: F401
    from . import metrics  # noqa: F401
