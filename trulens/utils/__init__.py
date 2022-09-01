import importlib


def try_import(m, msg):
    try:
        return importlib.import_module(m)
    except:
        raise ImportError(
            f"{msg} requires the {m} module. Try 'pip install {m}'."
        )