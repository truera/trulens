def is_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def setup_widget_stdout_stderr():
    from ipywidgets import widgets
    out_stdout = widgets.Output()
    out_stderr = widgets.Output()

    from IPython.display import display
    acc = widgets.Accordion(
        children=[
            widgets.VBox(
                [
                    widgets.VBox([widgets.Label("STDOUT"), out_stdout]),
                    widgets.VBox([widgets.Label("STDERR"), out_stderr])
                ]
            )
        ],
        open=True
    )
    acc.set_title(0, "Dashboard log")
    display(acc)
    return out_stdout, out_stderr
