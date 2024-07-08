import inspect

from trulens_eval.utils.imports import Dummy
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_NOTEBOOK

with OptionalImports(messages=REQUIREMENT_NOTEBOOK):
    from IPython import get_ipython
    from IPython.core.magic import register_line_cell_magic
    from IPython.display import display
    from ipywidgets import widgets


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except Exception:
        return False


def setup_widget_stdout_stderr():
    out_stdout = widgets.Output()
    out_stderr = widgets.Output()

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


if not isinstance(register_line_cell_magic, Dummy) and is_notebook():

    @register_line_cell_magic
    def writefileinterpolated(line, cell):
        caller_frame = inspect.stack()[2]
        caller_globals = caller_frame.frame.f_globals

        with open(line, 'w') as f:
            f.write(cell.format(**caller_globals))
