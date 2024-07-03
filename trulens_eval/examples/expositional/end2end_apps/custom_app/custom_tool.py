import dis
import inspect
from typing import Callable, Optional

from examples.expositional.end2end_apps.custom_app.dummy import Dummy

from trulens_eval.tru_custom_app import instrument
from trulens_eval.utils.python import superstack

# Collect a few string methods to use as "tools".
str_maps = []

for name in dir(str):
    func = getattr(str, name)

    if isinstance(func, Callable):
        try:
            ret = func("test me ! LOLS")
            if isinstance(ret, str):
                str_maps.append(func)

        except Exception:
            pass


class CustomTool(Dummy):
    """Dummy tool implementation."""

    def __init__(
        self,
        *args,
        description: Optional[str] = None,
        imp: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if imp is None:
            imp = self.random.choice(str_maps)

        if description is None:
            description = imp.__doc__

        self.description = description
        self.imp = imp

        self.dummy_allocate()

    @instrument
    def invoke(self, data: str):
        """Invoke the dummy tool."""

        self.dummy_wait()

        return self.imp(data)


class CustomStackTool(CustomTool):
    """A tool that returns a rendering of the call stack when it is invokved."""

    last_stack = None
    """The last stack that was rendered.
    
    You can use this to get the readout even if this tool is used deep in an app
    that processes the return from the tool in destructive ways."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, imp=self.save_stack, **kwargs)

    def save_stack(self, data: str) -> str:
        CustomStackTool.last_stack = list(superstack())

        ret = "<table>\n"
        for frame in CustomStackTool.last_stack:
            fmod = inspect.getmodule(frame)
            if fmod is None:
                continue
            else:
                fmod = fmod.__name__
            ffunc = frame.f_code.co_name
            if not fmod.startswith("examples.") or fmod.startswith(
                    "trulens_eval"):
                continue

            bytecode = dis.Bytecode(frame.f_code)
            ret += f"""
            <tr>
                <td>{fmod}</td>
                <td>{ffunc}</td>
                <td><pre><code>{bytecode.dis()}</code></pre></td>
            </tr>
            """

        ret += "</table>\n"

        return ret
