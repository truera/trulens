import inspect
from typing import Callable, List, Optional

from trulens.apps.app import instrument
from trulens.core.utils import python as python_utils

from examples.dev.dummy_app.dummy import Dummy

# A few string->string functions to use as "tools".
str_maps = [
    str.capitalize,
    str.casefold,
    str.lower,
    str.lstrip,
    str.rstrip,
    str.strip,
    str.swapcase,
    str.title,
    str.upper,
]


class DummyTool(Dummy):
    """Dummy tool implementation."""

    def __init__(
        self,
        *args,
        description: Optional[str] = None,
        imp: Optional[Callable] = None,
        **kwargs,
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

    @instrument
    async def ainvoke(self, data: str):
        """Invoke the dummy tool."""

        await self.dummy_await()

        return self.imp(data)


class DummyStackTool(DummyTool):
    """A tool that returns a rendering of the call stack when it is invoked."""

    last_stacks: List[python_utils.WeakWrapper] = []
    """The stacks seen during save_stack invocations.

    You can use this to get the readout even if this tool is used deep in an app
    that processes the return from the tool in destructive ways."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, imp=self.save_stack, **kwargs)

    @staticmethod
    def save_stack(data: str) -> str:
        """Save the call stack for later rendering or inspection in the recorded
        trace."""

        # Save a copy of the live stack here which makes it available for the
        # developer to check later. Note that it overwrites any previously saved
        # stack. This is here because we cannot serialize everything about a
        # stack to include in the return of this method but at the same time
        # want to be able to take a look at those things which we didn't
        # serialize.
        current_stack = list(f.frame for f in inspect.stack())
        DummyStackTool.last_stacks.append(
            # Has to be a weakref to prevent GC test failures.
            python_utils.WeakWrapper(current_stack)
        )

        ret = "<table>\n"
        for frame in current_stack:
            fmod = inspect.getmodule(frame)
            if fmod is None:
                continue
            else:
                fmod = fmod.__name__
            ffunc = frame.f_code.co_name
            if not fmod.startswith("examples.") or fmod.startswith(
                "trulens_eval"
            ):
                continue

            ret += f"""
            <tr>
                <td>{fmod}</td>
                <td>{ffunc}</td>
            </tr>
            """

            # Don't include bytecode as it has non-deterministic addresses which mess with
            # golden test comparison.
            # bytecode = dis.Bytecode(frame.f_code)
            # <td><pre><code>{bytecode.dis()}</code></pre></td>

        ret += "</table>\n"

        return ret

    @staticmethod
    def clear_stack() -> None:
        """Clear the call stack"""
        DummyStackTool.last_stacks = []
