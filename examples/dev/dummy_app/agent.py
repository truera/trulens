from typing import Any, Dict, Optional

from trulens.apps.custom import TruCustomApp
from trulens.apps.custom import instrument

from examples.dev.dummy_app.dummy import Dummy


class DummyAgent(Dummy):
    """Dummy agent implementation.

    This agent serves to demonstrate both the span type for agents but also the
    use of _TruLens-Eval_ internally in an app that is instrumented using
    _TruLens-Eval_.

    Args:
        app: The app to use internally.

        description: A description for the agent.

        use_app: Whether to use a custom app internally.
            This will result in a complex stack where a custom app calls an
            agent which itself calls an app and further agents. The agents
            inside the inner app do not include further apps to prevent an
            infinite structure.

        use_recorder: Whether to use a recorder internally to capture execution
            of itself.

        **kwargs: Dummy class arguments.
    """

    def __init__(
        self,
        *args,
        app: Any,
        description: Optional[str] = None,
        use_app: bool = False,
        use_recorder: bool = False,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(*args, **kwargs)

        self.use_app = use_app
        self.use_recorder = use_recorder
        self.description = description or "Custom Agent"
        self.app = app

        if self.use_app:
            self.tru_app = TruCustomApp(self.app, app_name=description)

        self.dummy_allocate()

    @instrument
    def invoke(self, data: str) -> str:
        """Invoke the dummy tool."""

        self.dummy_wait()

        # TODO: see prior note.

        if self.use_app:
            if self.use_recorder:
                with self.tru_app as recorder:
                    self.app.respond_to_query(query=data)
                return recorder.get().model_dump_json()
            else:
                return self.app.respond_to_query(query=data)
        else:
            return "Record placeholder"

    @instrument
    async def ainvoke(self, data: str) -> str:
        """Invoke the dummy tool."""

        await self.dummy_await()

        # TODO: see prior note.

        if self.use_app:
            if self.use_recorder:
                with self.tru_app as recorder:
                    await self.app.arespond_to_query(query=data)
                return recorder.get().model_dump_json()
            else:
                return await self.app.arespond_to_query(query=data)
        else:
            return "Record placeholder"
