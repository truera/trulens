import asyncio
from pprint import PrettyPrinter
from threading import Thread
from typing import Callable, List, Mapping, Optional, Sequence, Union

from trulens_eval import app as mod_app
from trulens_eval.instruments import Instrument
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_NOTEBOOK
from trulens_eval.utils.json import JSON_BASES
from trulens_eval.utils.json import jsonify_for_ui
from trulens_eval.utils.serial import Lens

with OptionalImports(messages=REQUIREMENT_NOTEBOOK) as opt:
    # Here just for the assertion below. Including in a seperate context because
    # auto import organizer might move it below another import and if that other
    # import fails, this name will not be defined to check the assertion below.

    # TODO: The optional imports system should not allow this to happen, figure
    # out what is going wrong.
    import ipywidgets
    from ipywidgets import widgets

with opt:
    import traitlets
    from traitlets import HasTraits
    from traitlets import Unicode

opt.assert_installed(ipywidgets).assert_installed(traitlets)

pp = PrettyPrinter()

debug_style = dict(border="0px solid gray", padding="0px")

VALUE_MAX_CHARS = 1024


class Selector(HasTraits):
    select = Unicode()
    jpath = traitlets.Any()

    def __init__(self, select: Union[Lens, str], make_on_delete: Callable):
        if isinstance(select, Lens):
            self.select = str(select)
            self.jpath = select
        else:
            self.select = select
            self.jpath = Lens.of_string(select)

        self.w_edit = widgets.Text(value=select, layout=debug_style)
        self.w_delete = widgets.Button(
            description="x", layout=dict(width="30px", **debug_style)
        )

        self.on_delete = make_on_delete(self)
        self.w_delete.on_click(self.on_delete)

        traitlets.link((self.w_edit, "value"), (self, "select"))

        def on_update_select(ev):
            try:
                jpath = Lens.of_string(ev.new)
                self.jpath = jpath
                self.w_edit.layout.border = "0px solid black"
            except Exception:
                self.w_edit.layout.border = "1px solid red"

        self.observe(on_update_select, ["select"])

        self.w = widgets.HBox([self.w_delete, self.w_edit], layout=debug_style)


class SelectorValue(HasTraits):
    selector = traitlets.Any()
    obj = traitlets.Any()

    def __init__(
        self, selector: Selector, stdout_display: widgets.Output,
        instrument: Instrument
    ):
        self.selector = selector
        self.obj = None

        self.stdout_display = stdout_display

        self.w_listing = widgets.HTML(layout=debug_style)
        self.w = widgets.VBox(
            [self.selector.w, self.w_listing], layout=debug_style
        )

        self.selector.observe(self.update_selector, "jpath")
        self.observe(self.update_obj, "obj")

        self.instrument = instrument

    def _jsonify(self, obj):
        return jsonify_for_ui(obj=obj, instrument=self.instrument)

    def update_selector(self, ev):
        self.update()

    def update_obj(self, ev):
        self.update()

    def update(self):
        obj = self.obj
        jpath = self.selector.jpath

        inner_obj = None
        inner_class = None

        if obj is None:
            ret_html = "no listing yet"
        else:
            with self.stdout_display:
                try:
                    ret_html = ""

                    for inner_obj in jpath.get(obj):
                        inner_class = type(inner_obj)
                        inner_obj_id = id(inner_obj)
                        inner_obj = self._jsonify(inner_obj)

                        ret_html += f"<div>({inner_class.__name__} at 0x{inner_obj_id:x}): "  # as {type(inner_obj).__name__}): "

                        # if isinstance(inner_obj, pydantic.BaseModel):
                        #    inner_obj = inner_obj.model_dump()

                        if isinstance(inner_obj, JSON_BASES):
                            ret_html += str(inner_obj)[0:VALUE_MAX_CHARS]

                        elif isinstance(inner_obj, Mapping):
                            ret_html += "<ul>"
                            for key, val in inner_obj.items():
                                ret_html += f"<li>{key} = {str(val)[0:VALUE_MAX_CHARS]}</li>"
                            ret_html += "</ul>"

                        elif isinstance(inner_obj, Sequence):
                            ret_html += "<ul>"
                            for i, val in enumerate(inner_obj):
                                ret_html += f"<li>[{i}] = {str(val)[0:VALUE_MAX_CHARS]}</li>"
                            ret_html += "</ul>"

                        else:
                            ret_html += str(inner_obj)[0:VALUE_MAX_CHARS]

                        ret_html += "</div>"

                except Exception as e:
                    self.w_listing.layout.border = "1px solid red"
                    return

        self.w_listing.layout.border = "0px solid black"
        self.w_listing.value = f"<div>{ret_html}</div>"


class RecordWidget():

    def __init__(
        self,
        record_selections,
        instrument: Instrument,
        record=None,
        human_or_input=None,
        stdout_display: widgets.Output = None
    ):
        self.record = record
        self.record_selections = record_selections
        self.record_values = dict()

        self.human_or_input = widgets.HBox([human_or_input], layout=debug_style)
        self.w_human = widgets.HBox(
            [widgets.HTML("<b>human:</b>"), self.human_or_input],
            layout=debug_style
        )
        self.d_comp = widgets.HTML(layout=debug_style)
        self.d_extras = widgets.VBox(layout=debug_style)

        self.stdout_display = stdout_display

        self.human = ""
        self.comp = ""

        self.instrument = instrument

        self.d = widgets.VBox(
            [self.w_human, self.d_comp, self.d_extras],
            layout={
                **debug_style, "border": "5px solid #aaaaaa"
            }
        )

    def update_selections(self):
        # change to trait observe
        for s in self.record_selections:
            if s not in self.record_values:
                sv = SelectorValue(
                    selector=s,
                    stdout_display=self.stdout_display,
                    instrument=self.instrument
                )
                self.record_values[s] = sv
                self.d_extras.children += (sv.w,)

            if self.record is not None:
                record_filled = self.record.layout_calls_as_app()
            else:
                record_filled = None

            self.record_values[s].obj = record_filled

    def remove_selector(self, selector: Selector):
        if selector not in self.record_values:
            return

        item = self.record_values[selector]
        del self.record_values[selector]
        new_children = list(self.d_extras.children)
        new_children.remove(item.w)
        self.d_extras.children = tuple(new_children)

    def set_human(self, human: str):
        self.human = human
        self.human_or_input.children = (
            widgets.HTML(f"<div>{human}</div>", layout=debug_style),
        )

    def set_comp(self, comp: str):
        self.comp = comp
        self.d_comp.value = f"<div><b>computer:</b> {comp}</div>"


class AppUI(traitlets.HasTraits):
    # very prototype

    def __init__(
        self,
        app: mod_app.App,
        use_async: bool = False,
        app_selectors: Optional[List[Union[str, Lens]]] = None,
        record_selectors: Optional[List[Union[str, Lens]]] = None
    ):
        self.use_async = use_async

        self.app = app

        self.main_input = widgets.Text(layout=debug_style)
        self.app_selector = widgets.Text(layout=debug_style)
        self.record_selector = widgets.Text(layout=debug_style)

        self.main_input_button = widgets.Button(
            description="+ Record", layout=debug_style
        )
        self.app_selector_button = widgets.Button(
            description="+ Select.App", layout=debug_style
        )
        self.record_selector_button = widgets.Button(
            description="+ Select.Record", layout=debug_style
        )

        self.display_top = widgets.VBox([], layout=debug_style)
        self.display_side = widgets.VBox(
            [], layout={
                'width': "50%",
                **debug_style
            }
        )

        self.display_stdout = widgets.Output()

        self.display_records = []

        self.app_selections = {}
        self.record_selections = []

        self.current_record = RecordWidget(
            record_selections=self.record_selections,
            human_or_input=self.main_input,
            stdout_display=self.display_stdout,
            instrument=self.app.instrument
        )
        self.current_record_record = None

        self.records = [self.current_record]

        self.main_input.on_submit(self.add_record)
        self.app_selector.on_submit(self.add_app_selection)
        self.record_selector.on_submit(self.add_record_selection)

        self.main_input_button.on_click(self.add_record)
        self.app_selector_button.on_click(self.add_app_selection)
        self.record_selector_button.on_click(self.add_record_selection)

        outputs_widget = widgets.Accordion(children=[self.display_stdout])
        outputs_widget.set_title(0, 'stdpipes')

        self.display_bottom = widgets.VBox(
            [
                widgets.HBox(
                    [self.main_input_button, self.main_input],
                    layout=debug_style
                ),
                widgets.HBox(
                    [self.app_selector_button, self.app_selector],
                    layout=debug_style
                ),
                widgets.HBox(
                    [self.record_selector_button, self.record_selector],
                    layout=debug_style
                ),
            ],
            layout=debug_style
        )

        self.display_top.children += (self.current_record.d,)

        self.widget = widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.VBox(
                            [self.display_top, self.display_bottom],
                            layout={
                                **debug_style, 'width': '50%'
                            }
                        ), self.display_side
                    ],
                    layout=debug_style
                ), outputs_widget
            ]
        )

        if app_selectors is not None:
            for selector in app_selectors:
                self._add_app_selector(selector)

        if record_selectors is not None:
            for selector in record_selectors:
                self._add_record_selector(selector)

    def make_on_delete_record_selector(self, selector):

        def on_delete(ev):
            self.record_selections.remove(selector)

            for r in self.records:
                r.remove_selector(selector)

        return on_delete

    def make_on_delete_app_selector(self, selector):

        def on_delete(ev):
            sw = self.app_selections[selector]
            del self.app_selections[selector]

            new_children = list(self.display_side.children)
            new_children.remove(sw.w)

            self.display_side.children = tuple(new_children)

        return on_delete

    def update_app_selections(self):
        for _, sw in self.app_selections.items():
            sw.update()

    def _add_app_selector(self, selector: Union[Lens, str]):
        with self.display_stdout:
            sel = Selector(
                select=selector,
                make_on_delete=self.make_on_delete_app_selector
            )

        sw = SelectorValue(
            selector=sel,
            stdout_display=self.display_stdout,
            instrument=self.app.instrument
        )
        self.app_selections[sel] = sw
        sw.obj = self.app

        self.display_side.children += (sw.w,)

    def add_app_selection(self, w):
        self._add_app_selector(self.app_selector.value)

    def _add_record_selector(self, selector: Union[Lens, str]):
        with self.display_stdout:
            sel = Selector(
                select=selector,
                make_on_delete=self.make_on_delete_record_selector
            )

        self.record_selections.append(sel)

        for r in self.records:
            r.update_selections()

    def add_record_selection(self, w):
        s = self.record_selector.value

        self._add_record_selector(s)

    def add_record(self, w):
        human = self.main_input.value

        if len(human) == 0:
            return

        self.current_record.set_human(human)

        with self.app as recording:
            # generalize
            if self.use_async:
                self.current_record.set_comp("generating:")

                comp = ""

                def run_in_thread(comp):

                    async def run_in_main_loop(comp):
                        comp_generator = await self.app.main_acall(human)
                        async for tok in comp_generator:
                            comp += tok
                            self.current_record.set_comp(comp)

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(
                        asyncio.Task(run_in_main_loop(comp))
                    )

                t = Thread(target=run_in_thread, args=(comp,))
                t.start()
                t.join()

            else:
                with self.display_stdout:
                    self.current_record.set_comp("...")
                    comp = self.app.main_call(human)
                    self.current_record.set_comp(comp)

        self.current_record_record = recording.get()
        self.current_record.record = self.current_record_record
        self.current_record.update_selections()

        self.update_app_selections()

        self.current_record = RecordWidget(
            record_selections=self.record_selections,
            human_or_input=self.main_input,
            stdout_display=self.display_stdout,
            instrument=self.app.instrument
        )
        self.records.append(self.current_record)
        self.display_top.children += (self.current_record.d,)
