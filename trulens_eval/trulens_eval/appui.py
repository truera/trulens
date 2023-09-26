import asyncio
from pprint import PrettyPrinter
from threading import Thread
from typing import Callable, Mapping, Sequence

from ipywidgets import widgets
import traitlets
from traitlets import HasTraits
from traitlets import Unicode

from trulens_eval.app import App
from trulens_eval.utils.json import JSON_BASES
from trulens_eval.utils.json import jsonify
from trulens_eval.utils.serial import JSONPath

pp = PrettyPrinter()

debug_style = dict(border="0px solid gray", padding="0px")

class Selector(HasTraits):
    select = Unicode()
    jpath = traitlets.Any()

    def __init__(self, select: str, make_on_delete: Callable):
        self.select = select
        self.jpath = JSONPath.of_string(select)

        self.w_edit = widgets.Text(value=select, layout=debug_style)
        self.w_delete = widgets.Button(description="x", layout=dict(width="30px", **debug_style))

        self.on_delete = make_on_delete(self)
        self.w_delete.on_click(self.on_delete)

        traitlets.link((self.w_edit, "value"), (self, "select"))

        def on_update_select(ev):
            try:
                jpath = JSONPath.of_string(ev.new)
                self.jpath = jpath
                self.w_edit.layout.border = "0px solid black"
            except Exception:
                self.w_edit.layout.border = "1px solid red"

        self.observe(on_update_select, ["select"])

        self.w = widgets.HBox([self.w_delete, self.w_edit], layout=debug_style)

    
class SelectorValue(HasTraits):
    selector = traitlets.Any()
    obj = traitlets.Any()

    def __init__(self, selector: Selector):
        self.selector = selector
        self.obj = None

        self.w_listing = widgets.HTML(layout=debug_style)
        self.w = widgets.VBox([self.selector.w, self.w_listing], layout=debug_style)

        self.selector.observe(self.update_selector, "jpath")
        self.observe(self.update_obj, "obj")

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
            try:
                ret_html = ""

                for inner_obj in jpath(obj):
                    inner_class = type(inner_obj)
                    inner_obj = jsonify(inner_obj)

                    ret_html += f"<div>({inner_class.__name__} as {type(inner_obj).__name__}): "

                    # if isinstance(inner_obj, pydantic.BaseModel):
                    #    inner_obj = inner_obj.dict()

                    if isinstance(inner_obj, JSON_BASES):
                        ret_html += str(inner_obj)[0:256]

                    elif isinstance(inner_obj, Mapping):
                        ret_html += "<ul>"
                        for key, val in inner_obj.items():
                            ret_html += f"<li>{key} = {str(val)[0:256]}</li>"
                        ret_html += "</ul>"

                    elif isinstance(inner_obj, Sequence):
                        ret_html += "<ul>"
                        for i, val in enumerate(inner_obj):
                            ret_html += f"<li>[{i}] = {str(val)[0:256]}</li>"
                        ret_html += "</ul>"                    

                    else:
                        ret_html += str(inner_obj)[0:256]
                    
                    ret_html += "</div>"

            except Exception as e:
                self.w_listing.layout.border = "1px solid red"
                return

        self.w_listing.layout.border = "0px solid black"
        self.w_listing.value = f"<div>{ret_html}</div>"


class RecordWiget():
    def __init__(self, record_selections, record=None, human_or_input=None):
        self.record = record
        self.record_selections = record_selections
        self.record_values = dict()

        self.human_or_input = widgets.HBox([human_or_input], layout=debug_style)
        self.w_human = widgets.HBox([widgets.HTML("<b>human:</b>"), self.human_or_input], layout=debug_style)
        self.d_comp = widgets.HTML(layout=debug_style)
        self.d_extras = widgets.VBox(layout=debug_style)

        self.human = ""
        self.comp = ""

        self.d = widgets.VBox([self.w_human, self.d_comp, self.d_extras], layout={**debug_style, "border": "5px solid #aaaaaa"})

    def update_selections(self):
        # change to trait observe
        for s in self.record_selections:
            if s not in self.record_values:
                sv = SelectorValue(selector=s)
                self.record_values[s] = sv
                self.d_extras.children += (sv.w, )

            if self.record is not None:
                record_filled = self.record.layout_calls_as_app()
            else:
                record_filled = None

            self.record_values[s].obj =  record_filled

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
        self.human_or_input.children = (widgets.HTML(f"<div>{human}</div>", layout=debug_style), )

    def set_comp(self, comp: str):
        self.comp = comp
        self.d_comp.value = f"<div><b>computer:</b> {comp}</div>"

class AppUI(traitlets.HasTraits):
    # very prototype

    def __init__(self, app: App, use_async: bool = False):
        self.use_async = use_async

        self.app = app

        self.main_input = widgets.Text(layout=debug_style)
        self.app_selector = widgets.Text(layout=debug_style)
        self.record_selector = widgets.Text(layout=debug_style)

        self.main_input_button = widgets.Button(description="+ Record", layout=debug_style)
        self.app_selector_button = widgets.Button(description="+ Select.App", layout=debug_style)
        self.record_selector_button = widgets.Button(description="+ Select.Record", layout=debug_style)

        self.display_top = widgets.VBox([], layout=debug_style)
        self.display_side = widgets.VBox([], layout={'min_width':"500px", **debug_style})

        self.display_records = []

        self.app_selections = {}
        self.record_selections = []

        self.current_record = RecordWiget(record_selections=self.record_selections, human_or_input=self.main_input)
        self.current_record_record = None

        self.records = [self.current_record]

        self.main_input.on_submit(self.add_record)
        self.app_selector.on_submit(self.add_app_selection)
        self.record_selector.on_submit(self.add_record_selection)

        self.main_input_button.on_click(self.add_record)
        self.app_selector_button.on_click(self.add_app_selection)
        self.record_selector_button.on_click(self.add_record_selection)

        self.display_bottom = widgets.VBox([
            widgets.HBox([self.main_input_button, self.main_input], layout=debug_style),
            widgets.HBox([self.app_selector_button, self.app_selector], layout=debug_style),
            widgets.HBox([self.record_selector_button, self.record_selector], layout=debug_style)],
            layout=debug_style
        )

        self.display_top.children += (self.current_record.d, )

        self.d = widgets.HBox([
                widgets.VBox([
                    self.display_top,
                    self.display_bottom
                ], layout=debug_style),
                self.display_side], layout=debug_style
            )

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

    def add_app_selection(self, w):
        s = self.app_selector.value

        sel = Selector(select=s, make_on_delete=self.make_on_delete_app_selector)

        sw = SelectorValue(selector=sel)
        self.app_selections[sel] = sw
        sw.obj = self.app

        self.display_side.children += (sw.w, )

    def add_record_selection(self, w):
        s = self.record_selector.value

        sel = Selector(select=s, make_on_delete=self.make_on_delete_record_selector)
        self.record_selections.append(sel)

        for r in self.records:
            r.update_selections()

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
                    loop.run_until_complete(asyncio.Task(run_in_main_loop(comp)))

                t = Thread(target=run_in_thread, args=(comp, ))
                t.start()
                t.join()

            else:
                self.current_record.set_comp("...")
                comp = self.app.main_call(human)
                self.current_record.set_comp(comp)

        self.current_record_record = recording.get()
        self.current_record.record = self.current_record_record
        self.current_record.update_selections()

        self.update_app_selections()

        self.current_record = RecordWiget(record_selections=self.record_selections, human_or_input=self.main_input)
        self.records.append(self.current_record)
        self.display_top.children += (self.current_record.d, )