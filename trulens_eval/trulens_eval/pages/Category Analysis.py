import asyncio
import json
from typing import Dict, Iterable, Tuple

asyncio.set_event_loop(asyncio.new_event_loop())

from pprint import PrettyPrinter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from st_aggrid.shared import JsCode
import streamlit as st
#from ux.add_logo import add_logo_and_style_overrides
from ux.page_config import set_page_config
from ux.styles import CATEGORY

pp = PrettyPrinter()

from trulens_eval import Tru
from trulens_eval.app import Agent
from trulens_eval.app import ComponentView
from trulens_eval.app import instrumented_component_views
from trulens_eval.app import LLM
from trulens_eval.app import Other
from trulens_eval.app import Prompt
from trulens_eval.app import Tool
from trulens_eval.database.base import MULTI_CALL_NAME_DELIMITER
from trulens_eval.react_components.record_viewer import record_viewer
from trulens_eval.schema.record import Record
from trulens_eval.schema.feedback import Select
from trulens_eval.utils.json import jsonify_for_ui
from trulens_eval.utils.serial import Lens
from trulens_eval.ux.components import draw_agent_info
from trulens_eval.ux.components import draw_call
from trulens_eval.ux.components import draw_llm_info
from trulens_eval.ux.components import draw_metadata
from trulens_eval.ux.components import draw_prompt_info
from trulens_eval.ux.components import draw_tool_info
from trulens_eval.ux.components import render_selector_markdown
from trulens_eval.ux.components import write_or_json
from trulens_eval.ux.styles import cellstyle_jscode


#first method - set up page and title
set_page_config(page_title="Category Analysis")
st.title("Category Analysis")
st.runtime.legacy_caching.clear_cache()
#add_logo_and_style_overrides()

#retrieve tru database and get results of recording
tru = Tru()
lms = tru.db
#lms is an SQLalchemyDB
df_results, feedback_cols = lms.get_records_and_feedback([])


#find the list of topics and categories of the records - which is stored in the metadata
#categories - Informative, Evaluate, Generative
#topic_list = []
category_list = []
for record_json in df_results['record_json']:
    loaded_record = json.loads(record_json) #load the json into a dictionary format
    metadata = loaded_record['meta']
    #topic_list.append(metadata['Topic'])
    category_list.append(metadata['Category'])

category_set = set(category_list)
#thresholds taken from styles.py (for which higher feedback = better)
pass_threshold = 0.8
warning_threshold = 0.6
#tells you the overall number of records for each category
overall_count = {key: 0 for key in category_set}

#tells you the number of records for each category with failing scores for the feedback functions
#groundedness - g
#answer relevance - ar
#context relevance - cr
fail_count_g = {key:0 for key in category_set}
fail_count_ar = {key: 0 for key in category_set}
fail_count_cr = {key: 0 for key in category_set}

#tells you the percentage of records (compared to a category's overall number of records) for each category with a failing score
fail_percent_g = {}
fail_percent_ar = {}
fail_percent_cr = {}

#tells you the number of records for each category with warning scores for the feedback functions
#groundedness - g
#answer relevance - ar
#context relevance - cr
warning_count_g = {key:0 for key in category_set}
warning_count_ar = {key: 0 for key in category_set}
warning_count_cr = {key: 0 for key in category_set}

#similar to above
warning_percent_g = {}
warning_percent_ar = {}
warning_percent_cr = {}


#can maybe use iterrows
#each row is a record
for index, row in df_results.iterrows():
    record_json = row['record_json']
    loaded_record = json.loads(record_json)
    metadata = loaded_record['meta']
    category = metadata['Category']
    groundedness = row['Groundedness']
    answer_relevance = row['Answer Relevance']
    context_relevance = row['Context Relevance']
    overall_count[category] += 1
    if groundedness < warning_threshold:
        fail_count_g[category] += 1
    if answer_relevance < warning_threshold:
        fail_count_ar[category] += 1
    if context_relevance < warning_threshold:
        fail_count_cr[category] += 1
    if groundedness > warning_threshold and groundedness < pass_threshold:
        warning_count_g[category] += 1
    if answer_relevance > warning_threshold and answer_relevance < pass_threshold:
        warning_count_ar[category] += 1
    if context_relevance > warning_threshold and context_relevance < pass_threshold:
        warning_count_cr[category] += 1
    

for key in overall_count:
    fail_percent_g[key] = round(fail_count_g[key]/overall_count[key], 3)
    fail_percent_ar[key] = round(fail_count_ar[key]/overall_count[key], 3)
    fail_percent_cr[key] = round(fail_count_cr[key]/overall_count[key], 3)
    warning_percent_g[key] = round(warning_count_g[key]/overall_count[key], 3)
    warning_percent_ar[key] = round(warning_count_ar[key]/overall_count[key], 3)
    warning_percent_cr[key] = round(warning_count_cr[key]/overall_count[key], 3)
    
#Groundedness
st.subheader("% of queries by category with failing Groundedness scores 🔴")
for col, category, percentage in zip(st.columns(len(fail_percent_g)), fail_percent_g.keys(), fail_percent_g.values()):
    col.metric(label = category, value = f"{round(percentage * 100, 3)}%")

#finding the topics that have a % of failing Groundedness scores higher than one standard deviation away
#from the mean failing percentage of all topics
list_stddev_away_g = []
avg_fail_percent_g = np.mean(list(fail_percent_g.values()))
dev_fail_percent_g = np.std(list(fail_percent_g.values()))
upper_bound_g = avg_fail_percent_g + dev_fail_percent_g
for key in fail_percent_g:
    if fail_percent_g[key] >= upper_bound_g:
        list_stddev_away_g.append(key)
st.selectbox(label = f"Categories with a failing percentage on Groundedness one standard deviation above the mean failing percentage across all categories: {round(avg_fail_percent_g * 100, 3)}%",
             options = list_stddev_away_g)

st.subheader("% of queries by category with warning Groundedness scores ⚠️")
for col, category, percentage in zip(st.columns(len(warning_percent_g)), warning_percent_g.keys(), warning_percent_g.values()):
    col.metric(label = category, value = f"{round(percentage * 100, 3)}%")
    
st.divider()


#Answer Relevance
st.subheader("% of queries by category with failing Answer Relevance scores 🔴")
for col, category, percentage in zip(st.columns(len(fail_percent_ar)), fail_percent_ar.keys(), fail_percent_ar.values()):
    col.metric(label = category, value = f"{round(percentage * 100, 3)}%")

list_stddev_away_ar = []
avg_fail_percent_ar = np.mean(list(fail_percent_ar.values()))
dev_fail_percent_ar = np.std(list(fail_percent_ar.values()))
upper_bound_ar = avg_fail_percent_ar + dev_fail_percent_ar
for key in fail_percent_ar:
    if fail_percent_ar[key] > upper_bound_ar:
        list_stddev_away_ar.append(key)
st.selectbox(label = f"Categories with a failing percentage on Answer Relevance one standard deviation above the mean failing percentage across all categories: {round(avg_fail_percent_ar * 100, 3)}%",
             options = list_stddev_away_ar)

st.subheader("% of queries by category with warning Answer Relevance scores ⚠️")
for col, category, percentage in zip(st.columns(len(warning_percent_ar)), warning_percent_ar.keys(), warning_percent_ar.values()):
    col.metric(label = category, value = f"{round(percentage * 100, 3)}%")
    
st.divider()

#Context Relevance    
st.subheader("% of queries by category with failing Context Relevance scores 🔴")
for col, category, percentage in zip(st.columns(len(fail_percent_cr)), fail_percent_cr.keys(), fail_percent_cr.values()):
    col.metric(label = category, value = f"{round(percentage * 100, 3)}%")
    
list_stddev_away_cr = []
avg_fail_percent_cr = np.mean(list(fail_percent_cr.values()))
dev_fail_percent_cr = np.std(list(fail_percent_cr.values()))
upper_bound_cr = avg_fail_percent_cr + dev_fail_percent_cr
for key in fail_percent_cr:
    if fail_percent_cr[key] > upper_bound_cr:
        list_stddev_away_cr.append(key)
st.selectbox(label = f"Categories with a failing percentage on Context Relevance one standard deviation above the mean failing percentage across all categories: {round(avg_fail_percent_cr, 3)}%",
             options = list_stddev_away_cr)

st.subheader("% of queries by category with warning Context Relevance scores ⚠️")
for col, category, percentage in zip(st.columns(len(warning_percent_cr)), warning_percent_cr.keys(), warning_percent_cr.values()):
    col.metric(label = category, value = f"{round(percentage * 100, 3)}%")
    
st.divider()