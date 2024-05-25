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
set_page_config(page_title= "A Closer Look at Hotspots 🔥")
st.title("A Closer Look at Hotspots 🔥")
st.caption("Hotspots are topics that score especially poorly on a certain feedback measure,  compared to the average score across all topics for that measure. In this case, if the failing percentage of a topic on a specific measure is one standard deviation greater than the average across all topics, that topic is considered a hotspot for that measure, as displayed in these tabs.")
st.runtime.legacy_caching.clear_cache()
#add_logo_and_style_overrides()

#retrieve tru database and get results of recording
tru = Tru()
lms = tru.db
#lms is an SQLalchemyDB
df_results, feedback_cols = lms.get_records_and_feedback([])


#find the list of topics of the records - which is stored in the metadata
topic_list = []
#category_list = []
for record_json in df_results['record_json']:
    loaded_record = json.loads(record_json) #load the json into a dictionary format
    metadata = loaded_record['meta']
    topic_list.append(metadata['Topic'])

topic_set = set(topic_list)
#thresholds taken from styles.py (for which higher feedback = better)
pass_threshold = 0.8
warning_threshold = 0.6
#tells you the overall number of records for this topic
overall_count = {key: 0 for key in topic_set}

#tells you the number of records for each topic with failing scores for the feedback functions
#groundedness - g
#answer relevance - ar
#context relevance - cr
fail_count_g = {key:0 for key in topic_set}
fail_count_ar = {key: 0 for key in topic_set}
fail_count_cr = {key: 0 for key in topic_set}

#tells you the percentage of records (compared to a topic's overall number of records) for each topic with a failing score
fail_percent_g = {}
fail_percent_ar = {}
fail_percent_cr = {}

#tells you the number of records with warning scores for the feedback functions
#groundedness - g
#answer relevance - ar
#context relevance - cr
warning_count_g = {key:0 for key in topic_set}
warning_count_ar = {key: 0 for key in topic_set}
warning_count_cr = {key: 0 for key in topic_set}

#similar to above
warning_percent_g = {}
warning_percent_ar = {}
warning_percent_cr = {}

#the questions, contexts, and answers for the queries of each topic that lead to failing scores for groundedness
#the key is the topic/the value is a list containing the questions/contexts/answers depending on the dictionary
bad_questions_g = {key: [] for key in topic_set}
bad_contexts_g = {key: [] for key in topic_set}
bad_answers_g = {key: [] for key in topic_set}
bad_reasons_g = {key: [] for key in topic_set}
bad_scores_g = {key: [] for key in topic_set}

#same as above, for answer relevance
bad_questions_ar = {key: [] for key in topic_set}
bad_contexts_ar = {key: [] for key in topic_set}
bad_answers_ar = {key: [] for key in topic_set}
bad_reasons_ar = {key: [] for key in topic_set}
bad_scores_ar = {key: [] for key in topic_set}

#same as above, for context relevance
bad_questions_cr = {key: [] for key in topic_set}
bad_contexts_cr = {key: [] for key in topic_set}
bad_answers_cr = {key: [] for key in topic_set}
bad_reasons_cr = {key: [] for key in topic_set}
bad_scores_cr = {key: [] for key in topic_set}



#each row is a record - find how many queries of each topic are failing or leading to warnings
#and for such queries, store the question being asked, the context being retrieved, and the answer being generated
for index, row in df_results.iterrows():
    record_json = row['record_json']
    loaded_record = json.loads(record_json)
    metadata = loaded_record['meta']
    topic = metadata['Topic']
    
    groundedness = row['Groundedness']
    answer_relevance = row['Answer Relevance']
    context_relevance = row['Context Relevance']
    
    g_call = row['Groundedness_calls']
    context = g_call[0]['args']['source']
    g_cot = g_call[0]['meta']['reasons']
    
    ar_call = row['Answer Relevance_calls']
    ar_cot = ar_call[0]['meta']['reason']
    
    cr_call = row['Context Relevance_calls']
    cr_cot = cr_call[0]['meta']['reason']
    
    question = row['input']
    #context = loaded_record['calls'][0]['args']['context_str']
    answer = row['output']
    
    overall_count[topic] += 1
    
    if groundedness < warning_threshold:
        fail_count_g[topic] += 1
        bad_questions_g[topic].append(question)
        bad_contexts_g[topic].append(context)
        bad_answers_g[topic].append(answer)
        bad_reasons_g[topic].append(g_cot)
        bad_scores_g[topic].append(groundedness)
    if answer_relevance < warning_threshold:
        fail_count_ar[topic] += 1
        bad_questions_ar[topic].append(question)
        bad_contexts_ar[topic].append(context)
        bad_answers_ar[topic].append(answer)
        bad_reasons_ar[topic].append(ar_cot)
        bad_scores_ar[topic].append(answer_relevance)
    if context_relevance < warning_threshold:
        fail_count_cr[topic] += 1
        bad_questions_cr[topic].append(question)
        bad_contexts_cr[topic].append(context)
        bad_answers_cr[topic].append(answer)
        bad_reasons_cr[topic].append(cr_cot)
        bad_scores_cr[topic].append(context_relevance)
    if groundedness > warning_threshold and groundedness < pass_threshold:
        warning_count_g[topic] += 1
    if answer_relevance > warning_threshold and answer_relevance < pass_threshold:
        warning_count_ar[topic] += 1
    if context_relevance > warning_threshold and context_relevance < pass_threshold:
        warning_count_cr[topic] += 1
    
#finding the failing and warning percentages for each topic
for key in overall_count:
    fail_percent_g[key] = round(fail_count_g[key]/overall_count[key], 3)
    fail_percent_ar[key] = round(fail_count_ar[key]/overall_count[key], 3)
    fail_percent_cr[key] = round(fail_count_cr[key]/overall_count[key], 3)
    warning_percent_g[key] = round(warning_count_g[key]/overall_count[key], 3)
    warning_percent_ar[key] = round(warning_count_ar[key]/overall_count[key], 3)
    warning_percent_cr[key] = round(warning_count_cr[key]/overall_count[key], 3)
    
#tells us the average failing percentage across all topics for each feedback measure
avg_fail_percent_g = np.mean(list(fail_percent_g.values()))
avg_fail_percent_ar = np.mean(list(fail_percent_ar.values()))
avg_fail_percent_cr = np.mean(list(fail_percent_cr.values()))

#tells us the standard deviation of failing percentage across all topics for each feedback measure
std_dev_g = np.std(list(fail_percent_g.values()))
std_dev_ar = np.std(list(fail_percent_ar.values()))
std_dev_cr = np.std(list(fail_percent_cr.values()))

    
#hotspot determination
#display_tabs has keys corresponding to topics and values corresponding to the string to display on the actual tab in StreamLit
#metric_tabs has keys corresponding to topics and values corresponding to which feedback function makes the topic a hotspot
display_tabs = {}
metric_tabs = {}

#going through each topic and seeing if it's failing percentage for any of the feedback measures is at least one standard #deviation above the mean failing percentage for any of the feedback measures
#if it is, store in the list of hotspots to display - if a topic can be considered a hotspot in multiple categories, classify it #as a hotspot for the feedback measure the topic is failing at the highest rate (percentage)
for key in overall_count:
    max_percent = max(fail_percent_g[key], fail_percent_ar[key], fail_percent_cr[key])
    if fail_percent_g[key] > avg_fail_percent_g + std_dev_g:
        if key not in display_tabs or fail_percent_g == max_percent:
            display_tabs[key] = f"{key} - Groundedness: {round(fail_percent_g[key] * 100 , 3)}%"
            metric_tabs[key] = "Groundedness"
    if fail_percent_ar[key] > avg_fail_percent_ar + std_dev_ar:
        if key not in display_tabs or fail_percent_ar[key] == max_percent:
            display_tabs[key] = f"{key} - Answer Relevance: {round(fail_percent_ar[key] * 100, 3)}%"
            metric_tabs[key] = "Answer Relevance"
    if fail_percent_cr[key] > avg_fail_percent_cr + std_dev_cr:
        if key not in display_tabs or fail_percent_cr[key] == max_percent:
            display_tabs[key] = f"{key} - Context Relevance: {round(fail_percent_cr[key] * 100, 3)}%"
            metric_tabs[key] = "Context Relevance"
        

#create tabs with hotspots
#for each tab, display the feedback scores for the topic being displayed in the tab
#also display how many standard deviations away the feedback scores are from the average feedback scores across all topics
#finally at the end, display a dataframe showing some of the queries leading to failing scores for the topic being displayed and the metric that makes it a hotspot

sections = st.tabs(list(display_tabs.values()))
for topic, metric, section in zip(metric_tabs.keys(), metric_tabs.values(), sections):
    with section:
        st.subheader(f"The percentage of queries with failing scores on each feedback metric for {topic}")
        col1, col2, col3 = st.columns(3)
        
        col1.metric(label = "Groundedness", value = f"{round(fail_percent_g[topic] * 100, 3)}%")
        distance_from_mean_g = round((fail_percent_g[topic] - avg_fail_percent_g)/std_dev_g, 2)
        col1.caption(f"{distance_from_mean_g} standard deviations away from the average percentage of queries with failing scores on Groundedness across all topics")
        
        
        col2.metric(label = "Answer Relevance", value = f"{round(fail_percent_ar[topic] * 100, 3)}%")
        distance_from_mean_ar = round((fail_percent_ar[topic] - avg_fail_percent_ar)/std_dev_ar, 2)
        col2.caption(f"{distance_from_mean_ar} standard deviations away from the average percentage of queries with failing scores on Answer Relevance across all topics")
        
        col3.metric(label = "Context Relevance", value = f"{round(fail_percent_cr[topic] * 100, 3)}%")
        distance_from_mean_cr = round((fail_percent_cr[topic] - avg_fail_percent_cr)/std_dev_cr, 2)
        col3.caption(f"{distance_from_mean_cr} standard deviations away from the average percentage of queries with failing scores on Context Relevance across all topics")
        
        st.divider()
        
        st.subheader(f"{topic} queries that lead to failing feedback scores on {metric}")
        st.caption("Click on a row to see the full texts")
        
        if metric == "Groundedness":
            bad_queries_df_g = pd.DataFrame({'Question': bad_questions_g[topic], 
                                             'Context': bad_contexts_g[topic], 
                                             'Answer': bad_answers_g[topic], 
                                             'Score': bad_scores_g[topic], 
                                             'Reasoning': bad_reasons_g[topic]}, 
                                            index = None)
            
            
            gb = GridOptionsBuilder.from_dataframe(bad_queries_df_g)
            gb.configure_pagination()
            gb.configure_selection(selection_mode = "single", use_checkbox = False)
            go = gb.build()
            
            grid = AgGrid(bad_queries_df_g,
                          gridOptions = go,
                          update_mode = GridUpdateMode.SELECTION_CHANGED)
            
            selected_rows = grid["selected_rows"]
            if selected_rows:
                #only one row selected at a time hence index 0
                selected_row = selected_rows[0]
                with st.expander("Question", expanded = True):
                    st.caption(selected_row['Question'])
                with st.expander("Context", expanded = True):
                    st.caption(selected_row['Context'])
                with st.expander("Answer", expanded = True):
                    st.caption(selected_row['Answer'])
                with st.expander(f"Reasoning behind {metric} score of {round(selected_row['Score'], 2)}", expanded = True): 
                    st.caption(selected_row['Reasoning'])
                
        if metric == "Answer Relevance":
            bad_queries_df_ar = pd.DataFrame({'Question': bad_questions_ar[topic], 
                                              'Context': bad_contexts_ar[topic], 
                                              'Answer': bad_answers_ar[topic], 
                                              'Score': bad_scores_ar[topic], 
                                              'Reasoning': bad_reasons_ar[topic]}, 
                                             index = None)
            gb = GridOptionsBuilder.from_dataframe(bad_queries_df_ar)
            gb.configure_pagination()
            gb.configure_selection(selection_mode = "single", use_checkbox = False)
            go = gb.build()
            
            grid = AgGrid(bad_queries_df_ar,
                          gridOptions = go,
                          update_mode = GridUpdateMode.SELECTION_CHANGED)
            selected_rows = grid["selected_rows"]
            if selected_rows:
                #only one row selected at a time hence index 0
                selected_row = selected_rows[0]
                with st.expander("Question", expanded = True):
                    st.caption(selected_row['Question'])
                with st.expander("Context", expanded = True):
                    st.caption(selected_row['Context'])
                with st.expander("Answer", expanded = True):
                    st.caption(selected_row['Answer'])
                with st.expander(f"Reasoning behind {metric} score of {round(selected_row['Score'], 2)}", expanded = True):
                    st.caption(selected_row['Reasoning'])
                
        if metric == "Context Relevance":
            bad_queries_df_cr = pd.DataFrame({'Question': bad_questions_cr[topic], 
                                              'Context': bad_contexts_cr[topic], 
                                              'Answer': bad_answers_cr[topic], 
                                              'Score': bad_scores_cr[topic], 
                                              'Reasoning': bad_reasons_cr[topic]}, 
                                             index = None)
            
            gb = GridOptionsBuilder.from_dataframe(bad_queries_df_cr)
            gb.configure_pagination()
            gb.configure_selection(selection_mode = "single", use_checkbox = False)
            go = gb.build()
            
            grid = AgGrid(bad_queries_df_cr,
                          gridOptions = go,
                          update_mode = GridUpdateMode.SELECTION_CHANGED)
            selected_rows = grid["selected_rows"]
            if selected_rows:
                selected_row = selected_rows[0]
                with st.expander("Question", expanded = True):
                    st.caption(selected_row['Question'])
                with st.expander("Context", expanded = True):
                    st.caption(selected_row['Context'])
                with st.expander("Answer", expanded = True):
                    st.caption(selected_row['Answer'])
                with st.expander(f"Reasoning behind {metric} score of {round(selected_row['Score'], 2)}", expanded = True):
                    st.caption(selected_row['Reasoning'])