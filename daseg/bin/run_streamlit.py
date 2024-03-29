#!/usr/bin/env python
# coding: utf-8
import pickle
from typing import List

import streamlit as st

from daseg import Call, DialogActCorpus

DATASET_PATHS = [
    '/Users/pzelasko/jhu/daseg/deps/mrda',
    '/Users/pzelasko/jhu/daseg/deps/swda/swda',
]
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""


@st.cache(allow_output_mutation=True)
def load_dataset(path, tagset: str = 'basic') -> DialogActCorpus:
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        return DialogActCorpus.from_path(path, tagset=tagset)


@st.cache(hash_funcs={DialogActCorpus: id}, allow_output_mutation=True)
def find_dialog_acts(dataset: DialogActCorpus, act: str, left_context: int, right_context: int) -> List[Call]:
    return list(
        dataset.search(dialog_act=act, left_segments_context=left_context, right_segments_context=right_context))


st.sidebar.title("Dialog Act Explorer")
st.sidebar.markdown("""Explore dialog acts appearing in context in real conversations.""")

tagset = st.sidebar.selectbox("Tag set:", ['basic', 'general', 'full', 'segmentation'])
dataset_path = st.sidebar.selectbox("Dataset path", DATASET_PATHS)
dataset_load_state = st.info(f"Loading dataset'{dataset_path}'...")
dataset = load_dataset(dataset_path, tagset=tagset)
dataset_load_state.empty()

st.sidebar.header("Dialog Acts")
left_context = st.sidebar.slider('Left context acts', 0, 20, 5)
right_context = st.sidebar.slider('Right context acts', 0, 20, 5)
label_set = dataset.dialog_acts
selected_act = st.sidebar.selectbox(
    "Select dialog act class to explore:",
    options=label_set,
    index=0
)

dialog_acts_with_context = find_dialog_acts(dataset, selected_act, left_context, right_context)

st.header(f"Viewing Dialog Act: {selected_act}")
example_index = st.slider("Select which example to display:", 0, len(dialog_acts_with_context), 0)
# example_index = st.selectbox("Select which example to display:", list(range(len(dialog_acts_with_context))))
selected_example = Call(dialog_acts_with_context[example_index])

htmls = selected_example.render(jupyter=False, tagset=label_set)
# Newlines seem to mess with the rendering
htmls = [html.replace("\n", " ") for html in htmls]
st.write(HTML_WRAPPER.format('\n'.join(html for html in htmls)), unsafe_allow_html=True)
