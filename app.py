from ikepono.configuration import Configuration
from ikepono.reidentifier import Reidentifier
from pathlib import Path

import streamlit as st
import sys
import traceback

configPathStr = sys.argv[1] if len(sys.argv) > 1 else "tests/test_configuration.json"

st.title("Ikepono")

reidentifier = None
try:
    print("Loading Reidentifier config %s" % configPathStr)
    reidentifier = Reidentifier.for_training(Configuration(Path(configPathStr)))
except Exception as err:
    with st.container():
        st.error("An error occurred")
        # get stack trace from err
        st.code("%s\n%s" % (err, traceback.format_exc()))

st.text('TRAIN!')

if reidentifier:
    reidentifier.train()

st.text('Now what!')