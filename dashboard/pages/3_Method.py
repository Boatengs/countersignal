"""Method and assumptions."""

import streamlit as st
from common import ROOT, page_setup

page_setup("Method")
st.markdown((ROOT / "docs" / "method.md").read_text())
