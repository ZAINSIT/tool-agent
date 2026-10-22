import streamlit as st
from agent_loop import run_agent

st.title("Tool-Calling Agent")
st.caption("Ask a question. The agent decides whether to search local docs or the web.")

query = st.text_input("Your question:")
if st.button("Run") and query:
    with st.spinner("Agent thinking..."):
        answer = run_agent(query, logger=st.write)
    st.markdown("### Answer")
    st.write(answer)