import streamlit as st
import plotly.graph_objects as go

if "selected" not in st.session_state:
    st.session_state.selected = None
if "key_counter" not in st.session_state:
    st.session_state.key_counter = 0

st.write("Selected:", st.session_state.selected)

fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[10, 11, 12], mode="markers+lines", marker=dict(size=10)))

if st.session_state.selected is not None:
    fig.add_trace(go.Scatter(
        x=[st.session_state.selected['x']], 
        y=[st.session_state.selected['y']], 
        mode="markers", 
        marker=dict(size=20, color="red")
    ))

event = st.plotly_chart(
    fig,
    on_select="rerun",
    selection_mode=["points", "box"],
    key=f"chart_{st.session_state.key_counter}"
)

if event and "selection" in event:
    pts = event["selection"].get("points", [])
    if pts:
        st.session_state.selected = pts[-1]
        st.session_state.key_counter += 1
        st.rerun()
