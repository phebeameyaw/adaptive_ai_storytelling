import streamlit as st

st.set_page_config(page_title="AI Storytelling Dashboard", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #ffe6f0;
    }
    .stButton>button {
        background-color: #ff4b9b;
        color: white;
        border-radius: 10px;
        padding: 0.75em 1.5em;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff79b0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎀📚💖 Adaptive AI Storytelling Dashboard")
st.markdown("Welcome! Explore the experiments below to test different storytelling techniques.")

# Sidebar
with st.sidebar:
    st.header("🧭 Navigation")
    st.markdown("Choose an experiment from the main view or sidebar below.")
    st.markdown("- 📄 [Documentation](https://your-docs-link.com)")  #placeholder 
    st.markdown("- 💬 [Feedback Form](https://your-form-link.com)")  #placeholder
    st.markdown("---")
    st.caption("Developed by Phebe Bonuah Ameyaw")

# Main buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Experiment 1"):
        st.toast("Loading Experiment 1...")
        st.switch_page("pages/experiment1.py")
    st.caption("RL fine-tuning impact test")

with col2:
    if st.button("Experiment 2"):
        st.toast("Loading Experiment 2...")
        st.switch_page("pages/experiment2.py")
    st.caption("Structured  vs open ended prompt testing")
   


st.markdown("---")
st.info("Experiments test different generation strategies using the Mistral-7B model and user feedback.")
