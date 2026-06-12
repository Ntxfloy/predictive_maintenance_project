import streamlit as st

# Configure Streamlit page configurations first
st.set_page_config(
    page_title="Предиктивное обслуживание оборудования",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup navigation structure
pages = [
    st.Page("analysis_and_model.py", title="Анализ и модель", icon="📊"),
    st.Page("presentation.py", title="Презентация проекта", icon="📝"),
]

# Run the navigation routing
current_page = st.navigation(pages, position="sidebar", expanded=True)
current_page.run()
