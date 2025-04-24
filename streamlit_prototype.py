import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt


# Sidebar title
st.sidebar.title("Anna is Best")

# Sidebar selectbox
page = st.sidebar.selectbox("Choose a page:", ["Home", "Data", "Settings"])

# Sidebar text input
user_name = st.sidebar.text_input("Enter your name:")

# Sidebar number input
age = st.sidebar.number_input("Enter your age:", min_value=0, max_value=120)

# Sidebar checkbox
show_data = st.sidebar.checkbox("Show data table")

st.sidebar.slider(
  'Label goes here',
  0,
  550,
  450,
  help='Help message goes here'
)

# Sidebar button
if st.sidebar.button("Submit"):
    st.sidebar.success(f"Hello {user_name}, you are {age} years old!")

# Display based on page selection
st.title(f"Anna is my favorite girl 1:)")


data = pd.DataFrame({
    'Category': ['IPC', 'Motion', 'I/O'],
    'Value': [45, 30, 25]
})

fig = px.treemap(
    data,
    path=['Category'],
    values='Value',
)

# Optional: control text inside boxes
fig.update_traces(
    textinfo='label+value+percent entry',
    textfont_size=25,
    texttemplate="%{label}<br>%{percentEntry:.1%}"
)

st.plotly_chart(fig)