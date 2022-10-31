import streamlit as st
from multiapp import MultiApp

from apps import home  # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Classification", home.app)
# The main app
app.run()
