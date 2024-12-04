import streamlit as st
import os
import json

# Helper function to load and display text from files (JSON or TXT)
def load_and_display_file(file_path):
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Assuming the JSON data is a dictionary and you want to display all values
        text = "\n".join(data.values())
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            text = f.read()  # Read the entire content of the text file
    else:
        text = "Unsupported file format"
    return text

# Display content from each file
st.title("HR Analysis Results 📊")
st.subheader("----------------------------------------------------------------")
# Load and display content from finalSuggestion.json
st.subheader("HR Suggestions")
final_suggestion_text = load_and_display_file('data/finalSuggestion.txt')
st.write(final_suggestion_text)

# Load and display content from finalSummary.json
st.subheader("----------------------------------------------------------------")
st.subheader("Analisys Summary")
final_summary_text = load_and_display_file('data/finalSummary.txt')
st.write(final_summary_text)

# Load and display content from finalTopicPeriod.json
st.subheader("----------------------------------------------------------------")
st.subheader("Topic Period")
final_topic_period_text = load_and_display_file('data/finalTopicPeriod.txt')
st.write(final_topic_period_text)

