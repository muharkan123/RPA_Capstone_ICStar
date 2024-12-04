from fastapi import FastAPI
from io import BytesIO
from fastapi.responses import StreamingResponse, HTMLResponse
import matplotlib.pyplot as plt
import google.generativeai as genai
import os

app = FastAPI()

# Set up Google Generative AI
API_KEY = 'AIzaSyDdMpg0E1PPVEnBI6nB0PzQv-WXwAwQeMA'
os.environ['my_api_key'] = API_KEY
genai.configure(api_key=API_KEY)

# Helper function to generate sentiment plot
def generate_sentiment_plot():
    sentiments = [120, 5, 1]  # Sample data: neutral, positive, negative
    labels = ['Neutral', 'Positive', 'Negative']
    colors = ['#1f77b4', '#2ca02c', '#d62728']

    plt.figure(figsize=(6, 6))
    plt.pie(sentiments, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Sentiment Analysis Summary')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return buf

# Helper function to generate HR recommendations
def generate_hr_recommendations():
    topic_summary = "Topic 1: communication, feedback, growth; Topic 2: salary, benefits, work-life balance"
    prompt = (
        "I am an HR doing analysis. Here is the sentiment and topic analysis results:\\n\\n"
        f"Topic Modeling Results:\\n{topic_summary}\\n\\n"
        "Based on this information, provide suggestions on how to improve employee satisfaction."
    )
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    
    return response.text

# Endpoint to display the chart and recommendations in a browser
@app.get("/display")
async def display_results():
    # Generate the chart
    buf = generate_sentiment_plot()

    # Get HR recommendations
    hr_recommendations = generate_hr_recommendations()

    # HTML content to display the plot and recommendations
    html_content = f"""
    <html>
        <body>
            <h1>Sentiment Analysis Summary</h1>
            <img src="/sentiment-chart" alt="Sentiment Chart" />
            <h2>HR Recommendations</h2>
            <p>{hr_recommendations}</p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Endpoint to serve the chart as an image
@app.get("/sentiment-chart")
async def sentiment_chart():
    buf = generate_sentiment_plot()
    return StreamingResponse(buf, media_type="image/png")

@app.get("/")
def root():
    return {"message": "Hello World"}