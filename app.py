import streamlit as st
from time import sleep
from stqdm import stqdm
import pandas as pd
from transformers import pipeline
import os
import spacy
import spacy_streamlit

# Disable symlink warnings for Hugging Face Hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Create a models directory to save the models
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Default models for each task
DEFAULT_SUMMARIZER_MODEL = os.path.join(MODELS_DIR, "summarizer")
DEFAULT_SENTIMENT_MODEL = os.path.join(MODELS_DIR, "sentiment")
DEFAULT_QUESTION_ANSWERING_MODEL = os.path.join(MODELS_DIR, "question-answering")
DEFAULT_TEXT_GENERATION_MODEL = os.path.join(MODELS_DIR, "text-generation")


def draw_all(key, plot=False):
    """Draws the initial information about the NLP Web App."""
    st.write(
        """
        # NLP 

        This Natural Language Processing Based Application can do some of the important tasks with the Text.

        This App is built using pretrained transformers which are capable of doing wonders with the Textual data.

        ```python
        # Key Features of this App.
        1. Advanced Text Summarizer
        2. Named Entity Recognition
        3. Sentiment Analysis
        4. Question Answering
        5. Text Completion
        ```
        """
    )

with st.sidebar:
    draw_all("sidebar")
    
    
def main():
    """Main function to run the Streamlit app."""
    st.title("🌐 NLP ")
    st.markdown("""
        <style>
            .sidebar .selectbox { font-size: 18px; color: #4CAF50; }
            .stButton>button { font-size: 18px; background-color: #4CAF50; color: white; }
        </style>
        """, unsafe_allow_html=True)

    st.write("👋 Welcome! Explore the powerful capabilities of Natural Language Processing (NLP) with our interactive tools.\n🔄  Select a task from the sidebar!")
    menu = ["--Select--", "Summarizer 📝", "Named Entity Recognition 🔍", "Sentiment Analysis 😊", "Question Answering ❓", "Text Completion ✍️"]
    choice = st.sidebar.selectbox("👉 **Select a Task:**", menu)

    if choice == "--Select--":
        display_intro()
    elif choice == "Summarizer 📝":
        st.write("### 📝 **Summarizer**")
        st.write("Provide a long piece of text, and watch it condense into a concise summary.")
        text_summarization()
    elif choice == "Named Entity Recognition 🔍":
        st.write("### 🔍 **Named Entity Recognition**")
        st.write("Discover and categorize entities in your text, like names, places, and organizations.")
        named_entity_recognition()
    elif choice == "Sentiment Analysis 😊":
        st.write("### 😊 **Sentiment Analysis**")
        st.write("Analyze the emotional tone of your text to understand underlying sentiments.")
        sentiment_analysis()
    elif choice == "Question Answering ❓":
        st.write("### ❓ **Question Answering**")
        st.write("Ask a question about a text passage, and let the app provide an accurate answer.")
        question_answering()
    elif choice == "Text Completion ✍️":
        st.write("### ✍️ **Text Completion**")
        st.write("Start a sentence, and let the AI predict how it might continue.")
        text_completion()

    st.sidebar.write("💡 **Tip:** Select any option above to try out each feature!")
    st.sidebar.write("📘 **Use Cases:** This app is designed to showcase NLP capabilities with tasks relevant to real-world applications, from text analysis to content creation.")

    st.write("---")
   

def display_intro():
    """Displays introductory information about the app."""
    st.image('nlp_image.jpeg')

    st.write("""
             This is a Natural Language Processing Based Web App that can do anything you can imagine with the Text.
             """)

    st.write("""
             Natural Language Processing (NLP) is a computational technique to understand human language in the way they are spoken and written.
             """)

    st.write("""
             NLP is a subfield of Artificial Intelligence (AI) that aims to understand the context of text just like humans.
             """)

    st.write("""
             **Key Aspects of NLP:**
             
             1. **Tokenization:** Breaking down text into smaller, manageable parts, often words or subwords, allowing models to understand the sentence structure.
             
             2. **Syntactic Parsing:** Analyzing sentence structure to identify grammatical relationships, which is essential for applications like translation and question answering.
             
             3. **Semantic Analysis:** Understanding the meaning behind words and phrases to capture the context, crucial for resolving ambiguities.
             
             4. **Sentiment Analysis:** Gauging the emotional tone of text to provide insights into opinions, trends, and feedback.
             
             5. **Machine Translation:** Converting text from one language to another by considering linguistic rules and context.
             
             6. **Named Entity Recognition (NER):** Identifying and classifying proper names, dates, locations, and other significant entities within text.
             """)

    st.write("""
             **How NLP Works:**
             
             NLP leverages both traditional linguistic methods and deep learning models. Modern NLP techniques use large datasets to train models on human language patterns. These models, such as transformers, handle complex language tasks like text generation, translation, and summarization.
             """)

    st.write("""
             **Applications of NLP in the Real World:**
             
             NLP powers search engines, voice assistants, chatbots, social media platforms, and e-commerce. By processing language effectively, these systems provide personalized recommendations, automate responses, and translate languages with accuracy.
             """)

def text_summarization():
    """Handles text summarization functionality."""
    st.subheader("Text Summarization")
    st.write("Enter the Text you want to summarize!")

    raw_text = st.text_area("Your Text", "Enter Your Text Here")
    num_words = st.number_input("Enter Number of Words in Summary")

    if raw_text != "" and num_words is not None:
        num_words = int(num_words)
        try:
            summarizer = pipeline('summarization', model=DEFAULT_SUMMARIZER_MODEL)
        except:
            summarizer = pipeline('summarization', model="sshleifer/distilbart-cnn-12-6")
            summarizer.save_pretrained(DEFAULT_SUMMARIZER_MODEL)
        summary = summarizer(raw_text, min_length=num_words, max_length=50)
        result_summary = summary[0]['summary_text']
        result_summary = '. '.join(map(lambda x: x.strip().capitalize(), result_summary.split('.')))
        st.write(f"Here's your Summary: {result_summary}")

def named_entity_recognition():
    """Handles named entity recognition functionality."""
    nlp = spacy.load("en_core_web_sm")
    st.subheader("Text Based Named Entity Recognition")
    st.write("Enter the Text below to extract Named Entities!")

    raw_text = st.text_area("Your Text", "Enter Text Here")
    if raw_text != "Enter Text Here":
        doc = nlp(raw_text)
        for _ in stqdm(range(50), desc="Please wait a bit. The model is fetching the results !!"):
            sleep(0.1)
        spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels, title="List of Entities")

def sentiment_analysis():
    """Handles sentiment analysis functionality."""
    st.subheader("Sentiment Analysis")
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model=DEFAULT_SENTIMENT_MODEL)
    except:
        sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        sentiment_pipeline.save_pretrained(DEFAULT_SENTIMENT_MODEL)

    st.write("Enter the Text below to find out its Sentiment!")

    raw_text = st.text_area("Your Text", "Enter Text Here")

    if raw_text != "Enter Text Here":
        result = sentiment_pipeline(raw_text)[0]
        sentiment = result['label']

        for _ in stqdm(range(50), desc="Please wait a bit. The model is fetching the results !!"):
            sleep(0.1)

        if sentiment == "POSITIVE":
            st.write("""# This text has a Positive Sentiment. 🤗""")
        elif sentiment == "NEGATIVE":
            st.write("""# This text has a Negative Sentiment. 😤""")
        else:
            st.write("""# This text seems Neutral ... 😐""")

def question_answering():
    """Handles question answering functionality."""
    st.subheader("Question Answering")
    st.write("Enter the Context and ask the Question to find out the Answer!")

    try:
        question_pipeline = pipeline("question-answering", model=DEFAULT_QUESTION_ANSWERING_MODEL)
    except:
        question_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        question_pipeline.save_pretrained(DEFAULT_QUESTION_ANSWERING_MODEL)

    context = st.text_area("Context", "Enter the Context Here")

    question = st.text_area("Your Question", "Enter your Question Here")

    if context != "Enter Text Here" and question != "Enter your Question Here":
        result = question_pipeline(question=question, context=context)
        generated_text = result['answer']
        generated_text = '. '.join(map(lambda x: x.strip().capitalize(), generated_text.split('.')))

        st.write(f"Here's your Answer:\n {generated_text}")

def text_completion():
    """Handles text completion functionality."""
    st.subheader("Text Completion")
    st.write("Enter the incomplete Text to complete it automatically using AI!")

    try:
        text_generation_pipeline = pipeline("text-generation", model=DEFAULT_TEXT_GENERATION_MODEL)
    except:
        text_generation_pipeline = pipeline("text-generation", model="gpt2")
        text_generation_pipeline.save_pretrained(DEFAULT_TEXT_GENERATION_MODEL)

    message = st.text_area("Your Text", "Enter the Text to complete")

    if message != "Enter the Text to complete":
        generator = text_generation_pipeline(message)

        generated_text = generator[0]['generated_text']
        generated_text = '. '.join(map(lambda x: x.strip().capitalize(), generated_text.split('.')))

        st.write(f"Here's your Generated Text:\n   {generated_text}")

if __name__ == '__main__':
    main()
    
    
    
st.markdown(
    """
    <style>
        .footer {
            left: 0;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            border-top: 1px solid #e5e5e5;
        }
    </style>
    <div class='footer'>
        <strong>NLP FA-2</strong><br>
        Members: Prathamesh Mandge, Pranav Narkhede, Nilesh Morkar, Akash Palve
    </div>
    """,
    unsafe_allow_html=True
)
