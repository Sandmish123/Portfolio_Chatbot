from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Hugging Face API URL for distilBERT question answering model
API_URL = "https://api-inference.huggingface.co/models/distilbert/distilbert-base-uncased-distilled-squad"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

@app.route('/ask', methods=['POST'])
def ask():
    # Get the user's question from the request
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    # Resume content to use as context for the model
    resume_text = """
                                    Sandeep Mishra
                                sandeepmishra991056@gmail.com
                                www.linkedin.com/in/sandeep-mishra-298bb5219
                                9871832553
                                https://github.com/Sandmish123
                                Objective
                                I am a Software Engineer with 2 years of experience in Python development, machine learning, and deploying RL models on servers and hardware. I’ve trained RL models using Unreal Engine, worked on on-board computation, and explored cutting-edge technologies like natural language processing (NLP) and generative AI. Passionate about creating efficient, scalable solutions, I aim to leverage my expertise to drive impactful innovations in AI and software development.

                                Professional Experience
                                Software Engineer, Appolo Computers Pvt Ltd
                                •Leveraged Python to automate workflows, develop ML models, and enhance scalability.
                                •Worked on NLP projects using transformer models like BERT, GPT-3.5, and Mistral to deliver advanced language solutions.
                                03/2023 – present
                                New Delhi, India
                                •Trained and deployed reinforcement learning models on Unreal Engine and hardware for real-world applications.
                                •Designed object detection and avoidance systems, integrating them into autonomous workflows for improved navigation.
                                Projects
                                Searching and Summarizing Over Enterprise Data
                                Description: Developed a system using NLP and Generative AI models for efficient question-answering and document summarization, enabling enterprises to search, retrieve, and summarize information from large document repositories. Implemented advanced embedding, indexing, and summarization techniques using Elasticsearch and vector databases to improve accuracy and efficiency.

                                Technologies: Mistral Model, BART Model, Hugging Face, LangChain, Elasticsearch, Vector Stores, RAG, MinIO, SummarizeIQ

                                Number Recognition using MNIST Dataset
                                Description : A handwritten digit recognition system uses Convolutional Neural Networks (CNNs) to detect and classify handwritten digits, typically using the MNIST dataset of labeled images (0-9) in 28x28 grayscale format. CNNs learn to identify features like curves and edges, distinguishing variations in handwriting styles. Once trained, the system can recognize new digit images in real-time, making it ideal for applications such as postal code recognition, bank check processing, and digitized document analysis.

                                Technologies : Python ,Tensorflow, Keras , Numpy ,Pandas , MNIST Dataset.

                                Skills
                                Python
                                Gen AI
                                NLP
                                Retrieval Augmented Generation(RAG)
                                Elastic Search
                                Vector Database
                                Machine Learning & Deep Learning.
                                Linux
                                Certificates
                                Python Begineer to Advance
                                Oops in Python
                                Artificial Neural Networks (ANN) with Keras in Python
                                Applied Gen AI and NLP
                                Education
                                Bachelor of  Technology
                                JC Bose University of Science and Technology (YMCA)

                                8.95 CGPA

                                2019 – 2023
                                Faridabad, India
                                Intermediate, Balvantray Mehta Vidya Bhawan A.S.M.A
                                8.02 CGPA

                                2016 – 2018
                                New Delhi, India
    """
    
    # Call Hugging Face model to get the answer
    result = query({
        "inputs": {
            "question": question,
            "context": resume_text
        }
    })
    
    # Extract the answer from the result
    answer = result.get('answer', 'Sorry, I could not find an answer.')

    return jsonify({'answer': answer})

if __name__ == "__main__":
    app.run(debug=True)
