import os
import requests
import pandas as pd
import numpy as np
import joblib
import os
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
dotenv_path=r"D:\Desktop\Desktop\Data_science_course\Rag Based project\RAG BASED PROJECT\.env"
load_dotenv(dotenv_path)


def create_embeddings(text_file):
    r=requests.post("http://localhost:11434/api/embed",json=
    {
        "model":"bge-m3",
        "input":text_file
    }
    )
    embeddings=r.json()["embeddings"]
    return embeddings

def interface(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    }
    
    r = requests.post(url, json=payload)
    response = r.json()['response']
    return response
def inferance_gemini(prompt):
    print("Generating response using gemma-3-27b-it...")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("models/gemma-3-27b-it")
    response = model.generate_content(prompt)
    return response.text
  

while True:
  df = joblib.load("embeddings.joblib", mmap_mode=None)
  input_query=input("enter your query:")
  query_embedding=create_embeddings([input_query])[0]
  similarity=cosine_similarity(np.vstack(df['embeddings']),[query_embedding]).flatten()
  top_results=5
  max_index=similarity.argsort()[::-1][0:top_results]
  newdf=df.loc[max_index]
  prompt = f"""
    As a Python course assistant, format your answer PRECISELY:

    VIDEO SEGMENTS:
    {newdf[["number","name","start","end","text"]].to_json(orient="records")}

    USER QUESTION: {input_query}

    FORMAT REQUIREMENTS:
    1. Start with: "Query: [user query]"
    2. Then: "Response: [your answer]"
    3. In the response, use THIS STRUCTURE:

    [Topic from query] is taught in:

    *   **Video Number:** [number]
    *   **Video Name:** "[name]"
        *   **Segment 1:**
            *   **Timestamps:** [start] - [end]
            *   **Summary:** [Detailed 2-3 sentences explaining what is taught]
            *   **Duration:** [calculate: end - start] seconds
        *   **Segment 2:**
            *   **Timestamps:** [start] - [end]
            *   **Summary:** [Detailed 2-3 sentences]
            *   **Duration:** [end - start] seconds
        *   [More segments as needed...]

    4. Sort segments by timestamp order
    5. Calculate duration for each: end - start
    6. Make summaries EDUCATIONAL and DETAILED
    7. Use proper Markdown formatting with asterisks and indentation

    Now answer: {input_query}
    """
  answer=inferance_gemini(prompt)
  print(answer)
  choice=input("do you want to save this response? (yes/no):")
  if choice.lower()=='yes':
    file_name=input("enter the file name(in which you want to save response):")
    if os.path.exists(f"{file_name}.txt"):
      print("file already exists, please choose a different name.")
    else:
      with open(f"{file_name}.txt","w") as f:
          f.write(f"Query: {input_query}\n")
          f.write(f"Response: {answer}\n")
      print(f"Response saved to {file_name}.txt")
  breaking=input("do you want to continue? (yes/no):") 
  if breaking.lower()=='no':
      break   
      