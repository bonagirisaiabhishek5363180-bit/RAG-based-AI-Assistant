import os
import requests
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
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
  api_key="AIzaSyA7V-N_-T3Q7x_sw6s2weLUHfXOIm1YUr4"
  genai.configure(api_key=api_key)
  model=genai.GenerativeModel("gemini-2.5-flash")
  response=model.generate_content(prompt)
  return response.text


df = joblib.load("embeddings.joblib", mmap_mode=None)
input_query=input("enter your query:")
query_embedding=create_embeddings([input_query])[0]
similarity=cosine_similarity(np.vstack(df['embeddings']),[query_embedding]).flatten()
top_results=5
max_index=similarity.argsort()[::-1][0:top_results]
newdf=df.loc[max_index]
prompt = f"""
You are an assistant for an online Python course.
You have access to the following video chunks with metadata:
[Video Data]
{newdf[["number","name","start","end","text"]].to_json(orient="records")}
[Instruction]
- If the user query matches content in the videos:
  • Tell them which video number and name.  
  • Give exact start–end timestamps.  
  • Summarize what is taught in that segment.  
  • Mention how much content (duration = end - start) is covered.  
- If the query is unrelated to the course, reply strictly with:  
  "I can only answer course-related questions."  
- Do not ask any questions back to the user. Just give the answer.  

[User Query]
{input_query}

[Answer]
"""
answer=inferance_gemini(prompt)
print(answer)
