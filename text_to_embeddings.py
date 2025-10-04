import json
import os
import requests
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib
import numpy as np
# jsons_file=input("enter the json files folder path:")
def create_embeddings(text_file):
    r=requests.post("http://localhost:11434/api/embed",json=
    {
        "model":"bge-m3",
        "input":text_file
    }
    )
    embeddings=r.json()["embeddings"]
    return embeddings
jsons=os.listdir("json_files")
mydicts=[]
for json_file in jsons:
    with open(f"json_files/{json_file}") as f:
        content=json.load(f)
    chunkid=0
    print(f"creating embeddings for {json_file}")
    embeddings=create_embeddings([c['text'] for c in content['chunks']])
    for i,chunk in enumerate(content["chunks"]):
        chunk["chuk_id"]=chunkid
        chunk["embeddings"]=embeddings[i]
        chunkid+=1
        mydicts.append(chunk)
          
print("all text in json files are converted into embeddings!")
df=pd.DataFrame.from_records(mydicts)
joblib.dump(df, 'embeddings.joblib')
