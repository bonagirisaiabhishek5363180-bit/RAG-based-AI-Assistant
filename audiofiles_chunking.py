import whisper
import os
import json
# i have run this file on google colab u can run it on local machine too 
# from google.colab import drive
# Mount Google Drive
# drive.mount("/content/drive/", force_remount=True)

# Input and output folders
# input_folder = "/content/drive/MyDrive/audios"
# output_folder = "/content/drive/MyDrive/json_files"
input_folder="audios"
output_folder="json_files"
# Load model
model = whisper.load_model("small")

# Get audio files
audios = os.listdir(input_folder)

# Process each audio
for audio in audios:
    print(f"processing: {audio}")
    
    audio_number = audio.split("_")[0]
    audio_name = audio.split("_")[1][:-4]
    chunks = []
    output_path=f"{output_folder}/{audio}.json"
    if os.path.exists(output_path):
        print(f"the audio:{audio} already transcribed!")
        continue
    result = model.transcribe(
        audio=f"{input_folder}/{audio}",
        language="hi",
        task="translate",
        word_timestamps=False,
        fp16=False
    )
    
    for chunk in result["segments"]:
        chunks.append({
            "number": audio_number,
            "name": audio_name,
            "start": chunk["start"],
            "end": chunk["end"],
            "text": chunk["text"]
        })
    
    chunks_with_text = {
        "chunks": chunks,
        "text": result["text"]
    }
    
    with open(output_path, "w") as f:
        json.dump(chunks_with_text, f)
