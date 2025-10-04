import subprocess
import os
files=os.listdir("videos")
for file in files:
    print(file)
    # here i split the file number and file title on basis of my videos in videos folder
    video_number=file.split(".mp4")[0].split("#")[1]
    video_name=file.split(" - ")[0]
    print(video_number,video_name)
    # subprocess run  riins the given statement of list in command prompt
    subprocess.run(["ffmpeg","-i",f"videos/{file}",f"audios/{video_number}_{video_name}.mp3"])
print("All the video files are converted into audio files!")