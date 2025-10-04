# you can enter the url of yt playlist or you can add videos to videos folder manually
from pytubefix import   Playlist
# url = "https://youtube.com/playlist?list=PLu0W_9lII9agwh1XjRt242xIpHhPT2llg&si=-1_aqk-VBziyCe3Z"
url=input("Enter the playlist url:")
playlist = Playlist(url)
for video in playlist.videos:
    print(f"Downloading: {video.title}")
    stream = video.streams.filter(progressive=True, file_extension="mp4").order_by('resolution').desc().first()
    if stream:
        stream.download(output_path="RAG BASED PROJECT/videos/")
print("All videos are downloaded successfully!")        