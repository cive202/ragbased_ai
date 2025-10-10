#convert videos to mp3
import os
import subprocess

files = os.listdir("videos")
for file in files:
    print(file)
    tutorial_num = file.split(" -")[0].split("l ")[1]
    print(tutorial_num)
    tutorial_name = file.split("- ")[1]
    print(tutorial_name)
    subprocess.run(["ffmpeg", "-i",f"videos/{file}",f"audios/{tutorial_num}_{tutorial_name}.mp3"])


