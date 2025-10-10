import whisper
import json

model = whisper.load_model("base")
result = model.transcribe("audios/23_Blinn-Phong Lighting.mp3", language="en")

#with open ("output.json","w") as f:
#    json.dump(result["segments"],f,indent=4)
chunks = []
for segment in result["segments"]:
    chunks.append({"start" : segment["start"],"end" : segment["end"], "text" : segment["text"]}) 
with open ("output.json","w") as f:
    json.dump(chunks,f,indent=4)