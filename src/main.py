from pathlib import Path
import os
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

with open(Path.cwd()/"src"/"lost_debit_card.wav", "rb") as audioFile:
  transcript = openai.Audio.transcribe("whisper-1", audioFile)
  print(transcript)
