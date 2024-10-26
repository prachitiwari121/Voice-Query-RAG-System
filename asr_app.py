# from fastapi import FastAPI, File, UploadFile, HTTPException
# import nemo.collections.asr as nemo_asr
# from pydub import AudioSegment
# import os
# import time
# import torch

# app = FastAPI()

# #------------------------------------------------------------------Device Configuration-----------------------------------------------------------------------------------------------
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load the ASR model
# try:
#     model_path = "C:\\Users\\prach\\Downloads\\ai4b_indicConformer_hi.nemo"  # Update with your model path
#     model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=model_path)
#     model.freeze()
#     model.to(device)
#     print("Model loaded successfully and moved to device:", device)
# except Exception as e:
#     print(f"Error loading model: {e}")
#     model = None

# # Function to convert audio to WAV format
# def convert_audio_to_wav(audio_file):
#     try:
#         audio = AudioSegment.from_file(audio_file)
#         audio = audio.set_channels(1)  # Convert to mono
#         wav_file = "temp.wav"
#         audio.export(wav_file, format="wav")
#         return wav_file
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Audio conversion error: {e}")

# # Function to transcribe audio using CTC decoding
# def transcribe_audio_ctc(wav_file):
#     try:
#         model.cur_decoder = 'ctc'
#         transcription = model.transcribe([wav_file], batch_size=1, logprobs=False)[0]
#         return transcription
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"CTC Transcription error: {e}")

# # Function to transcribe audio using RNN-T decoding (if supported)
# def transcribe_audio_rnnt(wav_file):
#     try:
#         model.cur_decoder = 'rnnt'
#         transcription = model.transcribe([wav_file], batch_size=1, logprobs=False)[0]
#         return transcription
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"RNN-T Transcription error: {e}")

# # Endpoint to handle audio file transcription
# @app.post("/transcribe/")
# async def transcribe_audio_endpoint(file: UploadFile = File(...), decoding_type: str = "ctc"):
#     if not model:
#         raise HTTPException(status_code=500, detail="ASR model is not available.")

#     audio_file_path = f"temp_{file.filename}"
#     with open(audio_file_path, "wb") as buffer:
#         buffer.write(await file.read())

#     wav_file_path = None
#     try:
#         wav_file_path = convert_audio_to_wav(audio_file_path)
#         if decoding_type == "ctc":
#             transcription = transcribe_audio_ctc(wav_file_path)
#         elif decoding_type == "rnnt":
#             transcription = transcribe_audio_rnnt(wav_file_path)
#         else:
#             raise HTTPException(status_code=400, detail="Invalid decoding type specified.")
#     finally:
#         if os.path.exists(audio_file_path):
#             os.remove(audio_file_path)
#         if wav_file_path and os.path.exists(wav_file_path):
#             time.sleep(1)
#             os.remove(wav_file_path)

#     return {"transcription": transcription}





from fastapi import FastAPI, File, UploadFile, HTTPException
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
import os
import time
import torch

app = FastAPI()

#------------------------------------ Device Configuration --------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the ASR model
try:
    model_path = "C:\\Users\\prach\\Downloads\\ai4b_indicConformer_hi.nemo"  # Update with your model path
    model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=model_path)
    model.freeze()
    model.to(device)
    print("Model loaded successfully and moved to device:", device)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

#------------------------------------ Audio Conversion --------------------------------------
def convert_audio_to_wav(audio_file):
    try:
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_channels(1)  # Convert to mono
        wav_file = "temp.wav"
        audio.export(wav_file, format="wav")
        return wav_file
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio conversion error: {e}")

#------------------------------------ CTC Decoding Transcription --------------------------------------
def transcribe_audio_ctc(wav_file):
    try:
        model.cur_decoder = 'ctc'
        transcription = model.transcribe([wav_file], batch_size=1, logprobs=False)[0]
        return transcription
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CTC Transcription error: {e}")

#------------------------------------ RNN-T Decoding Transcription --------------------------------------
def transcribe_audio_rnnt(wav_file):
    try:
        model.cur_decoder = 'rnnt'
        transcription = model.transcribe([wav_file], batch_size=1, logprobs=False)[0]
        return transcription
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RNN-T Transcription error: {e}")

#------------------------------------ Endpoint for Audio Transcription --------------------------------------
@app.post("/transcribe/")
async def transcribe_audio_endpoint(file: UploadFile = File(...), decoding_type: str = "ctc"):
    if not model:
        raise HTTPException(status_code=500, detail="ASR model is not available.")

    audio_file_path = f"temp_{file.filename}"
    with open(audio_file_path, "wb") as buffer:
        buffer.write(await file.read())

    wav_file_path = None
    try:
        wav_file_path = convert_audio_to_wav(audio_file_path)
        if decoding_type == "ctc":
            transcription = transcribe_audio_ctc(wav_file_path)
        elif decoding_type == "rnnt":
            transcription = transcribe_audio_rnnt(wav_file_path)
        else:
            raise HTTPException(status_code=400, detail="Invalid decoding type specified.")
    finally:
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
        if wav_file_path and os.path.exists(wav_file_path):
            time.sleep(1)
            os.remove(wav_file_path)

    return {"transcription": transcription}
