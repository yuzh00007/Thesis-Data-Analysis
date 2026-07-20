import speech_recognition as sr
import time
import glob
import numpy as np
import librosa
import scipy.io.wavfile as wavfile
import os
import csv
from datetime import datetime
import subprocess
import logging
logger = logging.getLogger(__name__)


def mp3_to_wav(path: str) -> str:
    """
    Converts an MP3 file to WAV format without using ffmpeg.

    Parameters:
        path (str): The file path of the MP3 file to convert.

    Returns:
        str: The file path of the newly created WAV file.
    """
    # Define the output folder and create it if necessary
    output_folder = "./dataOutput/tmp"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Generate a unique output filename
    output_filename = os.path.join(
        output_folder, "converted_{:%Y%m%d%H%M%S}.wav".format(datetime.now())
    )
    # Load MP3 file with librosa
    audio_data, sample_rate = librosa.load(path, sr=None)
    wavfile.write(output_filename, sample_rate, (audio_data * 32767).astype("int16"))

    return output_filename


# Initialize the speech recognition object
r = sr.Recognizer()


def extract_text(path: str) -> str:
    """
    Converts speech in an audio file to text by splitting the audio based on silence.

    Parameters:
        path (str): File path to the audio file to be processed.

    Returns:
        str: The transcribed text from the audio file.

    Notes:
        This function splits audio where silence is detected, processes each chunk
        for speech recognition, and removes temporary files after processing.
    """
    # Convert MP3 to WAV if needed
    file_name = mp3_to_wav(path)

    # Load the audio with librosa
    audio_data, sample_rate = librosa.load(file_name, sr=None)

    # Define silence threshold and chunking parameters
    silence_threshold = np.mean(np.abs(audio_data)) * 0.02  # adjust based on audio
    chunk_duration = 0.5  # seconds of silence that defines a split, adjustable

    # Split based on silence
    non_silent_intervals = librosa.effects.split(audio_data, top_db=14)

    # Recognize the audio chunk
    try:
        with sr.AudioFile(file_name) as source:
            r.adjust_for_ambient_noise(source)

            audio = r.record(source)
            text = r.recognize_google(audio)
            text = text.lower()

            return text

    except sr.UnknownValueError:
        print(f'Could not process, continuing')

    except sr.RequestError as e:
        print("Google error; {0}".format(e))

    except Exception as e:
        print("Exception - Continuing")


def txt_file(file_path: str, output: str):
    """
    Extracts text from an audio file and saves it as a text file on the user's desktop.

    Parameters:
        file_path (str): Path to the audio file for transcription.

    Returns:
        None
    """
    # Extract text from the audio file
    text = extract_text(file_path)

    # Write to the text file
    with open(output, "w") as file:
        file.write(text)


def generate_csv(text_recognized, outname):
    out = csv.writer(open(outname, 'w'))
    out.writerow(['participant', 'audio1', 'audio2', 'audio3'])

    for row in text_recognized:
        out.writerow(row)


if __name__ == "__main__":
    start = time.time()

    audio_folder = "./dataOutput/decodedAudio/"
    subprocess_mp3_args = ["ffmpeg", "-i", "INPUTWAV", "-vn", "-ar", "44100", "-ac", "2", "-b:a", "192k", "OUTPUTMP3"]

    for participantFolder in os.scandir(audio_folder):
        for audioFile in sorted(os.listdir(participantFolder)):
            audioFilePath = participantFolder.path + "/" + audioFile
            if os.path.isfile(audioFilePath):
                output_file = audioFilePath.replace(".wav", ".mp3")

                # in case file exists, skip it
                if not os.path.isfile(output_file):
                    subprocess_mp3_args[2] = audioFilePath
                    subprocess_mp3_args[-1] = output_file
                    subprocess.run(subprocess_mp3_args)

    logger.info("starting audio transcription")
    all_transcripts = []
    for participant in os.scandir(audio_folder):
        partname = participant.name.replace("_audio", "")
        indiv_transcripts = [partname]
        for audioFile in sorted(glob.glob(f'{participant.path}/*.mp3')):
            text = extract_text(audioFile)
            indiv_transcripts.append(text)

        all_transcripts.append(indiv_transcripts)

    generate_csv(all_transcripts, "./dataOutput/audioTranscript.csv")
    print(time.time() - start)
