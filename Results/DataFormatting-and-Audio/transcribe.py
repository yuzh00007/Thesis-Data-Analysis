import speech_recognition as sr
import numpy as np
import timeit
import librosa
import scipy.io.wavfile as wavfile
import os
import csv
from datetime import datetime
import subprocess


def mp3_to_wav(path: str) -> str:
    """
    Converts an MP3 file to WAV format without using ffmpeg.

    Parameters:
        path (str): The file path of the MP3 file to convert.

    Returns:
        str: The file path of the newly created WAV file.
    """
    # Define the output folder and create it if necessary
    output_folder = "tmp"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Generate a unique output filename
    output_filename = os.path.join(
        output_folder, "converted_{:%Y%m%d%H%M%S}.wav".format(datetime.now())
    )

    # Load MP3 file with librosa
    audio_data, sample_rate = librosa.load(path, sr=None)

    # Write to WAV format using scipy
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
            print(text)
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
    start = timeit.timeit()

    audio_folder = "./decodedAudioTest"
    subprocess_mp3_args = ["ffmpeg", "-i", "INPUTWAV", "-vn", "-ar", "44100", "-ac", "2", "-b:a", "192k", "OUTPUTMP3"]

    for folder in os.scandir(audio_folder):
        for file in os.scandir(folder):
            if os.path.isfile(file):
                output_file = file.path.replace(".wav", ".mp3")
                subprocess_mp3_args[2] = file.path
                subprocess_mp3_args[-1] = output_file
                subprocess.run(subprocess_mp3_args)

    all_transcripts = []
    for participant in os.scandir(audio_folder):
        indiv_transcripts = [participant]
        for file in os.scandir(participant):
            if file.name.endswith('.mp3'):
                text = extract_text(file.path)
                indiv_transcripts.append(text)

        all_transcripts.append(indiv_transcripts)

    generate_csv(all_transcripts, "transcript.csv")
    print(timeit.timeit() - start)
