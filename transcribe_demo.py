#! python3.7

import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch
import obsws_python as obs

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
from deep_translator import MicrosoftTranslator

msftkey = None
obscl = None



def translate(text, fromlang, tolang):
    result = "Needs translation"
    
    microsoftTranslator = MicrosoftTranslator(api_key=msftkey, source=fromlang, target=tolang, region='southcentralus')
    try:
        result = microsoftTranslator.translate(f"""{text}""")
    except Exception as e:
        result = text + f" (Translation failed - {e})"
    return result

def sendCaption(text):
    try:
        obscl.send_stream_caption(text)
    except Exception as e:
        print(f"OBS request failed: {e}")
    pass

def main():
    global msftkey
    global obspass
    global obsserver
    global obscl
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--english_only", action='store_true',
                        help="Use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    parser.add_argument('--obs-server', default='localhost:4455')
    parser.add_argument('--obs-pass')
    parser.add_argument('--msft-apikey')
    parser.add_argument("--default_microphone", default='pulse',
                        help="Default microphone name for SpeechRecognition. "
                           "Run this with 'list' to view available Microphones.", type=str)
    # To get both myself and discord involved, have to listen to my mic onto VAC1, have discord output to VAC2 (and listen to VAC2 on actual audio output), and use a VAC Repeater to send VAC2 to VAC1
    args = parser.parse_args()
    obspass     = args.obs_pass
    obsserver   = args.obs_server
    if obspass:
        obscl = obs.ReqClient(host=obsserver.split(':')[0], port=obsserver.split(':')[1], password=obspass)
    msftkey     = args.msft_apikey
    # The last time a recording was retreived from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False
    
    # Important for linux users. 
    # Prevents permanent application hang and crash by using the wrong Microphone
    mic_name = args.default_microphone
    if mic_name == 'list':
        print("Available microphone devices are: ")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"Microphone with name \"{name}\" found")   
        return
    elif not mic_name:
        source = sr.Microphone(sample_rate=16000)
    else:
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if mic_name in name:
                source = sr.Microphone(sample_rate=16000, device_index=index)
                break

        
    # Load / Download model
    model = args.model
    if args.model != "large" and args.english_only:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name
    transcription = ['']
    
    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
                text = result['text'].strip()
                translated = text
                if result['language'] == 'en':
                    translated = translate(text, result['language'], 'es')
                else:
                    translated = translate(text, result['language'], 'en')
                text = translated
                # If we detected a pause between recordings, add a new item to our transcripion.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                    if obscl:
                        sendCaption(text)
                else:
                    transcription[-1] = text

                # Clear the console to reprint the updated transcription.
                os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription:
                    print(line)
                # Flush stdout.
                print('', end='', flush=True)

                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nOriginal Transcription:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()