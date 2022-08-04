import azure.cognitiveservices.speech as speechsdk
import click
import ffmpeg
import logging
import magic
import os
import sys
import subprocess
import time
import typing
from dotenv import load_dotenv
from enum import Enum


load_dotenv()
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

class MediaInfoSection(Enum):
    GENERAL = 'General'
    AUDIO = 'Audio'


class MediaInfoProperty(Enum):
    FORMAT = 'Format'
    WAV_BIT_DEPTH = 'BitDepth'
    WAV_CODEC_ID = 'CodecID'
    WAV_SAMPLING_RATE = 'SamplingRate'


language_mapping = {
    'es-CO': 'Español (Colombia)'
}
output_file_path = None


def get_audio_data(
    audio_filename: str,
    section: MediaInfoSection,
    property: MediaInfoProperty
) -> typing.Optional[str]:
    mediainfo_exe_path = 'C:\\MediaInfo\\MediaInfo.exe'

    metadata = subprocess.check_output([
        mediainfo_exe_path,
        fr'--Inform={section.value};%{property.value}%',
        audio_filename
    ])

    return metadata.decode('utf-8') if metadata else None


def convert_file_to_wav(filename: str) -> str:
    ffmpeg_exe_path = 'C:\\ffmpeg\\bin\\ffmpeg.exe'
    conversion_folder = f'{os.getcwd()}\\{os.getenv("CONVERSIONS_FOLDER")}\\'
    conversion_filename = os.path.basename(filename).split('.')[0] + '.wav'
    conversion_path = conversion_folder + conversion_filename

    logging.info('Converting the file to WAV...')

    subprocess.call([
        ffmpeg_exe_path,
        '-i',
        filename,
        '-acodec',
        'pcm_s16le',
        '-ac',
        '1',
        '-ar',
        '16000',
        conversion_path,
        '-y',
        '-hide_banner',
        '-loglevel',
        'error'
    ])

    if not os.path.isfile(conversion_path):
        logging.error('The file couldn\'t be converted to WAV.')

    return conversion_path


def meets_audio_file_requirements(filename: str) -> bool:
    mime = magic.Magic(mime=True)
    mimetype = mime.from_file(filename)
    mime_parts = mimetype.split('/')

    f_bit_depth = get_audio_data(filename, MediaInfoSection.AUDIO, MediaInfoProperty.WAV_BIT_DEPTH)
    f_codec_id = get_audio_data(filename, MediaInfoSection.AUDIO, MediaInfoProperty.WAV_CODEC_ID)
    f_format = get_audio_data(filename, MediaInfoSection.AUDIO, MediaInfoProperty.FORMAT)
    f_sampling_rate = get_audio_data(filename, MediaInfoSection.AUDIO, MediaInfoProperty.WAV_SAMPLING_RATE)

    if isinstance(f_bit_depth, str):
        f_bit_depth = f_bit_depth.rstrip()

    if isinstance(f_codec_id, str):
        f_codec_id = f_codec_id.rstrip()

    if isinstance(f_format, str):
        f_format = f_format.rstrip()

    if isinstance(f_sampling_rate, str):
        f_sampling_rate = f_sampling_rate.rstrip()

    if mime_parts[0] == 'audio' \
        and 'wav' in mime_parts[1]\
        and f_bit_depth == '16' \
        and f_codec_id == '1' \
        and f_format == 'PCM' \
        and f_sampling_rate == '16000':
        return True

    return False


def parse_file(filename: str) -> typing.Optional[str]:
    if not os.path.isfile(filename):
        logging.error('This file doesn\'t exists.')

    conversion_folder = f'{os.getcwd()}\\{os.getenv("CONVERSIONS_FOLDER")}\\'
    wav_speculative_filename = os.path.basename(filename).split('.')[0] + '.wav'
    wav_speculative_path = conversion_folder + wav_speculative_filename

    if os.path.isfile(wav_speculative_path) and \
        meets_audio_file_requirements(wav_speculative_path):
        logging.info('A conversion of this file was found that satisfies all requirements.')
        return wav_speculative_path

    if not meets_audio_file_requirements(filename):
        logging.info('No conversion was found for this file.')
        return convert_file_to_wav(filename)

    return None


def transcript_file(filename: str):
    subscription = os.getenv('AZURE_SUBSCRIPTION_SPEECH_KEY')
    region = os.getenv('AZURE_SUBSCRIPTION_REGION')
    language = os.getenv('AUDIO_LANGUAGE')

    logging.info(f'Setting up the audio transcription language to: {language_mapping.get(language)}')

    speech_config = speechsdk.SpeechConfig(
        subscription=subscription,
        region=region,
        speech_recognition_language=language
    )
    audio_input = speechsdk.AudioConfig(filename=filename)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_input
    )

    done = False

    def stop_cb(evt):
        logging.info(f'CLOSING on {evt}')
        logging.info(f'{evt.result}')
        export_transcription(evt.result.text, output_file_path)
        nonlocal done
        done = True

    logging.info('Transcripting file...')

    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognizing.connect(lambda evt: analyze_transcription(speech_recognizer, evt))
    speech_recognizer.recognized.connect(lambda evt: analyze_transcription(speech_recognizer, evt))
    speech_recognizer.session_started.connect(lambda evt: analyze_transcription(speech_recognizer, evt))
    speech_recognizer.session_stopped.connect(lambda evt: analyze_transcription(speech_recognizer, evt))
    speech_recognizer.canceled.connect(lambda evt: analyze_transcription(speech_recognizer, evt))

    # stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    speech_recognizer.start_continuous_recognition_async()

    while not done:
        time.sleep(.5)

    speech_recognizer.stop_continuous_recognition_async()


def export_transcription(
    transcription_text: str,
    destination_filename: str
) -> str:
    destination_path = os.getenv('TRANSCRIPTIONS_FOLDER') + '/' + destination_filename

    with open(destination_path, 'w') as df:
        df.write(transcription_text)

    logging.info('Transcription successfully exported.')

    return destination_path


def analyze_transcription(
    speech_recognizer: speechsdk.SpeechRecognizer,
    event: speechsdk.SpeechRecognitionEventArgs
) -> str:
    if speech_recognizer.recognizing:
        logging.info(f'RECOGNIZING: {event.result.text}')
    elif speech_recognizer.recognized:
        if event.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            logging.info(f'RECOGNIZED: Successful transcription!')
            export_transcription(event.result.text, output_file_path)
        elif event.result.reason == speechsdk.ResultReason.NoMatch:
            logging.error(f'NOMATCH: Transcription couldn\'t be recognized. {event.result.no_match_details}')
    elif speech_recognizer.canceled:
        if event.result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = event.result.cancellation_details
            logging.warning(f'CANCELED: Transcription canceled: {cancellation_details.reason}')
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                logging.error(f'CANCELED: Error details: {cancellation_details.error_details}')
    elif speech_recognizer.session_started:
        logging.info(f'TRANSCRIPTION SESSION STARTED: {event.session_id}')
    elif speech_recognizer.session_stopped:
        logging.info(f'TRANSCRIPTION SESSION STOPPED: {event.session_id}')


@click.command()
@click.argument('input_file', type=click.STRING)
@click.argument('output_file', type=click.STRING)
def main(input_file, output_file):
    output_file_path = output_file
    parsed_file = parse_file(input_file)
    transcript_file(parsed_file)

if __name__ == '__main__':
    main()
