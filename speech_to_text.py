"""
Speech-to-Text Module using OpenAI Whisper
Captures audio from microphone and transcribes to English text
"""

from this import d
import whisper
import sounddevice as sd
import numpy as np
import soundfile as sf
import tempfile
import os


class SpeechToText:
    """Speech-to-Text Module which uses OpenAI Whisper to transcribe audio from microphone to English text"""

    def __init__(self, model_size="base"):
        """
        Initialize Whisper model

        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
                       base = good balance of speed/accuracy for M4 Pro
        """

        print(f"loading Whisper model ({model_size})...")
        self.model = whisper.load_model(model_size)
        print("Whisper model loaded successfully")

        # audio capture settings
        self.sample_rate = 16000  # standard for whisper
        self.channels = 1  # mono audio

    # capture audio from microphone
    def record_audio(self, duration=5):
        """
        Record audio from microphone

        Args:
            duration: Recording duration in seconds

        Returns:
            audio_data: NumPy array of audio samples
        """
        print(f"Recording {duration} seconds of audio...")
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
        )
        sd.wait()
        print("Recording complete")
        return audio_data.flatten()

    # transcribe audio to text
    def transcribe_audio(self, audio_data):
        """
        Transcribe audio to text using Whisper

        Args:
            audio_data: NumPy array of audio samples

        Returns:
            text: Transcribed English text
        """
        print("Transcribing audio to text...")
        result = self.model.transcribe(audio_data, language="en")
        text = result["text"].strip()
        print(f"Transcribed text: {text}")
        return text

    # record and transcribe audio
    def record_and_transcribe(self, duration=5):
        """
        Record audio and transcribe in one step

        Args:
            duration: Recording duration in seconds

        Returns:
            text: Transcribed text
        """
        audio_data = self.record_audio(duration)
        text = self.transcribe_audio(audio_data)
        return text

    # save audio to file
    def save_audio(self, audio_data, filename="temp_audio.wav"):
        """
        Save audio to file

        Args:
            audio_data: NumPy array of audio samples
            filename: Name of the file to save
        """
        print(f"Saving audio to {filename}...")
        sf.write(filename, audio_data, self.sample_rate)
        print(f"Audio saved to {filename}")


# main function to test the speech to text module
if __name__ == "__main__":
    # intialise speech to text module
    stt = SpeechToText(model_size="base")

    print("testing speech to text module...")
    print("=" * 60)
    # test 1 record and transcribe audio

    print("\nTest 1: Record and Transcribe Audio")

    while True:
        choice = (
            input("\nDo you want to record and transcribe audio? (y/n /quit/q/exit): ")
            .lower()
            .strip()
        )
        if choice in ["quit", "q", "exit"]:
            break

        if choice == "y":
            try:
                text = stt.record_and_transcribe(duration=5)
                print(f"\n Result: '{text}'")
            except Exception as e:
                print(f" Error: {e}")

        print()

    print("Test complete!")
