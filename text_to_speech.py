"""
Text-to-Speech Module using gTTS
Converts Fiji Hindi text to speech audio
"""

from gtts import gTTS
import os
import tempfile
import platform
import time
from whisper import audio


class TextToSpeech:
    """gTTS-based text-to-speech for Fiji Hindi"""

    def __init__(self, lang="hi", slow=False):
        """
        Initialize TTS

        Args:
            lang: Language code (hi = Hindi, closest to Fiji Hindi)
        """

        print(f"Text-to-speech ready (language: {lang} slow: {slow})")
        self.lang = lang
        self.slow = slow
        print()

    # convert text to speech
    def speak(self, text, save_path=None, play=True):
        """
        Convert text to speech and play

        Args:
            text: Fiji Hindi text to speak
            save_path: Optional path to save audio file
            play: Whether to play audio automatically

        Returns:
            audio_file: Path to generated audio file
        """
        if not text:
            print("Error: No text to speak")
            return None

        print(f"Speaking text: {text}")
        # add padding to text
        padded_text = f", ... {text} "

        # generate audio
        tts = gTTS(text=padded_text, lang=self.lang, slow=self.slow)

        # save audio to file
        if save_path:
            # Create temporary file in system temp directory
            temp_dir = tempfile.gettempdir()
            timestamp = int(time.time() * 1000)
            audio_file = os.path.join(temp_dir, f"tts_{timestamp}.mp3")
        else:
            audio_file = save_path

        # save audio
        tts.save(audio_file)
        print(f"audio saved to {audio_file}")

        # play audio
        if play:
            self._play_audio(audio_file)
        return audio_file

    # play audio
    def _play_audio(self, audio_file):
        """
        Play audio file using system player

        Args:
            audio_file: Path to audio file
        """
        print(f"Playing audio from {audio_file}...")
        system = platform.system()

        try:
            if system == "Darwin":  # macOS
                time.sleep(0.1)
                os.system(f"afplay {audio_file}")
            elif system == "Windows":  # Windows
                os.system(f"start {audio_file}")
            elif system == "Linux":  # Linux
                os.system(f"aplay {audio_file}")
            else:
                print(f"Unsupported system: {system}")
                print("Please manually play the audio file")
        except Exception as e:
            print(f"Error playing audio: {e}")
            print("Please manually play the audio file")
        return True

    # speck and clean up
    def speak_and_cleanup(self, text):
        """
        Speak text and clean up temporary file

        Args:
            text: Text to speak
        """
        audio_file = self.speak(text)

        # wait for audio to finish playing
        time.sleep(0.5)

        # clean up temporary file
        if audio_file and os.path.exists(audio_file):
            try:
                os.remove(audio_file)
                print(f"Temporary file {audio_file} removed")
            except Exception as e:
                print(f"Error removing temporary file: {e}")
        return True


# main function to test the text to speech module
if __name__ == "__main__":
    # Initialize
    tts = TextToSpeech(lang="hi", slow=False)

    print("=" * 70)
    print("TEXT-TO-SPEECH TEST")
    print("=" * 70)
    print()

    # Test phrases
    test_phrases = ["namaste", "dhanyavaad", "paani", "kaise ho", "hamke paani deo"]

    print("Testing common Fiji Hindi phrases:")
    print("-" * 70)

    for phrase in test_phrases:
        print(f"\nTesting: '{phrase}'")
        choice = input("Play this? (y/n/quit): ").strip().lower()

        if choice in ["quit", "q", "exit"]:
            break

        if choice == "y":
            try:
                tts.speak_and_cleanup(phrase)
            except Exception as e:
                print(f"Error: {e}")

        print()

    # Interactive mode
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("Type Fiji Hindi text to hear it spoken")
    print("Type 'quit' to exit")
    print("=" * 70)
    print()

    while True:
        text = input("\nFiji Hindi text: ").strip()

        if text.lower() in ["quit", "q", "exit"]:
            break

        if not text:
            continue

        try:
            tts.speak_and_cleanup(text)
        except Exception as e:
            print(f"Error: {e}")

    print("\nTest complete!")
