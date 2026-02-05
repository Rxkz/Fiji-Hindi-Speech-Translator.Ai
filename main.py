"""
PROJECT 47: Live Speech Translation System
English ↔ Fiji Hindi Real-Time Translation

Architecture:
  English Speech → [Whisper] → English Text
                     ↓
              [Transformer] → Fiji Hindi Text
                     ↓
                  [gTTS] → Fiji Hindi Speech

Usage: python3 main.py
"""

import sys
import time
from speech_to_text import SpeechToText
from text_to_speech import TextToSpeech
from translator import FijiHindiTranslator


class LiveSpeechTranslator:
    """Real-time English to Fiji Hindi speech translation"""

    def __init__(self):
        """Initialize components"""
        print("=" * 70)
        print("PROJECT 47: LIVE SPEECH TRANSLATION SYSTEM")
        print("English → Fiji Hindi Real-Time Translation")
        print("=" * 70)
        self.stt = SpeechToText(model_size="base")
        self.translator = FijiHindiTranslator()
        self.tts = TextToSpeech(lang="hi", slow=False)
        print(" System ready for translation")
        print()

    # start the translation loop
    def translate_once(self, duration=5):
        """
        Single translation cycle

        Args:
            duration: Recording duration in seconds

        Returns:
            (english_text, fiji_hindi_text)
        """

        print(f" translating for {duration} seconds...")

        # step 1: record audio
        print("Recording English speech...")
        english_text = self.stt.record_and_transcribe(duration=duration)

        if not english_text:
            print("no english speech detected")
            return None

        print(f"English text: {english_text}")

        # step 2: translate to fiji hindi
        print(f"Translating to Fiji Hindi...")
        fiji_hindi_text = self.translator.translate(english_text)
        print(f"Fiji Hindi text: {fiji_hindi_text}")

        # step 3: speak the fiji hindi text
        print(f"Speaking Fiji Hindi...")
        self.tts.speak_and_cleanup(fiji_hindi_text)
        # time.sleep(0.1)
        return english_text, fiji_hindi_text

    # interactive mode for real-time translation
    def run_interactive(self):
        """Interactive mode with user control"""

        print("\n" + "=" * 70)
        print("INTERACTIVE MODE")
        print("=" * 70)
        print("Commands:")
        print("  'go' or 'g'     - Start translation (5 seconds)")
        print("  'long' or 'l'   - Long recording (10 seconds)")
        print("  'quit' or 'q'   - Exit")
        print("=" * 70)
        print()

        translation_count = 0

        while True:
            command = input("\nCommand: ").strip().lower()

            if command in ["quit", "q", "exit"]:
                print("\n" + "=" * 70)
                print(f"Total translations: {translation_count}")
                print("Vinaka! (Thank you!)")
                print("=" * 70)
                break

            elif command in ["go", "g"]:
                translation_count += 1
                self.translate_once(duration=5)

            elif command in ["long", "l"]:
                translation_count += 1
                self.translate_once(duration=10)

            elif command == "":
                continue

            else:
                print(f"Unknown command: '{command}'")
                print("Use 'go', 'long', or 'quit'")


# main function to test the live speech translator
def main():
    """Main function to test the live speech translator"""
    # intialise the live speech translator
    try:
        translator = LiveSpeechTranslator()
    except Exception as e:
        print(f"\nError initializing system: {e}")
        print("Make sure all dependencies are installed and model files are present.")
        sys.exit(1)

    # choose mode
    print("SELECT MODE")
    print("=" * 70)
    print("1. Interactive Mode (translate with y/n control)")
    print("2. Single Test (one translation then exit)")
    print("=" * 70)

    while True:
        choice = input("\nSelect mode (1/2): ").strip()

        if choice == "1":
            translator.run_interactive()
            time.sleep(2)
            break
        elif choice == "2":
            translator.translate_once(duration=5)
            time.sleep(2)
            break
        else:
            print("Invalid choice. Enter 1 or 2.")


if __name__ == "__main__":
    main()
