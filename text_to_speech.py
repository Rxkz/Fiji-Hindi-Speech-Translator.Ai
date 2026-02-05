"""
Text-to-Speech Module using gTTS
Converts Fiji Hindi text to speech audio
"""

import edge_tts
import asyncio
import pygame
import os
import tempfile

import edge_tts
import asyncio
import pygame
import os
import tempfile


class TextToSpeech:
    def __init__(self, *args, **kwargs):
        self.voice = "hi-IN-MadhurNeural"
        self.rate = "-15%"

        # Ensure mixer is clean
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        pygame.mixer.init()
        print(f"Text-to-speech ready (Edge-Neural: Madhur)")

    def speak(self, text, **kwargs):
        if not text:
            return None

        # Create a temp file path
        temp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(temp_dir, f"tts_output.mp3")

        # Inner function to handle the async generation
        async def generate():
            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
            await communicate.save(tmp_path)

        # 1. Generate the file
        print(f"Generating audio for: {text}...")
        asyncio.run(generate())

        # 2. Play the file using a rock-solid blocking method
        if os.path.exists(tmp_path):
            try:
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()

                # Force the program to wait until the sound is 100% finished
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

                pygame.mixer.music.unload()
                os.remove(tmp_path)
            except Exception as e:
                print(f"Playback error: {e}")

        return True

    def speak_and_cleanup(self, text):
        return self.speak(text)


# main function to test the text to speech module
if __name__ == "__main__":
    # Initialize
    tts = TextToSpeech()

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
