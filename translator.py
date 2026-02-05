"""
Project 47: Fiji Hindi Live Speech Translator
Translator Module - Wrapper for Project 43 Transformer
Loads trained model and provides simple translate() function
"""

import numpy as np
import tensorflow as tf
from transformer import Transformer, create_vocab, encode_sentence, PAD, START, END


class FijiHindiTranslator:
    """Wrapper for Fiji Hindi translation model"""

    def __init__(
        self,
        weights_path="models/project_43_fiji_hindi_transformer.weights.h5",
        data_path="data/test_audio/fiji_hindi_data.txt",
    ):
        """
        Initialize translator and load model

        Args:
            weights_path: Path to trained model weights
            data_path: Path to training data (needed for vocabulary)
        """
        print(f"Loading model from {weights_path}...")
        print(f"Loading data from {data_path}...")

        # load training to rebuild vocabulary
        self.eng_sentences, self.fiji_sentences = self._load_data(data_path)

        # create vocabulary
        self.eng_w2i, self.eng_i2w, self.eng_vocab = create_vocab(self.eng_sentences)
        self.fiji_w2i, self.fiji_i2w, self.fiji_vocab = create_vocab(
            self.fiji_sentences
        )

        print(f"English vocabulary: {self.eng_vocab} words")
        print(f"Fiji Hindi vocabulary: {self.fiji_vocab} words")

        # calculate max length of sentences
        self.max_eng_len = max(len(s.split()) for s in self.eng_sentences)
        self.max_fiji_len = max(len(s.split()) for s in self.fiji_sentences)

        # build model same as training in project 43
        print("\nBuilding Transformer...")
        self.model = Transformer(
            num_layers=4,
            d_model=256,
            num_heads=8,
            dff=2028,
            input_vocab_size=self.eng_vocab,
            target_vocab_size=self.fiji_vocab,
            max_len=max(self.max_eng_len, self.max_fiji_len),
            dropout_rate=0.2,
        )

        # bulid model by running dummy data through it
        dummy_input = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32)
        dummy_target = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32)
        _ = self.model([dummy_input, dummy_target], training=False)

        # load model weights
        self.model.load_weights(weights_path)
        print("Model loaded successfully")
        print("Model is ready for translation!")

    # load training data
    def _load_data(self, filepath):
        """Load training data from file to rebuild vocabulary"""
        english_sentences = []
        fiji_sentences = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("\\"):
                    continue
                if "|" in line:
                    parts = line.split("|")
                    if len(parts) == 2:
                        eng = parts[0].strip().lower()
                        fiji = parts[1].strip().lower()
                        if eng and fiji:
                            english_sentences.append(eng)
                            fiji_sentences.append(fiji)
        print(f"Loaded {len(english_sentences)} translation pairs")
        return english_sentences, fiji_sentences

    # translate english to fiji hindi
    def translate(self, text):
        """
        Translate English text to Fiji Hindi

        Args:
            text: English text (string)

        Returns:
            Fiji Hindi translation (string)
        """
        # preprocess text
        text = text.lower().strip()

        if not text:
            return " plese enter some text to translate"

        # encode input
        enc_input = encode_sentence(text, self.eng_w2i, self.max_eng_len)
        enc_input = np.array([enc_input])

        # prep decoder input
        dec_input = np.zeros((1, self.max_fiji_len), dtype="int32")
        dec_input[0, 0] = self.fiji_w2i[START]

        translated = []

        for i in range(self.max_fiji_len - 1):
            predictions = self.model.predict([enc_input, dec_input], verbose=0)
            next_word_idx = np.argmax(predictions[0, i, :])
            next_word = self.fiji_i2w[next_word_idx]

            if next_word == END:
                break

            if next_word not in [PAD, START]:
                translated.append(next_word)

            if i + 1 < self.max_fiji_len:
                dec_input[0, i + 1] = next_word_idx
        return " ".join(translated)

    # test the translator
    def test_translation(self):
        """Test the translator with some common words/phrases"""
        print("testing translator with some common words/phrases")

        test_cases = ["hello", "thank you", "water", "how are you"]

        for eng in test_cases:
            fiji = self.translate(eng)
            print(f"   '{eng}' → '{fiji}'")

        print(" Test complete!")
        print()


# main use function
def main():
    # intialise translator
    translator = FijiHindiTranslator()

    # test the translator
    translator.test_translation()

    # Interactive mode
    print("=" * 70)
    print("INTERACTIVE TRANSLATOR")
    print("=" * 70)
    print("Type English text, get Fiji Hindi translation")
    print("Type 'quit' to exit")
    print("=" * 70)
    print()

    while True:
        english = input("English: ").strip()

        if english.lower() in ["quit", "exit", "q"]:
            print("Vinaka! (Thank you!)")
            break

        if not english:
            continue

        fiji_hindi = translator.translate(english)
        print(f"Fiji Hindi: {fiji_hindi}")
        print()


# run
if __name__ == "__main__":
    main()
