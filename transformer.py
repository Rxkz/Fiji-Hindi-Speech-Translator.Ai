"""
Note this is same copy paste from Project 43. I am using this for the live speech transfomer in 47 project.
PROJECT 43: Transformer Translation - Fiji Hindi
Goal: Build working English ↔ Fiji Hindi translator

Strategy:
- Expanded dataset (you're generating 5K+ pairs!)
- 100 epochs locally (M4 Pro will crush it!)
- Then Colab for 200-300 epochs


This completes the 37 → 38 → 43 arc (Project 37, 38, 43)
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers

PAD = "<PAD>"
START = "<START>"
END = "<END>"

print(f"TensorFlow version: {tf.__version__}")
print("PROJECT 43: Transformer Translation - Fiji Hindi")
print("=" * 70)


# Transformers
# step 1 positional encoding
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pos_encoding = np.zeros((max_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)

        self.pos_encoding = tf.constant(
            pos_encoding[np.newaxis, :, :], dtype=tf.float32
        )

    def call(self, x):
        return x + self.pos_encoding[:, : tf.shape(x)[1], :]


def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += mask * -1e9

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)
        return output, attention_weights


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(dff, activation="relu"), layers.Dense(d_model)]
        )

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = keras.Sequential(
            [layers.Dense(dff, activation="relu"), layers.Dense(d_model)]
        )

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask=None):
        # Self-attention with look-ahead mask
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # Cross-attention with encoder output
        attn2, _ = self.mha2(out1, enc_output, enc_output, None)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        # Feed-forward
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3


class Transformer(keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        target_vocab_size,
        max_len,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Encoder
        self.encoder_embedding = layers.Embedding(input_vocab_size, d_model)
        self.encoder_pos = PositionalEncoding(max_len, d_model)
        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]

        # Decoder
        self.decoder_embedding = layers.Embedding(target_vocab_size, d_model)
        self.decoder_pos = PositionalEncoding(max_len, d_model)
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]

        self.final_layer = layers.Dense(target_vocab_size)
        self.dropout = layers.Dropout(dropout_rate)

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def call(self, inputs, training=False):
        inp, tar = inputs

        # Create look-ahead mask for decoder
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])

        # Encoder
        enc = self.encoder_embedding(inp)
        enc *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        enc = self.encoder_pos(enc)
        enc = self.dropout(enc, training=training)

        for layer in self.encoder_layers:
            enc = layer(enc, training, None)

        # Decoder
        dec = self.decoder_embedding(tar)
        dec *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        dec = self.decoder_pos(dec)
        dec = self.dropout(dec, training=training)

        for layer in self.decoder_layers:
            dec = layer(dec, enc, training, look_ahead_mask)

        final_output = self.final_layer(dec)

        return final_output


# step 2 load data
def load_fiji_hindi_data(filepath="data/test_audio/fiji_hindi_data.txt"):
    """
    Load your expanded Fiji Hindi dataset

    You'll replace the file with augmented data!
    """

    print("\nLoading Fiji Hindi translation data...")

    english_sentences = []
    fiji_sentences = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#") or line.startswith("\\"):
                continue

            # Split by |
            if "|" in line:
                parts = line.split("|")
                if len(parts) == 2:
                    eng = parts[0].strip().lower()
                    fiji = parts[1].strip().lower()

                    if eng and fiji:
                        english_sentences.append(eng)
                        fiji_sentences.append(fiji)

    print(f"Loaded {len(english_sentences)} translation pairs")
    print(f"\nFirst 5 examples:")
    for i in range(min(5, len(english_sentences))):
        print(f"  {english_sentences[i]} → {fiji_sentences[i]}")

    print(f"\nDataset size: {len(english_sentences)} pairs")
    if len(english_sentences) < 3000:
        print("Dataset < 3000 pairs. More data = better results!")
    elif len(english_sentences) < 5000:
        print(" Good dataset size! Should work well!")
    else:
        print(" Excellent dataset size! High accuracy expected!")

    return english_sentences, fiji_sentences


# step 3 preprocessing
def create_vocab(sentences):
    """Create vocabulary"""
    words = set()
    for s in sentences:
        words.update(s.split())

    vocab = [PAD, START, END] + sorted(list(words))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for i, w in enumerate(vocab)}

    return word_to_idx, idx_to_word, len(vocab)


def encode_sentence(sentence, word_to_idx, max_len):
    """Encode sentence to indices"""
    words = sentence.split()
    encoded = [word_to_idx.get(w, 0) for w in words]

    if len(encoded) < max_len:
        encoded += [word_to_idx[PAD]] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]

    return encoded


def prepare_data(eng_sentences, fiji_sentences):
    """Prepare training data"""

    print("\nPreparing data...")

    eng_w2i, eng_i2w, eng_vocab = create_vocab(eng_sentences)
    fiji_w2i, fiji_i2w, fiji_vocab = create_vocab(fiji_sentences)

    max_eng_len = max(len(s.split()) for s in eng_sentences)
    max_fiji_len = max(len(s.split()) for s in fiji_sentences) + 2

    print(f"  English vocabulary: {eng_vocab} words")
    print(f"  Fiji Hindi vocabulary: {fiji_vocab} words")
    print(f"  Max English length: {max_eng_len} words")
    print(f"  Max Fiji Hindi length: {max_fiji_len} words")

    encoder_input = np.array(
        [encode_sentence(s, eng_w2i, max_eng_len) for s in eng_sentences]
    )

    decoder_input = np.array(
        [
            encode_sentence(START + " " + s, fiji_w2i, max_fiji_len)
            for s in fiji_sentences
        ]
    )

    decoder_target = np.array(
        [encode_sentence(s + " " + END, fiji_w2i, max_fiji_len) for s in fiji_sentences]
    )

    print(f"\n  Encoder input shape: {encoder_input.shape}")
    print(f"  Decoder input shape: {decoder_input.shape}")
    print(f"  Decoder target shape: {decoder_target.shape}")

    return (
        encoder_input,
        decoder_input,
        decoder_target,
        eng_w2i,
        eng_i2w,
        eng_vocab,
        fiji_w2i,
        fiji_i2w,
        fiji_vocab,
        max_eng_len,
        max_fiji_len,
    )


# training
def train_transformer(
    model, enc_input, dec_input, dec_target, epochs=100, batch_size=64
):
    """Train the transformer"""

    print("\n" + "=" * 70)
    print("TRAINING - M4 Pro Optimized")
    print("=" * 70)

    # Use legacy Adam for M-series Macs
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    # Simple loss - no masking complications!
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    dec_target_exp = np.expand_dims(dec_target, -1)

    # Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
    ]

    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Training samples: {len(enc_input)}")
    print(f"Validation split: 15%")
    print()

    history = model.fit(
        [enc_input, dec_input],
        dec_target_exp,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nTraining complete!")

    return history


# infrance
def translate(text, model, eng_w2i, fiji_w2i, fiji_i2w, max_eng_len, max_fiji_len):
    """Translate English to Fiji Hindi"""

    text = text.lower().strip()
    enc_input = encode_sentence(text, eng_w2i, max_eng_len)
    enc_input = np.array([enc_input])

    dec_input = np.zeros((1, max_fiji_len), dtype="int32")
    dec_input[0, 0] = fiji_w2i[START]

    translated = []

    for i in range(max_fiji_len - 1):
        predictions = model.predict([enc_input, dec_input], verbose=0)
        next_word_idx = np.argmax(predictions[0, i, :])
        next_word = fiji_i2w[next_word_idx]

        if next_word == END:
            break

        if next_word not in [PAD, START]:
            translated.append(next_word)

        if i + 1 < max_fiji_len:
            dec_input[0, i + 1] = next_word_idx

    return " ".join(translated)


# testing
def test_translations(
    model, test_pairs, eng_w2i, fiji_w2i, fiji_i2w, max_eng_len, max_fiji_len
):
    """Test the model"""

    print("\n" + "=" * 70)
    print("TESTING TRANSLATIONS")
    print("=" * 70)

    correct = 0
    partial = 0

    for eng, true_fiji in test_pairs:
        pred_fiji = translate(
            eng, model, eng_w2i, fiji_w2i, fiji_i2w, max_eng_len, max_fiji_len
        )

        # Exact match
        exact_match = pred_fiji.strip() == true_fiji.lower().strip()

        # Partial match (any word correct)
        pred_words = set(pred_fiji.split())
        true_words = set(true_fiji.lower().split())
        partial_match = len(pred_words & true_words) > 0

        if exact_match:
            correct += 1
            icon = "yup"
        elif partial_match:
            partial += 1
            icon = "eh"
        else:
            icon = "yeah na"

        print(f"\n{icon} English: {eng}")
        print(f"   Expected: {true_fiji}")
        print(f"   Got:      {pred_fiji}")

    accuracy = correct / len(test_pairs)
    partial_acc = (correct + partial) / len(test_pairs)

    print(f"\n{'='*70}")
    print(f"Exact Match: {accuracy:.1%} ({correct}/{len(test_pairs)})")
    print(f"Partial Match: {partial_acc:.1%} ({correct + partial}/{len(test_pairs)})")
    print("=" * 70)

    return accuracy, partial_acc


# for testiing if it works out interactive mode
def interactive_translator(
    model, eng_w2i, fiji_w2i, fiji_i2w, max_eng_len, max_fiji_len
):
    """Interactive translation mode"""

    print("\n" + "=" * 70)
    print("INTERACTIVE ENGLISH → FIJI HINDI TRANSLATOR")
    print("=" * 70)
    print("Type English sentences, get Fiji Hindi translation!")
    print("Type 'quit' to exit")
    print("=" * 70)

    while True:
        english = input("\nEnglish: ").strip()

        if english.lower() in ["quit", "exit", "q"]:
            print("Vinaka! (Thank you!)")
            break

        if not english:
            continue

        fiji_hindi = translate(
            english, model, eng_w2i, fiji_w2i, fiji_i2w, max_eng_len, max_fiji_len
        )

        print(f"Fiji Hindi: {fiji_hindi}")


# visuals
def plot_training_history(history):
    """Plot training metrics"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history.history["accuracy"], label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("project_43_training_history.png", dpi=150, bbox_inches="tight")
    print("\n Training plot saved: project_43_training_history.png")
    plt.show()


# main
def main():
    print("\n PROJECT 43: FIJI HINDI TRANSFORMER")
    print("Completing the 37 → 38 → 43 arc!")
    print()

    # Load your augmented dataset
    eng_sentences, fiji_sentences = load_fiji_hindi_data(
        "data/test_audio/fiji_hindi_data.txt"
    )

    # Prepare data
    (
        enc_input,
        dec_input,
        dec_target,
        eng_w2i,
        eng_i2w,
        eng_vocab,
        fiji_w2i,
        fiji_i2w,
        fiji_vocab,
        max_eng_len,
        max_fiji_len,
    ) = prepare_data(eng_sentences, fiji_sentences)

    # Build model - BIGGER for better results this for training
    print("\nBuilding Transformer...")
    model = Transformer(
        num_layers=4,  # 4 layers more than before
        d_model=256,  # 256 dimensions more capacity
        num_heads=8,  # 8 attention heads
        dff=2028,  # Large feed-forward
        input_vocab_size=eng_vocab,
        target_vocab_size=fiji_vocab,
        max_len=max(max_eng_len, max_fiji_len),
        dropout_rate=0.2,
    )

    print(f"\nModel Configuration:")
    print(f"  Layers: 4")
    print(f"  d_model: 256")
    print(f"  Heads: 8")
    print(f"  FFN dimension: 1024")
    print(f"  Total parameters: ~8-10M")

    # Train
    history = train_transformer(
        model,
        enc_input,
        dec_input,
        dec_target,
        epochs=100,  # 100 epochs
        batch_size=64,  # Smaller batch for M4 Pro
    )

    # Plot results
    plot_training_history(history)

    # Test on common words/phrases
    test_pairs = [
        ("hello", "namaste"),
        ("thank you", "dhanyavaad"),
        ("water", "paani"),
        ("mother", "maai"),
        ("father", "baap"),
        ("good morning", "suprabhat"),
        ("how are you", "kaise ho"),
        ("I am fine", "main theek hoon"),
        ("yes", "haan"),
        ("no", "nahin"),
    ]

    # Add some from your training data
    for i in [10, 50, 100, 200, 500]:
        if i < len(eng_sentences):
            test_pairs.append((eng_sentences[i], fiji_sentences[i]))

    accuracy, partial_acc = test_translations(
        model, test_pairs, eng_w2i, fiji_w2i, fiji_i2w, max_eng_len, max_fiji_len
    )

    # Save model
    print("\nSaving model...")
    model.save_weights("project_43_fiji_hindi_transformer.weights.h5")
    print(" Model saved: project_43_fiji_hindi_transformer.weights.h5")

    # Interactive mode
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Exact Match Accuracy: {accuracy:.1%}")
    print(f"Partial Match Accuracy: {partial_acc:.1%}")
    print()

    if accuracy > 0.7:
        print(" Model is working ")
    elif accuracy > 0.5:
        print(" Model learned the patterns")
    elif partial_acc > 0.5:
        print(" Getting right words, needs more training")
    else:
        print(" Needs more data or training. Upload to Colab")

    print("\nReady for interactive translation!")

    # Interactive translator
    interactive_translator(
        model, eng_w2i, fiji_w2i, fiji_i2w, max_eng_len, max_fiji_len
    )

    print("\n" + "=" * 70)
    print("PROJECT 43 COMPLETE!")
    print("\nNext steps:")
    print("1. If accuracy < 70%, upload to Colab for 200-300 epochs")
    print("2. Project 44: Attention Visualization")
    print("3. Project 45: Optimization Techniques")
    print("=" * 70)

    return model, history


# main
if __name__ == "__main__":
    main()
