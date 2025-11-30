import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping


np.random.seed(42)
tf.random.set_seed(42)

yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

def extract_embedding(file_path):
    """Extracts YAMNet embeddings and returns the averaged 1024 vector."""
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=5.0, mono=True)
        audio = audio.astype(np.float32)
        audio = audio / (np.max(np.abs(audio)) + 1e-6)

        scores, embeddings, melspec = yamnet_model(audio)
        return np.mean(embeddings.numpy(), axis=0)

    except Exception as e:
        print(f"[ERROR] {file_path} → {e}")
        return None


DATASET_DIR = "dataset"
X, y = [], []
seen_files = set()

if os.path.exists("seen_files.txt"):
    with open("seen_files.txt", "r", encoding="utf-8") as f:
        seen_files = set(line.strip() for line in f)
    print(f"Already processed: {len(seen_files)} files")

print("\nProcessing dataset...")
for label in os.listdir(DATASET_DIR):
    label_dir = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    print(f"\nProcessing class: {label}")
    count = 0

    for fname in os.listdir(label_dir):
        if not fname.endswith(".wav"):
            continue

        file_path = os.path.join(label_dir, fname)
        if file_path in seen_files:
            continue

        emb = extract_embedding(file_path)
        if emb is None:
            continue

        X.append(emb)
        y.append(label)
        seen_files.add(file_path)
        count += 1

    print(f"  Added: {count} new files")

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    raise ValueError("ERROR: No audio files found. Check dataset/ folder.")

print(f"\nTotal samples: {len(X)}")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nClasses:", list(label_encoder.classes_))


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

model = models.Sequential([
    layers.Input(shape=(1024,)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    callbacks=callbacks
)


print("\nClassification Report:")
preds = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, preds, target_names=label_encoder.classes_))

model.save("sound_classifier.h5")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

np.save("X_embeddings.npy", X)
np.save("y_labels.npy", y)

with open("seen_files.txt", "w", encoding="utf-8") as f:
    for p in sorted(seen_files):
        f.write(p + "\n")

print("TRAINING COMPLETE!")
