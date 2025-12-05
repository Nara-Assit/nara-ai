import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import soundfile as sf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import logging
from datetime import datetime
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

np.random.seed(42)
tf.random.set_seed(42)

logger.info("="*60)
logger.info("SOUND CLASSIFIER TRAINING WITH AUGMENTATION")
logger.info(f"Training started at: {datetime.now()}")
logger.info("="*60)

logger.info("Creating augmentation pipeline...")
augmentation_pipeline = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.85, max_rate=1.15, p=0.4),
    PitchShift(min_semitones=-3, max_semitones=3, p=0.4),
    Shift(min_shift=-0.4, max_shift=0.4, p=0.5),
])
logger.info("Augmentation pipeline ready")

logger.info("Loading YAMNet model...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
logger.info("YAMNet model loaded successfully")

def augment_audio(audio, sr=16000, num_augmentations=3):
    augmented_samples = []
    for _ in range(num_augmentations):
        try:
            augmented = augmentation_pipeline(samples=audio, sample_rate=sr)
            if np.random.rand() > 0.6:
                noise_level = np.random.uniform(0.002, 0.01)
                noise = np.random.normal(0, noise_level, len(augmented))
                augmented = augmented + noise
            augmented = augmented / (np.max(np.abs(augmented)) + 1e-8)
            augmented_samples.append(augmented)
        except:
            continue
    return augmented_samples

def extract_embedding(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=5.0, mono=True)
        audio = audio.astype(np.float32)
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        scores, embeddings, melspec = yamnet_model(audio)
        embedding = np.mean(embeddings.numpy(), axis=0)
        return embedding, audio
    except Exception as e:
        logger.error(f"Failed to extract embedding from {file_path}: {e}")
        return None, None

DATASET_DIR = "dataset"
TARGET_SAMPLES_PER_CLASS = 280

X, y = [], []
seen_files = set()

if os.path.exists("seen_files.txt"):
    with open("seen_files.txt", "r", encoding="utf-8") as f:
        seen_files = set(line.strip() for line in f)
    logger.info(f"Loaded {len(seen_files)} previously processed files")

logger.info(f"Starting dataset processing from: {DATASET_DIR}")
logger.info(f"Target samples per class: {TARGET_SAMPLES_PER_CLASS}")
logger.info("-"*60)

class_counts = {}
for label in os.listdir(DATASET_DIR):
    label_dir = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    wav_files = [f for f in os.listdir(label_dir) if f.endswith(".wav")]
    class_counts[label] = len(wav_files)

logger.info("Original dataset distribution:")
for label, count in class_counts.items():
    logger.info(f"  {label}: {count} files")

for label in os.listdir(DATASET_DIR):
    label_dir = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    logger.info(f"\nProcessing class: {label}")
    
    original_count = class_counts[label]
    
    if original_count >= TARGET_SAMPLES_PER_CLASS:
        augmentations_per_file = 0
        logger.info(f"  Already has enough samples ({original_count})")
    else:
        needed = TARGET_SAMPLES_PER_CLASS - original_count
        augmentations_per_file = max(0, int(np.ceil(needed / original_count)))
        logger.info(f"  Need {needed} more samples - {augmentations_per_file} augmentations per file")
    
    added_original = 0
    added_augmented = 0
    skipped = 0

    wav_files = [f for f in os.listdir(label_dir) if f.endswith(".wav")]
    
    for fname in tqdm(wav_files, desc=f"  Loading {label}", leave=False):
        file_path = os.path.join(label_dir, fname)
        
        if file_path in seen_files:
            skipped += 1
            continue

        emb, audio = extract_embedding(file_path)
        if emb is None:
            continue

        X.append(emb)
        y.append(label)
        seen_files.add(file_path)
        added_original += 1
        
        if augmentations_per_file > 0 and audio is not None:
            augmented_audios = augment_audio(audio, num_augmentations=augmentations_per_file)
            
            for aug_audio in augmented_audios:
                try:
                    scores, embeddings, melspec = yamnet_model(aug_audio)
                    aug_emb = np.mean(embeddings.numpy(), axis=0)
                    X.append(aug_emb)
                    y.append(label)
                    added_augmented += 1
                except:
                    continue

    logger.info(f"  Original: {added_original} | Augmented: {added_augmented} | Skipped: {skipped}")
    logger.info(f"  Total for {label}: {added_original + added_augmented}")

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    logger.error("No audio files found in dataset folder!")
    raise ValueError("ERROR: No audio files found. Check dataset/ folder.")

logger.info("-"*60)
logger.info(f"Total samples collected: {len(X)}")
logger.info(f"Feature vector shape: {X.shape}")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

classes = list(label_encoder.classes_)
logger.info(f"Classes ({len(classes)}): {classes}")

logger.info("\nFinal dataset distribution:")
for i, cls in enumerate(classes):
    count = np.sum(y_encoded == i)
    percentage = (count / len(X)) * 100
    logger.info(f"  {cls}: {count} samples ({percentage:.1f}%)")

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

logger.info("\nClass weights for balanced training:")
for i, cls in enumerate(classes):
    logger.info(f"  {cls}: {class_weight_dict[i]:.3f}")

logger.info("-"*60)
logger.info("Splitting dataset (80% train, 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

logger.info(f"Training samples: {len(X_train)}")
logger.info(f"Testing samples: {len(X_test)}")

logger.info("-"*60)
logger.info("Building improved model architecture...")

model = models.Sequential([
    layers.Input(shape=(1024,)),
    layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

logger.info("Model architecture:")
model.summary(print_fn=logger.info)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

logger.info("-"*60)
logger.info("Starting training (max 100 epochs with early stopping)...")
logger.info("Using class weights for balanced training")
start_time = datetime.now()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

training_time = (datetime.now() - start_time).total_seconds()
logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

logger.info("-"*60)
logger.info("Evaluating model on test set...")

preds = np.argmax(model.predict(X_test), axis=1)
test_accuracy = np.mean(preds == y_test)

logger.info(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
logger.info("\nClassification Report:")

report = classification_report(y_test, preds, target_names=label_encoder.classes_)
for line in report.split('\n'):
    logger.info(line)

logger.info("\nConfusion Matrix:")
cm = confusion_matrix(y_test, preds)
logger.info("Predicted -->")
logger.info(f"{'Actual':<15} " + " ".join([f"{cls[:8]:>8}" for cls in classes]))
for i, row in enumerate(cm):
    logger.info(f"{classes[i]:<15} " + " ".join([f"{val:>8}" for val in row]))

logger.info("\nPer-class accuracy:")
for i, cls in enumerate(classes):
    class_mask = y_test == i
    if np.sum(class_mask) > 0:
        class_acc = np.mean(preds[class_mask] == y_test[class_mask])
        logger.info(f"  {cls}: {class_acc*100:.2f}%")

logger.info("-"*60)
logger.info("Saving model and artifacts...")

model.save("sound_classifier.h5")
logger.info("Model saved: sound_classifier.h5")

joblib.dump(label_encoder, "label_encoder.pkl")
logger.info("Label encoder saved: label_encoder.pkl")

joblib.dump(scaler, "scaler.pkl")
logger.info("Scaler saved: scaler.pkl")

np.save("X_embeddings.npy", X)
logger.info("Embeddings saved: X_embeddings.npy")

np.save("y_labels.npy", y)
logger.info("Labels saved: y_labels.npy")

with open("seen_files.txt", "w", encoding="utf-8") as f:
    for p in sorted(seen_files):
        f.write(p + "\n")
logger.info(f"Processed files list saved: seen_files.txt ({len(seen_files)} files)")

np.save("training_history.npy", history.history)
logger.info("Training history saved: training_history.npy")

logger.info("="*60)
logger.info("TRAINING COMPLETE!")
logger.info(f"Finished at: {datetime.now()}")
logger.info(f"Total samples used: {len(X)}")
logger.info(f"Final test accuracy: {test_accuracy*100:.2f}%")
logger.info(f"Training time: {training_time/60:.2f} minutes")
logger.info("="*60)

