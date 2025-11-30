import os
import shutil
import pandas as pd


excel_path = r"C:\Users\Win 10\Downloads\archive (2)\esc50.csv"  
AUDIO_FOLDER = r"C:\Users\Win 10\Downloads\archive (2)\audio\audio"           
OUTPUT_FOLDER= r"C:\Users\Win 10\Desktop\sound_alert_feature\dataset"                 



TARGET_LABELS = {
    "crying_baby": "baby_crying",
    "car_horn": "car_horn",
    "door_wood_knock": "door_knock",     
    "glass_breaking": "glass_break",
    "siren": "fire_alarm"
}



df = pd.read_csv(excel_path)
df_filtered = df[df['category'].isin(TARGET_LABELS.keys())]

for _, row in df_filtered.iterrows():
    file_name = row['filename']
    label = row['category']

    source_path = os.path.join(AUDIO_FOLDER, file_name)
    target_label_folder = TARGET_LABELS[label]
    dest_path = os.path.join(OUTPUT_FOLDER, target_label_folder, file_name)

    if os.path.exists(source_path):
        shutil.copy(source_path, dest_path)
        print(f"Copied: {file_name} → {target_label_folder}")
    else:
        print(f"Missing file: {file_name}")

print("\ Done!")