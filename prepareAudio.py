import os
import re
import time
import sounddevice as sd
import scipy.io.wavfile as wav
import pyttsx3

# Your list of characters and digits
# POSSIBLE_KEYS = [
#     'character_10_yna', 'character_11_taamatar', 'character_12_thaa', 'character_13_daa',
#     'character_14_dhaa', 'character_15_adna', 'character_16_tabala', 'character_17_tha',
#     'character_18_da', 'character_19_dha', 'character_1_ka', 'character_20_na',
#     'character_21_pa', 'character_22_pha', 'character_23_ba', 'character_24_bha',
#     'character_25_ma', 'character_26_yaw', 'character_27_ra', 'character_28_la',
#     'character_29_waw', 'character_2_kha', 'character_30_motosaw', 'character_31_petchiryakha',
#     'character_32_patalosaw', 'character_33_ha', 'character_34_chhya', 'character_35_tra',
#     'character_36_gya', 'character_3_ga', 'character_4_gha', 'character_5_kna',
#     'character_6_cha', 'character_7_chha', 'character_8_ja', 'character_9_jha',
#     'digit_0', 'digit_1', 'digit_2', 'digit_3', 'digit_4', 'digit_5',
#     'digit_6', 'digit_7', 'digit_8', 'digit_9'
# ]

POSSIBLE_KEYS = ['digit_0', 'digit_7']


# Create audio folder if not exists
os.makedirs('audio', exist_ok=True)

# Sort function based on the number after underscore
def get_number(label):
    match = re.search(r'_(\d+)', label)
    return int(match.group(1)) if match else 1000

# Separate and sort characters first, then digits
characters = sorted([key for key in POSSIBLE_KEYS if key.startswith('character_')], key=get_number)
digits = sorted([key for key in POSSIBLE_KEYS if key.startswith('digit_')], key=get_number)
sorted_labels = characters + digits

# Setup Text-to-Speech
engine = pyttsx3.init()

# Recording settings
duration = 3  # seconds
sample_rate = 44100

print("🎤 Starting recording session...\n")

# afterSeven = sorted_labels[7:]
for label in sorted_labels:
    readable_label = label.split('_', 2)[-1]  # Get the last part as the spoken name
    print(f"Next: {label}")
    
    # Say the name
    # engine.say(readable_label)
    engine.runAndWait()
    
    print(f"Recording for '{readable_label}' will start in 2 seconds...")
    time.sleep(2)
    print("🎙️ Recording now... Speak!")

    # Record
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    
    # Save file
    filename = os.path.join('audio', f'{label}.wav')
    wav.write(filename, sample_rate, recording)
    print(f"✅ Saved: {filename}\n")
    time.sleep(1)

print("✅ All recordings complete!")
