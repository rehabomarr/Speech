import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import noisereduce as nr
import os
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier

# import warnings
# warnings.filterwarnings("ignore")

import re

RAVDESS_PATH = r"C:\Users\Rehab.Kambal\Desktop\roro\speech recognition\ravedess"
TESS_PATH    = r"C:\Users\Rehab.Kambal\Desktop\roro\speech recognition\tess\TESS Toronto emotional speech set data"
SAVEE_PATH   = r"C:\Users\Rehab.Kambal\Desktop\roro\speech recognition\surreyaudio\ALL"
CREMA_PATH   = "C:/Users/Rehab.Kambal/Desktop/roro/speech recognition/crem-d/AudioWAV"

RAVDESS_MAP = {
    "01": "neutral", "02": "neutral", "03": "happy", "04": "sad",
    "05": "angry", "06": "fear", "07": "disgust", "08": "surprise",
}

SAVEE_MAP = {
    "a": "angry", "d": "disgust", "f": "fear", "h": "happy",
    "n": "neutral", "sa": "sad", "su": "surprise",
}

CREMA_MAP = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}

TESS_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "ps": "surprise",
}
# function to take the audio folder return the result of
#searchin directories and subdirectories "**" for any file ending in .wav  "*.wav"
#os.path.join --> builds the path 
#root/**/*.wav
# glob.glob(...)
# finds files matching that pattern.
# Give me a list of every .wav file inside this folder and all its subfolders.
def _wavs(root):
    return glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)

# RAVDESS
#Go through RAVDESS audio files and figure out the emotion associated with each file.

# def parse_ravdess(root):
#     for f in _wavs(root):
#         try:
#             code = os.path.basename(f).split("-")[2]
#             yield f, RAVDESS_MAP.get(code), "RAVDESS"
#         except:
#             continue

def parse_ravdess(root):

    # Find every WAV file
    files = _wavs(root)

    # Process each file
    for f in files:

        try:
            # Get only the filename
            filename = os.path.basename(f)

            # Split filename
            parts = filename.split("-")

            # Get the emotion code
            code = parts[2]

            # Convert code to emotion
            emotion = RAVDESS_MAP.get(code)

            # Give back file path, emotion, and dataset
            yield f, emotion, "RAVDESS"

        except:
            # If something goes wrong, skip this file
            continue


# files = _wavs(RAVDESS_PATH)

# print(len(files))
# print(files[:5])

# print("Testing RAVDESS...")

# files = _wavs(RAVDESS_PATH)

# print("Number of WAV files:", len(files))

# for file, emotion, dataset in parse_ravdess(RAVDESS_PATH):
#     print(file, emotion, dataset)

def parse_tess(root):
    for f in _wavs(root):
        token = os.path.splitext(os.path.basename(f))[0].split("_")[-1].lower()
        yield f, TESS_MAP.get(token), "TESS"

def parse_savee(root):
    for f in _wavs(root):
        name = os.path.splitext(os.path.basename(f))[0]

        # مثال: DC_a01 -> a
        token = name.split("_")[-1]
        token = re.sub(r"\d+", "", token)

        yield f, SAVEE_MAP.get(token), "SAVEE"


def parse_crema(root):
    for f in _wavs(root):
        parts = os.path.basename(f).split("_")
        if len(parts) < 3:
            continue
        code = parts[2]
        yield f, CREMA_MAP.get(code), "CREMA-D"

def build_metadata():
    rows = []

    for parser, root in [
        (parse_ravdess, RAVDESS_PATH),
        (parse_tess, TESS_PATH),
        (parse_savee, SAVEE_PATH),
        (parse_crema, CREMA_PATH),
    ]:
        data = list(parser(root))
        print(parser.__name__, len(data))
        rows.extend(data)

    print("Total before DataFrame:", len(rows))

    df = pd.DataFrame(rows, columns=["path", "emotion", "source"])

    print("After DataFrame:", df.shape)

    print("Missing emotion:", df["emotion"].isna().sum())

    df = df.dropna(subset=["emotion"])

    print("After dropna:", df.shape)

    missing_files = (~df["path"].apply(os.path.exists)).sum()
    print("Files not found:", missing_files)

    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)

    print("Final:", df.shape)

    return df


df = build_metadata()

print(df.shape)
print(df["source"].value_counts())
print(df["emotion"].value_counts())