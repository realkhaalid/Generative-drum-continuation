import librosa as lib
import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import yaml
from pathlib import Path

#ignore librosa warnings
warnings.filterwarnings(
    "ignore",
    message="PySoundFile failed.*",
    category=UserWarning
)

warnings.filterwarnings(
    "ignore",
    message="librosa.core.audio.__audioread_load.*",
    category=FutureWarning
)

#Config
FRAMESIZE = 1024
HOPLENGTH = 512
N_MELS = 128
F_MIN = 20
F_MAX = 8000
LOUDNESS_THRESHOLD_DB = -55.0
TARGET_SAMPLE_RATE = 16000
SOURCE_DURATION_SECONDS = 210
PATCH_SIZE = 10
CQT_BINS = 84
BINS_PER_OCTAVE = 12
CLIP_DURATION_SECONDS = 5
SUPPORTED_EXTENSIONS = {
    ".wav",
    ".flac"
}
EXCLUDED_LABELS = []

#Slakh2100_redux_16k
SLAKH2100_REDUX_16K_TEST = Path("C:/Uni/YearProject/datasets/slakh2100_redux_16k/test")
SLAKH2100_REDUX_16K_TRAIN = Path("C:/Uni/YearProject/datasets/slakh2100_redux_16k/train")
SLAKH2100_REDUX_16K_VALIDATION = Path("C:/Uni/YearProject/datasets/slakh2100_redux_16k/validation")

# Retireve .wav and .flac files from datasets
def find_audio_files(
    dataset_path,
    set_limit=True,
    maximum_files=20,
    supervised=False
):
    """
    Finds supported audio files recursively.
    """

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_path}"
        )

    if not supervised:
        track_folders = []
        for folder in dataset_path.iterdir():

            if folder.is_dir():
                track_folders.append(folder)

                if (
                    set_limit
                    and len(track_folders) >= maximum_files
                ):
                    break

        track_folders = sorted(track_folders)

        audio_files = []
        for track_folder in track_folders:
            stems_folder = track_folder / "stems"

            if not stems_folder.exists():
                print(
                    f"Stems folder missing: "
                    f"{track_folder.name}"
                )
                continue

            for stem_file in stems_folder.iterdir():

                if (
                    stem_file.is_file()
                    and stem_file.suffix.lower()
                    in SUPPORTED_EXTENSIONS
                    and not stem_file.name.startswith("._")
                ):
                    audio_files.append(stem_file)

        audio_files = sorted(audio_files)        

        return audio_files

    track_folders = []
    for folder in dataset_path.iterdir():
    
        if folder.is_dir():
            track_folders.append(folder)
    
            if (
                set_limit
                and len(track_folders) >= maximum_files
            ):
                break
    
    track_folders = sorted(track_folders)

    labelled_audio_files = []
    for track_folder in track_folders:
        metadata_path = track_folder / "metadata.yaml"
        stems_folder = track_folder / "stems"

        if not metadata_path.exists():
            print(
                f"Metadata file missing: "
                f"{track_folder.name}"
            )
            continue

        if not stems_folder.exists():
            print(
                f"Stems folder missing: "
                f"{track_folder.name}"
            )
            continue

        try:
            with metadata_path.open(
                "r",
                encoding="utf-8"
            ) as metadata_file:
                metadata = yaml.safe_load(
                    metadata_file
                )

        except (OSError, yaml.YAMLError) as error:
            print(
                f"Could not read metadata for "
                f"{track_folder.name}: {error}"
            )
            continue

        stems_metadata = metadata.get(
            "stems",
            {}
        )

        audio_files = []
        for stem_file in stems_folder.iterdir():
        
            if (
                stem_file.is_file()
                and stem_file.suffix.lower()
                in SUPPORTED_EXTENSIONS
                and not stem_file.name.startswith("._")
            ):
                audio_files.append(stem_file)
        
        audio_files = sorted(audio_files)

        for stem_path in audio_files:
            stem_id = stem_path.stem

            stem_information = stems_metadata.get(
                stem_id
            )

            if stem_information is None:
                print(
                    f"Metadata missing for stem: "
                    f"{track_folder.name}/"
                    f"{stem_path.name}"
                )
                continue

            instrument_label = stem_information.get(
                "inst_class"
            )

            if instrument_label is None:
                print(
                    f"Label missing for stem: "
                    f"{track_folder.name}/"
                    f"{stem_path.name}"
                )
                continue

            if instrument_label in EXCLUDED_LABELS:
                continue

            labelled_audio_files.append(
                (
                    stem_path,
                    instrument_label
                )
            )

    return labelled_audio_files

def load_and_validate_audio(file_path: Path):
    """
    Loads an audio file, converts it to mono, resamples it,
    and checks whether it is corrupt, empty, invalid, too short,
    silent, or excessively quiet.
    """

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}"
        )

    # Corrupt or unreadable files should fail here.
    try:
        audio, sample_rate = lib.load(
            file_path,
            sr=TARGET_SAMPLE_RATE,
            mono=True
        )

    except Exception as error:
        raise ValueError(
            f"Corrupt or unreadable audio file: {error}"
        ) from error

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    # Check that samples were decoded.
    if audio.size == 0:
        raise ValueError(
            "The file contains no audio samples."
        )

    # Reject NaN and infinity values.
    if not np.all(np.isfinite(audio)):
        raise ValueError(
            "The audio contains NaN or infinite values."
        )

    # Check for complete digital silence.
    peak_amplitude = float(
        np.max(np.abs(audio))
    )

    if peak_amplitude == 0.0:
        raise ValueError(
            "The audio file is completely silent."
        )

    return audio, sample_rate

def cut_audio_to_consistent_length(audio, sr):
    duration = len(audio) / sr

    if duration < SOURCE_DURATION_SECONDS / 2:
        raise ValueError(
            f"Audio is only {duration:.2f} seconds long. "
            f"At least {SOURCE_DURATION_SECONDS / 2:.2f} "
            f"seconds is required."
        )

    required_samples = int(
        sr * SOURCE_DURATION_SECONDS
    )

    return audio[:required_samples]

def calculate_loudness_db(audio, eps=1e-10):
    """
    Calculates the RMS loudness of an audio sample in dBFS.
    """

    if audio.size == 0:
        return float("-inf")

    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))

    loudness_db = float(20.0 * np.log10(max(rms, eps)))

    return loudness_db

def convert_to_mel_spectrogram(audio, sr):
    """
    Converts an audio sub-sample into a log-mel spectrogram.
    """

    mel_spectrogram = lib.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=FRAMESIZE,
        hop_length=HOPLENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX
    )

    log_mel_spectrogram = lib.power_to_db(
        mel_spectrogram,
        ref=np.max
    ).astype(np.float32)

    log_mel_spectrogram_tensor = torch.from_numpy(log_mel_spectrogram)

    return log_mel_spectrogram_tensor

def convert_to_stft_spectrogram(audio, sr):
    """
    Converts an audio sub-sample into a log-STFT spectrogram.
    """

    stft = lib.stft(
        y=audio,
        n_fft=FRAMESIZE,
        hop_length=HOPLENGTH
    )

    stft_magnitude = np.abs(stft)

    log_stft_spectrogram = lib.amplitude_to_db(
        stft_magnitude,
        ref=np.max
    ).astype(np.float32)

    log_stft_spectrogram_tensor = torch.from_numpy(
        log_stft_spectrogram
    )

    return log_stft_spectrogram_tensor

def convert_to_cqt_spectrogram(audio, sr):
    """
    Converts an audio sub-sample into a log-CQT spectrogram.
    """

    cqt = lib.cqt(
        y=audio,
        sr=sr,
        hop_length=HOPLENGTH,
        fmin=F_MIN,
        n_bins=CQT_BINS,
        bins_per_octave=BINS_PER_OCTAVE
    )

    cqt_magnitude = np.abs(cqt)

    log_cqt_spectrogram = lib.amplitude_to_db(
        cqt_magnitude,
        ref=np.max
    ).astype(np.float32)

    log_cqt_spectrogram_tensor = torch.from_numpy(
        log_cqt_spectrogram
    )

    return log_cqt_spectrogram_tensor

def convert_to_patches(spectrogram, patch_size=PATCH_SIZE):
    """
    Converts a log-mel spectrogram into patches.
    """

    n_freq_bins, n_time_frames = spectrogram.shape
    n_patches = n_time_frames // patch_size
    usable_time_frames = n_patches * patch_size
    spectrogram = spectrogram[:, :usable_time_frames]
    patches = spectrogram.reshape(n_freq_bins, n_patches, patch_size)
    patches = patches.permute(1, 0, 2)
    patches = patches.reshape(n_patches, n_freq_bins * patch_size)
    patches = patches.unsqueeze(0)
    return patches

def check_patch_shapes(patches):
    print("Patches type:", type(patches))
    print("Full patch tensor shape:", patches.shape)
    print("Batch size:", patches.shape[0])
    print("Number of patches:", patches.shape[1])
    print("Values per patch:", patches.shape[2])
    print("Torch tensor dtype:", patches.dtype)
    print("First batch shape:", patches[0].shape)

def split_audio_into_subsamples(
        audio,
        sr,
        clip_duration=CLIP_DURATION_SECONDS,
        loudness_threshold=LOUDNESS_THRESHOLD_DB,
        data_representation=convert_to_mel_spectrogram
    ):
    """
    Splits audio into complete two-second sub-samples.

    Each sub-sample is:
    1. checked for loudness,
    2. discarded if too quiet,
    3. converted to a spectrogram,
    4. converted into patches.
    """
    sub_sample_length = int(
        sr * clip_duration
    )

    valid_patches = []

    total_sub_samples = 0
    quiet_sub_samples = 0

    for start_sample in range(
        0,
        len(audio),
        sub_sample_length
    ):
        end_sample = (
            start_sample + sub_sample_length
        )

        sub_sample = audio[
            start_sample:end_sample
        ]

        # Ignore an incomplete final sub-sample.
        if len(sub_sample) != sub_sample_length:
            continue

        total_sub_samples += 1

        sub_sample = sub_sample.astype(
            np.float32
        )

        loudness_db = calculate_loudness_db(
            sub_sample
        )

        # Discard sub-samples below the threshold.
        if loudness_db < loudness_threshold:
            quiet_sub_samples += 1
            continue

        spectrogram = data_representation(
            sub_sample,
            sr
        )

        patches = convert_to_patches(
            spectrogram
        )

        valid_patches.append(
            patches
        )

    statistics = {
        "total_sub_samples": total_sub_samples,
        "valid_sub_samples": len(valid_patches),
        "quiet_sub_samples": quiet_sub_samples
    }

    return valid_patches, statistics

if __name__ == "__main__":
    supervised_audio_files = find_audio_files(SLAKH2100_REDUX_16K_TRAIN, supervised=True)
    unsupervised_audio_files = find_audio_files(SLAKH2100_REDUX_16K_TRAIN)
    print(f"Total Supervised files: {len(supervised_audio_files)}")
    print(f"Total Unsupervised files: {len(unsupervised_audio_files)}")
    invalid_count = 0
    check_count = 0
    for audio_file in unsupervised_audio_files:
        try:
            audio, sr = load_and_validate_audio(audio_file)
            audio_cut = cut_audio_to_consistent_length(audio, sr)
            valid_patches, stats = split_audio_into_subsamples(audio_cut, sr)
            if check_count < 5:
                print(f"file: {audio_file.name}")
                print(f"sample rate: {sr}")
                print(f"file original length: {len(audio) / sr}")
                print(f"file new length: {len(audio_cut) / sr}")
                print(f"Valid sample count: {len(valid_patches)}")
                check_patch_shapes(valid_patches[check_count])
                print(f"Audio Processing Stats: {stats}")
                print("\n")
                check_count += 1
            
        except ValueError as error:
            print(f"Invalid file: {audio_file.name}")
            print(f"Reason: {error}")
            invalid_count += 1
            continue

    s_invalid_count = 0
    s_check_count = 0
    for audio_file, label in supervised_audio_files:
        try:
            audio, sr = load_and_validate_audio(audio_file)
            audio_cut = cut_audio_to_consistent_length(audio, sr)
            valid_patches, stats = split_audio_into_subsamples(audio_cut, sr)
            if s_check_count < 5:
                print(f"file: {audio_file.name}")
                print(f"label: {label}")
                print(f"sample rate: {sr}")
                print(f"file original length: {len(audio) / sr}")
                print(f"file new length: {len(audio_cut) / sr}")
                print(f"Valid sample count: {len(valid_patches)}")
                check_patch_shapes(valid_patches[s_check_count])
                print(f"Audio Processing Stats: {stats}")
                print("\n")
                s_check_count += 1
            
        except ValueError as error:
            print(f"Invalid file: {audio_file.name}")
            print(f"Reason: {error}")
            s_invalid_count += 1
            continue

    print(f"Invalid file count: {invalid_count}")
    print(f"Total viable audio files: {len(unsupervised_audio_files) - invalid_count}")
    print(f"Invalid labelled file count: {s_invalid_count}")
    print(f"Total viable labelled audio files: {len(supervised_audio_files) - s_invalid_count}")