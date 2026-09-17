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
FREQUENCY_PATCH_SIZE = 32
TIME_PATCH_SIZE = 8
CLIP_DURATION_SECONDS = 5
PAIR_DURATION_SECONDS = 10
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
def find_drum_audio_files_slakh_redux(
    dataset_path,
    set_limit=True,
    maximum_tracks=20
):
    """
    Finds supported audio files recursively.
    """

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_path}"
        )

    track_folders = []
    for folder in dataset_path.iterdir():
    
        if folder.is_dir():
            track_folders.append(folder)
    
            if (
                set_limit
                and len(track_folders) >= maximum_tracks
            ):
                break
    
    track_folders = sorted(track_folders)

    drum_audio_files = []
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

            instrument_class = stem_information.get(
                "inst_class"
            )

            if instrument_class != "Drums":
                continue

            drum_audio_files.append(
                stem_path
            )

    return drum_audio_files

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

def calculate_loudness_db(audio, eps=1e-10):
    """
    Calculates the RMS loudness of an audio sample in dBFS.
    """

    if audio.size == 0:
        return float("-inf")

    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))

    loudness_db = float(20.0 * np.log10(max(rms, eps)))

    return loudness_db

def convert_to_stft_spectrogram(audio):
    """
    Converts audio into real and imaginary STFT components.
    """

    stft = lib.stft(
        y=audio,
        n_fft=FRAMESIZE,
        hop_length=HOPLENGTH
    )

    real = np.real(stft).astype(
        np.float32
    )

    imaginary = np.imag(stft).astype(
        np.float32
    )

    real_tensor = torch.from_numpy(
        real
    )

    imaginary_tensor = torch.from_numpy(
        imaginary
    )

    return real_tensor, imaginary_tensor

def check_stft_tensor_shapes(
    real_tensor,
    imaginary_tensor
):
    """
    Checks the shapes of the real and imaginary STFT tensors.
    """

    print("Real tensor type:", type(real_tensor))
    print("Imaginary tensor type:", type(imaginary_tensor))

    print("Real tensor shape:", real_tensor.shape)
    print("Imaginary tensor shape:", imaginary_tensor.shape)

    if real_tensor.shape != imaginary_tensor.shape:
        print("WARNING: Real and imaginary shapes do not match.")
        return

    n_frequency_bins = real_tensor.shape[0]
    n_time_frames = real_tensor.shape[1]

    print("Number of frequency bins:", n_frequency_bins)
    print("Number of time frames:", n_time_frames)

    print("Real tensor dtype:", real_tensor.dtype)
    print("Imaginary tensor dtype:", imaginary_tensor.dtype)

def convert_stft_to_2d_patch_tokens(
    real_tensor,
    imaginary_tensor,
    frequency_patch_size=FREQUENCY_PATCH_SIZE,
    time_patch_size=TIME_PATCH_SIZE
):
    """
    Converts real and imaginary STFT tensors into 2D time-frequency patch tokens.
    """

    if real_tensor.shape != imaginary_tensor.shape:
        raise ValueError(
            "Real and imaginary tensors must have the same shape."
        )

    # Combine into two channels:
    # [2, frequency_bins, time_frames]
    stft_tensor = torch.stack(
        [
            real_tensor,
            imaginary_tensor
        ],
        dim=0
    )

    _, n_frequency_bins, n_time_frames = stft_tensor.shape

    n_frequency_patches = (
        n_frequency_bins // frequency_patch_size
    )

    n_time_patches = (
        n_time_frames // time_patch_size
    )

    usable_frequency_bins = (
        n_frequency_patches
        * frequency_patch_size
    )

    usable_time_frames = (
        n_time_patches
        * time_patch_size
    )

    # Remove incomplete patches at the edges.
    stft_tensor = stft_tensor[
        :,
        :usable_frequency_bins,
        :usable_time_frames
    ]

    # Reshape STFT tensor into patches
    patches = stft_tensor.reshape(
        2,
        n_frequency_patches,
        frequency_patch_size,
        n_time_patches,
        time_patch_size
    )

    # Permute index to return patches[0,1] as [n_frequency_patches, n_time_patches]
    patches = patches.permute(
        1,
        3,
        0,
        2,
        4
    )

    # Calculate patch dimension real and imaginary: 2 * frequency bins * time frames
    patch_dimension = (
        2
        * frequency_patch_size
        * time_patch_size
    )

    # Reshape patch tensor into tokens
    tokens = patches.reshape(
        n_frequency_patches
        * n_time_patches,
        patch_dimension
    )

    # Add batch dimension
    tokens = tokens.unsqueeze(0)

    return tokens

def check_token_shapes(tokens):

    print("Tokens type:", type(tokens))
    print("Full token tensor shape:", tokens.shape)
    print("Batch size:", tokens.shape[0])
    print("Number of tokens:", tokens.shape[1])
    print("Values per token:", tokens.shape[2])
    print("Torch tensor dtype:", tokens.dtype)
    print("First batch shape:", tokens[0].shape)
    print("First token shape:", tokens[0, 0].shape)

def find_audio_context_target_pairs(
    audio,
    sr,
    clip_duration=CLIP_DURATION_SECONDS
):
    """
    Finds valid start positions for contiguous
    context-target audio pairs.
    """

    clip_length = int(
        sr * clip_duration
    )

    pair_length = (
        clip_length * 2
    )

    pair_start_samples = []

    for start_sample in range(
        0,
        len(audio) - pair_length + 1,
        pair_length
    ):
        pair_start_samples.append(
            start_sample
        )

    statistics = {
        "total_pairs": len(
            pair_start_samples
        )
    }

    return (
        pair_start_samples,
        statistics
    )

def process_audio_context_target_pair(
    audio,
    sr,
    clip_duration=CLIP_DURATION_SECONDS
):
    """
    Splits a contiguous audio segment into a context
    and target, then converts both into STFT tokens.
    """

    clip_length = int(
        sr * clip_duration
    )

    required_length = (
        clip_length * 2
    )

    if len(audio) != required_length:
        raise ValueError(
            f"Expected {required_length} samples, "
            f"but received {len(audio)}."
        )

    context_audio = audio[
        :clip_length
    ]

    target_audio = audio[
        clip_length:
    ]

    context_real, context_imaginary = (
        convert_to_stft_spectrogram(
            context_audio
        )
    )

    target_real, target_imaginary = (
        convert_to_stft_spectrogram(
            target_audio
        )
    )

    context_tokens = (
        convert_stft_to_2d_patch_tokens(
            context_real,
            context_imaginary
        )
    )

    target_tokens = (
        convert_stft_to_2d_patch_tokens(
            target_real,
            target_imaginary
        )
    )

    return (
        context_tokens,
        target_tokens
    )

if __name__ == "__main__":

    slakh_redux_drum_audio_files = (
        find_drum_audio_files_slakh_redux(
            SLAKH2100_REDUX_16K_TRAIN
        )
    )

    print(
        f"Total drum audio files: "
        f"{len(slakh_redux_drum_audio_files)}"
    )

    invalid_count = 0
    check_count = 0

    for drum_audio_file in slakh_redux_drum_audio_files:

        try:
            audio, sr = load_and_validate_audio(
                drum_audio_file
            )

            pair_start_samples, statistics = (
                find_audio_context_target_pairs(
                    audio,
                    sr
                )
            )

            if len(pair_start_samples) == 0:
                raise ValueError(
                    "No complete context-target pairs "
                    "were found."
                )

            # Only inspect the first 5 files.
            if check_count < 5:

                start_sample = (
                    pair_start_samples[0]
                )

                clip_length = int(
                    sr
                    * CLIP_DURATION_SECONDS
                )

                pair_length = (
                    clip_length * 2
                )

                end_sample = (
                    start_sample
                    + pair_length
                )

                audio_pair = audio[
                    start_sample:end_sample
                ]

                context_tokens, target_tokens = (
                    process_audio_context_target_pair(
                        audio_pair,
                        sr
                    )
                )

                print(
                    f"file: "
                    f"{drum_audio_file.name}"
                )

                print(
                    f"sample rate: "
                    f"{sr}"
                )

                print(
                    f"file original length: "
                    f"{len(audio) / sr}"
                )

                print(
                    f"Audio Processing Stats: "
                    f"{statistics}"
                )

                print(
                    f"Pair start samples amount: "
                    f"{len(pair_start_samples)}"
                )

                print(
                    f"Example pair start sample: "
                    f"{start_sample}"
                )

                print(
                    f"Pair duration: "
                    f"{len(audio_pair) / sr}"
                )

                print(
                    "\nContext tokens:"
                )

                check_token_shapes(
                    context_tokens
                )

                print(
                    "\nTarget tokens:"
                )

                check_token_shapes(
                    target_tokens
                )

                print(
                    "=" * 60
                )

                check_count += 1

        except ValueError as error:

            print(
                f"Invalid file: "
                f"{drum_audio_file.name}"
            )

            print(
                f"Reason: {error}"
            )

            invalid_count += 1

            continue

    print(
        f"Invalid file count: "
        f"{invalid_count}"
    )

    print(
        f"Total viable audio files: "
        f"{len(slakh_redux_drum_audio_files) - invalid_count}"
    )