import librosa as lib
import numpy as np
import soundfile as sf
import torch
import warnings
from pathlib import Path
import yaml

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
LOUDNESS_THRESHOLD_DB = -55.0
TARGET_SAMPLE_RATE = 16000
SOURCE_DURATION_SECONDS = 210
FREQUENCY_PATCH_SIZE = 32
TIME_PATCH_SIZE = 2
CLIP_DURATION_SECONDS = 5
STFT_MEAN = 5.276912607275815e-06
STFT_STANDARD_DEVIATION = 0.7016274969136407
SUPPORTED_EXTENSIONS = {
    ".wav",
    ".flac"
}

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

def normalize_stft(
    stft_tensor,
    mean=STFT_MEAN,
    standard_deviation=STFT_STANDARD_DEVIATION
):
    """
    Standardizes STFT values using statistics
    calculated from the training dataset.
    """

    normalized_stft = (
        stft_tensor
        - mean
    ) / standard_deviation

    return normalized_stft

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

    stft_tensor = normalize_stft(
        stft_tensor
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
        3,
        1,
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

def denormalize_stft(
    normalized_stft,
    mean=STFT_MEAN,
    standard_deviation=STFT_STANDARD_DEVIATION
):

    stft_tensor = (
        normalized_stft
        * standard_deviation
    ) + mean

    return stft_tensor

def convert_2d_patch_tokens_to_stft(
    tokens,
    n_frequency_patches=16,
    frequency_patch_size=FREQUENCY_PATCH_SIZE,
    time_patch_size=TIME_PATCH_SIZE
):
    """
    Converts 2D STFT patch tokens back into
    real and imaginary STFT tensors.
    """

    if tokens.ndim != 2:

        raise ValueError(
            "Expected tokens with shape "
            "[number_of_tokens, token_dimension]."
        )

    number_of_tokens = (
        tokens.shape[0]
    )

    expected_token_dimension = (
        2
        * frequency_patch_size
        * time_patch_size
    )

    if (
        tokens.shape[1]
        != expected_token_dimension
    ):

        raise ValueError(
            "Token dimension does not match "
            "the configured STFT patch size."
        )

    if (
        number_of_tokens
        % n_frequency_patches
        != 0
    ):

        raise ValueError(
            "Number of tokens must be divisible "
            "by the number of frequency patches."
        )

    n_time_patches = (
        number_of_tokens
        // n_frequency_patches
    )

    # Reverse flattened token representation.
    patches = tokens.reshape(
        n_time_patches,
        n_frequency_patches,
        2,
        frequency_patch_size,
        time_patch_size
    )

    # Reverse the permutation performed during tokenization.
    patches = patches.permute(
        2,
        1,
        3,
        0,
        4
    )

    # [2, frequency_bins, time_frames]
    stft_tensor = patches.reshape(
        2,
        n_frequency_patches
        * frequency_patch_size,
        n_time_patches
        * time_patch_size
    )

    # Restore original STFT scale.
    stft_tensor = (
        denormalize_stft(
            stft_tensor
        )
    )

    real_tensor = (
        stft_tensor[0]
    )

    imaginary_tensor = (
        stft_tensor[1]
    )

    return (
        real_tensor,
        imaginary_tensor
    )

def reconstruct_audio_from_stft_tokens(
    tokens,
    sample_rate=TARGET_SAMPLE_RATE
):
    """
    Reconstructs waveform audio from generated
    normalized STFT patch tokens.
    """

    (
        real_tensor,
        imaginary_tensor
    ) = (
        convert_2d_patch_tokens_to_stft(
            tokens
        )
    )

    real = (
        real_tensor
        .detach()
        .cpu()
        .numpy()
    )

    imaginary = (
        imaginary_tensor
        .detach()
        .cpu()
        .numpy()
    )

    complex_stft = (
        real
        + (
            1j
            * imaginary
        )
    )

    # Tokenization currently retains:
    padded_stft = np.zeros(
        (
            513,
            157
        ),
        dtype=np.complex64
    )

    padded_stft[
        :complex_stft.shape[0],
        :complex_stft.shape[1]
    ] = (
        complex_stft
    )

    audio = lib.istft(
        padded_stft,
        hop_length=HOPLENGTH,
        length=(
            sample_rate
            * CLIP_DURATION_SECONDS
        )
    )

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    return audio

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
    clip_duration=CLIP_DURATION_SECONDS,
    loudness_threshold_db=LOUDNESS_THRESHOLD_DB
):
    """
    Finds valid contiguous context-target pairs
    containing audible audio in both clips.
    """

    clip_length = int(
        sr
        * clip_duration
    )

    pair_length = (
        clip_length
        * 2
    )

    pair_start_samples = []

    total_pairs = 0

    rejected_quiet_pairs = 0

    for start_sample in range(
        0,
        len(audio) - pair_length + 1,
        pair_length
    ):

        total_pairs += 1

        context_start = (
            start_sample
        )

        context_end = (
            context_start
            + clip_length
        )

        target_start = (
            context_end
        )

        target_end = (
            target_start
            + clip_length
        )

        context_audio = audio[
            context_start:
            context_end
        ]

        target_audio = audio[
            target_start:
            target_end
        ]

        context_loudness_db = (
            calculate_loudness_db(
                context_audio
            )
        )

        target_loudness_db = (
            calculate_loudness_db(
                target_audio
            )
        )

        if (
            context_loudness_db
            < loudness_threshold_db
            or target_loudness_db
            < loudness_threshold_db
        ):

            rejected_quiet_pairs += 1

            continue

        pair_start_samples.append(
            start_sample
        )

    statistics = {
        "total_pairs":
            total_pairs,

        "valid_pairs":
            len(
                pair_start_samples
            ),

        "rejected_quiet_pairs":
            rejected_quiet_pairs
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

    # Reconstruction testing configuration
    NUMBER_OF_TEST_EXAMPLES = 5

    reconstruction_output_directory = Path(
        "reconstruction_tests"
    )

    reconstruction_output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # Find training drum audio files
    slakh_redux_drum_audio_files = (
        find_drum_audio_files_slakh_redux(
            SLAKH2100_REDUX_16K_TRAIN,
            set_limit=True,
            maximum_tracks=2
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STFT TOKEN RECONSTRUCTION TEST"
    )

    print(
        "=" * 70
    )

    print(
        "\nTraining drum files found:",
        len(
            slakh_redux_drum_audio_files
        )
    )

    example_count = 0

    invalid_file_count = 0

    # Search through training files for valid examples
    for drum_audio_file in (
        slakh_redux_drum_audio_files
    ):

        if (
            example_count
            >= NUMBER_OF_TEST_EXAMPLES
        ):

            break

        try:

            audio, sample_rate = (
                load_and_validate_audio(
                    drum_audio_file
                )
            )

            (
                pair_start_samples,
                statistics
            ) = (
                find_audio_context_target_pairs(
                    audio,
                    sample_rate
                )
            )

            if len(
                pair_start_samples
            ) == 0:

                continue

            # Test valid pairs from this file
            for start_sample in (
                pair_start_samples
            ):

                if (
                    example_count
                    >= NUMBER_OF_TEST_EXAMPLES
                ):

                    break

                clip_length = int(
                    sample_rate
                    * CLIP_DURATION_SECONDS
                )

                pair_length = (
                    clip_length
                    * 2
                )

                end_sample = (
                    start_sample
                    + pair_length
                )

                audio_pair = audio[
                    start_sample:
                    end_sample
                ]

                # Original waveform sections
                original_context_audio = (
                    audio_pair[
                        :clip_length
                    ]
                )

                original_target_audio = (
                    audio_pair[
                        clip_length:
                    ]
                )

                # Convert original pair into model tokens
                (
                    context_tokens,
                    target_tokens
                ) = (
                    process_audio_context_target_pair(
                        audio_pair,
                        sample_rate
                    )
                )

                # Remove temporary batch dimension.
                context_tokens = (
                    context_tokens.squeeze(
                        0
                    )
                )

                target_tokens = (
                    target_tokens.squeeze(
                        0
                    )
                )

                # Reconstruct waveform from tokens
                reconstructed_context_audio = (
                    reconstruct_audio_from_stft_tokens(
                        context_tokens,
                        sample_rate=(
                            sample_rate
                        )
                    )
                )

                reconstructed_target_audio = (
                    reconstruct_audio_from_stft_tokens(
                        target_tokens,
                        sample_rate=(
                            sample_rate
                        )
                    )
                )

                # Create output directory for example
                example_number = (
                    example_count
                    + 1
                )

                example_directory = (
                    reconstruction_output_directory
                    / (
                        f"example_"
                        f"{example_number}"
                    )
                )

                example_directory.mkdir(
                    parents=True,
                    exist_ok=True
                )

                # Save original audio
                sf.write(
                    example_directory
                    / "original_context.wav",
                    original_context_audio,
                    sample_rate,
                    subtype="FLOAT"
                )

                sf.write(
                    example_directory
                    / "original_target.wav",
                    original_target_audio,
                    sample_rate,
                    subtype="FLOAT"
                )

                # Save reconstructed audio
                sf.write(
                    example_directory
                    / "reconstructed_context.wav",
                    reconstructed_context_audio,
                    sample_rate,
                    subtype="FLOAT"
                )

                sf.write(
                    example_directory
                    / "reconstructed_target.wav",
                    reconstructed_target_audio,
                    sample_rate,
                    subtype="FLOAT"
                )

                # Numerical checks
                context_mse = np.mean(
                    (
                        original_context_audio
                        - reconstructed_context_audio
                    ) ** 2
                )

                target_mse = np.mean(
                    (
                        original_target_audio
                        - reconstructed_target_audio
                    ) ** 2
                )

                context_mae = np.mean(
                    np.abs(
                        original_context_audio
                        - reconstructed_context_audio
                    )
                )

                target_mae = np.mean(
                    np.abs(
                        original_target_audio
                        - reconstructed_target_audio
                    )
                )

                context_original_loudness = (
                    calculate_loudness_db(
                        original_context_audio
                    )
                )

                context_reconstructed_loudness = (
                    calculate_loudness_db(
                        reconstructed_context_audio
                    )
                )

                target_original_loudness = (
                    calculate_loudness_db(
                        original_target_audio
                    )
                )

                target_reconstructed_loudness = (
                    calculate_loudness_db(
                        reconstructed_target_audio
                    )
                )

                # Print example information
                print(
                    "\n"
                    + "=" * 70
                )

                print(
                    f"EXAMPLE "
                    f"{example_number}"
                )

                print(
                    "=" * 70
                )

                print(
                    "Audio file:",
                    drum_audio_file
                )

                print(
                    "Pair start time:",
                    (
                        start_sample
                        / sample_rate
                    ),
                    "seconds"
                )

                print(
                    "\nToken shapes"
                )

                print(
                    "Context:",
                    context_tokens.shape
                )

                print(
                    "Target:",
                    target_tokens.shape
                )

                print(
                    "\nOriginal waveform ranges"
                )

                print(
                    "Context min/max:",
                    original_context_audio.min(),
                    original_context_audio.max()
                )

                print(
                    "Target min/max:",
                    original_target_audio.min(),
                    original_target_audio.max()
                )

                print(
                    "\nReconstructed waveform ranges"
                )

                print(
                    "Context min/max:",
                    reconstructed_context_audio.min(),
                    reconstructed_context_audio.max()
                )

                print(
                    "Target min/max:",
                    reconstructed_target_audio.min(),
                    reconstructed_target_audio.max()
                )

                print(
                    "\nLoudness"
                )

                print(
                    "Original context:",
                    f"{context_original_loudness:.2f} dB"
                )

                print(
                    "Reconstructed context:",
                    f"{context_reconstructed_loudness:.2f} dB"
                )

                print(
                    "Original target:",
                    f"{target_original_loudness:.2f} dB"
                )

                print(
                    "Reconstructed target:",
                    f"{target_reconstructed_loudness:.2f} dB"
                )

                print(
                    "\nReconstruction error"
                )

                print(
                    "Context MSE:",
                    context_mse
                )

                print(
                    "Context MAE:",
                    context_mae
                )

                print(
                    "Target MSE:",
                    target_mse
                )

                print(
                    "Target MAE:",
                    target_mae
                )

                print(
                    "\nSaved audio to:"
                )

                print(
                    example_directory
                )

                example_count += 1

        except (
            ValueError,
            TypeError,
            OSError
        ) as error:

            print(
                "\nInvalid file:",
                drum_audio_file.name
            )

            print(
                "Reason:",
                error
            )

            invalid_file_count += 1

    # Final summary
    print(
        "\n"
        + "=" * 70
    )

    print(
        "RECONSTRUCTION TEST SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "Examples tested:",
        example_count
    )

    print(
        "Invalid files:",
        invalid_file_count
    )

    print(
        "Output directory:",
        reconstruction_output_directory
    )