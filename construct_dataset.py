import torch
import librosa as lib
import numpy as np

from torch.utils.data import (
    Dataset,
    DataLoader
)

from pathlib import Path
from tqdm import tqdm

from data_processing_pipeline import (
    find_drum_audio_files_slakh_redux,
    load_and_validate_audio,
    find_audio_context_target_pairs,
    process_audio_context_target_pair,
    CLIP_DURATION_SECONDS
)

# Config
BATCH_SIZE = 16

SET_TRACK_LIMIT = False
MAXIMUM_TRACKS = 20

# Slakh2100 Redux 16k
SLAKH2100_REDUX_16K_TRAIN = Path(
    "C:/Uni/YearProject/datasets/"
    "slakh2100_redux_16k/train"
)

SLAKH2100_REDUX_16K_VALIDATION = Path(
    "C:/Uni/YearProject/datasets/"
    "slakh2100_redux_16k/validation"
)

# Dataset
class DrumContinuationDataset(
    Dataset
):
    """
    Lazy-loading dataset for drum audio continuation.
    """

    def __init__(
        self,
        drum_audio_files,
        dataset_source="unknown"
    ):

        self.samples = []

        self.dataset_source = (
            dataset_source
        )

        self.processing_stats = {
            "source_files":
                len(
                    drum_audio_files
                ),

            "invalid_source_files":
                0,

            "total_pairs":
                0,

            "valid_pairs":
                0,

            "rejected_quiet_pairs":
                0,

            "invalid_files":
                []
        }

        self._process_audio_files(
            drum_audio_files
        )

    def _process_audio_files(
        self,
        drum_audio_files
    ):
        """
        Scans source files and stores metadata
        describing each valid context-target pair.

        STFTs and tokens are NOT stored here.
        """

        for audio_file in tqdm(
            drum_audio_files,
            desc=(
                f"Indexing "
                f"{self.dataset_source}"
            )
        ):

            try:

                audio, sample_rate = (
                    load_and_validate_audio(
                        audio_file
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

                # Update pair statistics before checking whether any valid pairs remain.
                self.processing_stats[
                    "total_pairs"
                ] += statistics[
                    "total_pairs"
                ]

                self.processing_stats[
                    "valid_pairs"
                ] += statistics[
                    "valid_pairs"
                ]

                self.processing_stats[
                    "rejected_quiet_pairs"
                ] += statistics[
                    "rejected_quiet_pairs"
                ]

                # No usable pair exists in this file.
                if len(
                    pair_start_samples
                ) == 0:

                    continue

                # Store only lightweight metadata.
                for start_sample in (
                    pair_start_samples
                ):

                    self.samples.append(
                        {
                            "audio_file":
                                Path(
                                    audio_file
                                ),

                            "start_sample":
                                int(
                                    start_sample
                                ),

                            "sample_rate":
                                int(
                                    sample_rate
                                ),

                            "dataset_source":
                                self.dataset_source
                        }
                    )

            except (
                ValueError,
                TypeError,
                OSError
            ) as error:

                self.processing_stats[
                    "invalid_source_files"
                ] += 1

                self.processing_stats[
                    "invalid_files"
                ].append(
                    {
                        "file":
                            str(
                                audio_file
                            ),

                        "reason":
                            str(
                                error
                            )
                    }
                )

    def __len__(
        self
    ):
        """
        Returns the number of indexed
        context-target pairs.
        """

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index
    ):
        """
        Lazily loads and processes one
        context-target pair.
        """

        sample_information = (
            self.samples[
                index
            ]
        )

        sample_rate = (
            sample_information[
                "sample_rate"
            ]
        )

        start_sample = (
            sample_information[
                "start_sample"
            ]
        )

        clip_length = int(
            sample_rate
            * CLIP_DURATION_SECONDS
        )

        pair_length = (
            clip_length
            * 2
        )

        start_time_seconds = (
            start_sample
            / sample_rate
        )

        pair_duration_seconds = (
            pair_length
            / sample_rate
        )

        audio, loaded_sample_rate = (
            lib.load(
                sample_information[
                    "audio_file"
                ],
                sr=sample_rate,
                mono=True,
                offset=(
                    start_time_seconds
                ),
                duration=(
                    pair_duration_seconds
                )
            )
        )

        if loaded_sample_rate != (
            sample_rate
        ):

            raise ValueError(
                "Loaded sample rate does not "
                "match expected sample rate."
            )

        audio = np.asarray(
            audio,
            dtype=np.float32
        )

        if len(
            audio
        ) != pair_length:

            raise ValueError(
                f"Loaded context-target pair "
                f"has incorrect length. "
                f"Expected {pair_length} samples, "
                f"received {len(audio)}."
            )

        (
            context_tokens,
            target_tokens
        ) = (
            process_audio_context_target_pair(
                audio,
                sample_rate
            )
        )

        # Validate context tokens
        if not isinstance(
            context_tokens,
            torch.Tensor
        ):

            raise TypeError(
                "Context tokens must be "
                "a PyTorch tensor."
            )

        if not isinstance(
            target_tokens,
            torch.Tensor
        ):

            raise TypeError(
                "Target tokens must be "
                "a PyTorch tensor."
            )

        if context_tokens.ndim != 3:

            raise ValueError(
                "Expected context token shape "
                "[1, number_of_tokens, "
                "token_dimension], but received "
                f"{context_tokens.shape}."
            )

        if target_tokens.ndim != 3:

            raise ValueError(
                "Expected target token shape "
                "[1, number_of_tokens, "
                "token_dimension], but received "
                f"{target_tokens.shape}."
            )

        if context_tokens.shape[0] != 1:

            raise ValueError(
                "Context token batch "
                "dimension must be 1."
            )

        if target_tokens.shape[0] != 1:

            raise ValueError(
                "Target token batch "
                "dimension must be 1."
            )

        # Remove temporary batch dimension
        context_tokens = (
            context_tokens
            .squeeze(
                0
            )
            .to(
                dtype=torch.float32
            )
        )

        target_tokens = (
            target_tokens
            .squeeze(
                0
            )
            .to(
                dtype=torch.float32
            )
        )

        if (
            context_tokens.shape
            != target_tokens.shape
        ):

            raise ValueError(
                "Context and target token "
                "shapes do not match. "
                f"Context: {context_tokens.shape}, "
                f"Target: {target_tokens.shape}"
            )

        return (
            context_tokens,
            target_tokens
        )

    def return_processing_stats(
        self
    ):
        """
        Returns dataset processing statistics.
        """

        return (
            self.processing_stats
        )

    def return_sample_metadata(
        self,
        index
    ):
        """
        Returns the stored lightweight metadata
        for a sample without loading the audio.
        """

        return dict(
            self.samples[
                index
            ]
        )

# Dataset creation
def create_slakh_datasets(
    training_path,
    validation_path,
    set_limit=SET_TRACK_LIMIT,
    maximum_tracks=MAXIMUM_TRACKS
):
    """
    Creates lazy-loading Slakh training
    and validation datasets.
    """

    training_files = (
        find_drum_audio_files_slakh_redux(
            training_path,
            set_limit=(
                set_limit
            ),
            maximum_tracks=(
                maximum_tracks
            )
        )
    )

    validation_files = (
        find_drum_audio_files_slakh_redux(
            validation_path,
            set_limit=(
                set_limit
            ),
            maximum_tracks=(
                maximum_tracks
            )
        )
    )

    training_dataset = (
        DrumContinuationDataset(
            training_files,
            dataset_source="slakh"
        )
    )

    validation_dataset = (
        DrumContinuationDataset(
            validation_files,
            dataset_source="slakh"
        )
    )

    return (
        training_dataset,
        validation_dataset
    )

# Numerical dataset statistics
def calculate_dataset_statistics(
    dataset
):
    """
    Calculates global mean and standard deviation
    across all context and target token values
    in a dataset.
    """

    if len(
        dataset
    ) == 0:

        raise ValueError(
            "Dataset contains no samples."
        )

    total_sum = 0.0

    total_squared_sum = 0.0

    total_values = 0

    minimum_value = float(
        "inf"
    )

    maximum_value = float(
        "-inf"
    )

    for index in tqdm(
        range(
            len(
                dataset
            )
        ),
        desc=(
            "Calculating dataset statistics"
        )
    ):

        (
            context_tokens,
            target_tokens
        ) = (
            dataset[
                index
            ]
        )

        combined_tokens = torch.cat(
            [
                context_tokens,
                target_tokens
            ],
            dim=0
        )

        combined_tokens = (
            combined_tokens.to(
                dtype=torch.float64
            )
        )

        total_sum += (
            combined_tokens
            .sum()
            .item()
        )

        total_squared_sum += (
            combined_tokens
            .square()
            .sum()
            .item()
        )

        total_values += (
            combined_tokens.numel()
        )

        minimum_value = min(
            minimum_value,
            combined_tokens
            .min()
            .item()
        )

        maximum_value = max(
            maximum_value,
            combined_tokens
            .max()
            .item()
        )

    mean = (
        total_sum
        / total_values
    )

    variance = (
        total_squared_sum
        / total_values
    ) - (
        mean ** 2
    )

    standard_deviation = (
        variance
        ** 0.5
    )

    statistics = {
        "mean":
            mean,

        "standard_deviation":
            standard_deviation,

        "minimum":
            minimum_value,

        "maximum":
            maximum_value,

        "total_values":
            total_values
    }

    return statistics

def calculate_normalized_value_ranges(
    dataset,
    thresholds=(
        3.0,
        6.0,
        8.0,
        12.0
    )
):
    """
    Calculates what percentage of normalized STFT
    values fall within a set of absolute thresholds.
    """

    if len(
        dataset
    ) == 0:

        raise ValueError(
            "Dataset contains no samples."
        )

    threshold_counts = {
        threshold: 0
        for threshold
        in thresholds
    }

    total_values = 0

    for index in tqdm(
        range(
            len(
                dataset
            )
        ),
        desc=(
            "Calculating normalized value ranges"
        )
    ):

        (
            context_tokens,
            target_tokens
        ) = (
            dataset[
                index
            ]
        )

        combined_tokens = torch.cat(
            [
                context_tokens,
                target_tokens
            ],
            dim=0
        )

        absolute_values = (
            combined_tokens.abs()
        )

        total_values += (
            combined_tokens.numel()
        )

        for threshold in (
            thresholds
        ):

            threshold_counts[
                threshold
            ] += (
                absolute_values
                .le(
                    threshold
                )
                .sum()
                .item()
            )

    percentages = {}

    for threshold in (
        thresholds
    ):

        percentages[
            threshold
        ] = (
            threshold_counts[
                threshold
            ]
            / total_values
            * 100.0
        )

    statistics = {
        "total_values":
            total_values,

        "percentages":
            percentages
    }

    return statistics

# Dataset testing
def check_dataset(
    dataset,
    dataset_name
):
    """
    Prints information about one dataset
    and lazily loads its first sample.
    """

    print(
        "\n"
    )

    print(
        "=" * 60
    )

    print(
        f"{dataset_name} Dataset"
    )

    print(
        "=" * 60
    )

    stats = (
        dataset
        .return_processing_stats()
    )

    # Processing statistics
    print(
        f"Source files: "
        f"{stats['source_files']}"
    )

    print(
        f"Invalid source files: "
        f"{stats['invalid_source_files']}"
    )

    print(
        f"Total candidate pairs: "
        f"{stats['total_pairs']}"
    )

    print(
        f"Valid pairs: "
        f"{stats['valid_pairs']}"
    )

    print(
        f"Rejected quiet pairs: "
        f"{stats['rejected_quiet_pairs']}"
    )

    print(
        f"Dataset length: "
        f"{len(dataset)}"
    )

    # Retention/rejection percentages
    if stats[
        "total_pairs"
    ] > 0:

        rejection_percentage = (
            stats[
                "rejected_quiet_pairs"
            ]
            / stats[
                "total_pairs"
            ]
            * 100.0
        )

        retention_percentage = (
            stats[
                "valid_pairs"
            ]
            / stats[
                "total_pairs"
            ]
            * 100.0
        )

        print(
            f"Quiet-pair rejection rate: "
            f"{rejection_percentage:.2f}%"
        )

        print(
            f"Pair retention rate: "
            f"{retention_percentage:.2f}%"
        )

    # Sanity check
    print(
        f"Valid pairs match dataset length: "
        f"{stats['valid_pairs'] == len(dataset)}"
    )

    # Invalid files
    if stats[
        "invalid_files"
    ]:

        print(
            "\nInvalid files:"
        )

        for invalid_file in (
            stats[
                "invalid_files"
            ]
        ):

            print(
                f"{invalid_file['file']}: "
                f"{invalid_file['reason']}"
            )

    if len(
        dataset
    ) == 0:

        print(
            "Dataset contains no "
            "context-target pairs."
        )

        return

    # First sample metadata
    sample_metadata = (
        dataset
        .return_sample_metadata(
            0
        )
    )

    print(
        "\nFirst sample metadata"
    )

    print(
        "-" * 60
    )

    print(
        f"Audio file: "
        f"{sample_metadata['audio_file']}"
    )

    print(
        f"Start sample: "
        f"{sample_metadata['start_sample']}"
    )

    print(
        f"Sample rate: "
        f"{sample_metadata['sample_rate']}"
    )

    print(
        f"Dataset source: "
        f"{sample_metadata['dataset_source']}"
    )

    # Trigger lazy loading
    (
        context_tokens,
        target_tokens
    ) = (
        dataset[
            0
        ]
    )

    print(
        "\nFirst loaded sample"
    )

    print(
        "-" * 60
    )

    print(
        f"Context type: "
        f"{type(context_tokens)}"
    )

    print(
        f"Target type: "
        f"{type(target_tokens)}"
    )

    print(
        f"Context shape: "
        f"{context_tokens.shape}"
    )

    print(
        f"Target shape: "
        f"{target_tokens.shape}"
    )

    print(
        f"Context dtype: "
        f"{context_tokens.dtype}"
    )

    print(
        f"Target dtype: "
        f"{target_tokens.dtype}"
    )

    print(
        f"Number of context tokens: "
        f"{context_tokens.shape[0]}"
    )

    print(
        f"Number of target tokens: "
        f"{target_tokens.shape[0]}"
    )

    print(
        f"Values per context token: "
        f"{context_tokens.shape[1]}"
    )

    print(
        f"Values per target token: "
        f"{target_tokens.shape[1]}"
    )

    print(
        f"Context and target shapes match: "
        f"{context_tokens.shape == target_tokens.shape}"
    )


def check_data_loader(
    data_loader,
    loader_name
):
    """
    Loads and checks one batch from
    a DataLoader.
    """

    print(
        "\n"
    )

    print(
        "=" * 60
    )

    print(
        f"{loader_name} DataLoader"
    )

    print(
        "=" * 60
    )

    if len(
        data_loader.dataset
    ) == 0:

        print(
            "DataLoader contains "
            "no samples."
        )

        return

    (
        context_batch,
        target_batch
    ) = next(
        iter(
            data_loader
        )
    )

    print(
        f"Context batch type: "
        f"{type(context_batch)}"
    )

    print(
        f"Target batch type: "
        f"{type(target_batch)}"
    )

    print(
        f"Context batch shape: "
        f"{context_batch.shape}"
    )

    print(
        f"Target batch shape: "
        f"{target_batch.shape}"
    )

    print(
        f"Context batch dtype: "
        f"{context_batch.dtype}"
    )

    print(
        f"Target batch dtype: "
        f"{target_batch.dtype}"
    )

    print(
        f"Batch size: "
        f"{context_batch.shape[0]}"
    )

    print(
        f"Tokens per context sample: "
        f"{context_batch.shape[1]}"
    )

    print(
        f"Values per context token: "
        f"{context_batch.shape[2]}"
    )

    print(
        f"Tokens per target sample: "
        f"{target_batch.shape[1]}"
    )

    print(
        f"Values per target token: "
        f"{target_batch.shape[2]}"
    )

    print(
        f"Context and target batch "
        f"shapes match: "
        f"{context_batch.shape == target_batch.shape}"
    )


if __name__ == "__main__":

    # Create full training and validation datasets
    (
        training_dataset,
        validation_dataset
    ) = (
        create_slakh_datasets(
            training_path=(
                SLAKH2100_REDUX_16K_TRAIN
            ),
            validation_path=(
                SLAKH2100_REDUX_16K_VALIDATION
            ),
            set_limit=(
                SET_TRACK_LIMIT
            ),
            maximum_tracks=(
                MAXIMUM_TRACKS
            )
        )
    )

    training_statistics = (
        calculate_dataset_statistics(
            training_dataset
        )
    )

    print(
        "\n"
    )

    print(
        "=" * 60
    )

    print(
        "Training Dataset Numerical Statistics"
    )

    print(
        "=" * 60
    )

    print(
        f"Mean: "
        f"{training_statistics['mean']}"
    )

    print(
        f"Standard deviation: "
        f"{training_statistics['standard_deviation']}"
    )

    print(
        f"Minimum: "
        f"{training_statistics['minimum']}"
    )

    print(
        f"Maximum: "
        f"{training_statistics['maximum']}"
    )

    print(
        f"Total values: "
        f"{training_statistics['total_values']}"
    )

    range_statistics = (
        calculate_normalized_value_ranges(
            training_dataset,
            thresholds=(
                3.0,
                6.0,
                8.0,
                12.0
            )
        )
    )

    print(
        "\n"
    )

    print(
        "=" * 60
    )

    print(
        "Normalized STFT Value Distribution"
    )

    print(
        "=" * 60
    )

    for threshold, percentage in (
        range_statistics[
            "percentages"
        ].items()
    ):

        print(
            f"Within ±{threshold}: "
            f"{percentage:.6f}%"
        )

    print(
        f"Total values checked: "
        f"{range_statistics['total_values']}"
    )

    # Check datasets
    check_dataset(
        training_dataset,
        "Training"
    )

    check_dataset(
        validation_dataset,
        "Validation"
    )

    # Create DataLoaders
    training_loader = (
        DataLoader(
            training_dataset,
            batch_size=(
                BATCH_SIZE
            ),
            shuffle=True
        )
    )

    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=(
                BATCH_SIZE
            ),
            shuffle=False
        )
    )

    # Check DataLoaders
    check_data_loader(
        training_loader,
        "Training"
    )

    check_data_loader(
        validation_loader,
        "Validation"
    )