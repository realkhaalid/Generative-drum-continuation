import torch
import numpy as np
import json

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import (
    Dataset,
    DataLoader
)

from stft_tokeniser import (
    STFTTokenizer
)

from construct_dataset import (
    create_slakh_datasets,
    create_test_slakh_dataset,
    SLAKH2100_REDUX_16K_TRAIN,
    SLAKH2100_REDUX_16K_VALIDATION,
    SLAKH2100_REDUX_16K_TEST
)

#Config
TOKENIZER_CHECKPOINT_PATH = (
    "tokenizer_checkpoints/"
    "stft_tokenizer_2_retrained_best.pt"
)

OUTPUT_DIRECTORY = (
    "pretokenized_dataset_retrained"
)

BATCH_SIZE = 8

CODEBOOK_SIZE = 1024
NUMBER_OF_QUANTIZERS = 8

SET_LIMIT = False
MAXIMUM_TRACKS = 100

class PretokenizedDrumDataset(
    Dataset
):
    """
    Loads pre-tokenized context and target
    Residual VQ IDs.

    Stored representation:

        context:
            [number_of_examples, 1248, 8]

        target:
            [number_of_examples, 1248, 8]

    IDs are stored as uint16 on disk and
    converted to torch.long when retrieved.
    """

    def __init__(
        self,
        context_path,
        target_path
    ):

        self.context_path = Path(
            context_path
        )

        self.target_path = Path(
            target_path
        )

        self.context_tokens = np.load(
            self.context_path,
            mmap_mode="r"
        )

        self.target_tokens = np.load(
            self.target_path,
            mmap_mode="r"
        )

        if (
            self.context_tokens.shape
            != self.target_tokens.shape
        ):

            raise ValueError(
                "Context and target token "
                "array shapes must match."
            )

        if (
            self.context_tokens.ndim
            != 3
        ):

            raise ValueError(
                "Pretokenized arrays must have "
                "shape [examples, sequence, quantizers]."
            )

    def __len__(
        self
    ):

        return (
            self.context_tokens.shape[
                0
            ]
        )

    def __getitem__(
        self,
        index
    ):

        context_tokens = np.array(
            self.context_tokens[
                index
            ],
            dtype=np.int64,
            copy=True
        )

        target_tokens = np.array(
            self.target_tokens[
                index
            ],
            dtype=np.int64,
            copy=True
        )

        context_tokens = torch.from_numpy(
            context_tokens
        )

        target_tokens = torch.from_numpy(
            target_tokens
        )

        return (
            context_tokens,
            target_tokens
        )

def load_tokenizer(
    checkpoint_path,
    device
):
    """
    Loads the trained STFT tokenizer and
    freezes all tokenizer parameters.
    """

    tokenizer = (
        STFTTokenizer()
        .to(
            device
        )
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    tokenizer.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    tokenizer.eval()

    for parameter in (
        tokenizer.parameters()
    ):

        parameter.requires_grad = False

    print(
        "\nTokenizer loaded"
    )

    print(
        "=" * 70
    )

    print(
        "Checkpoint:",
        checkpoint_path
    )

    print(
        "Frozen:",
        all(
            not parameter.requires_grad
            for parameter
            in tokenizer.parameters()
        )
    )

    return tokenizer

def convert_patches_to_token_ids(
    tokenizer,
    patches
):
    """
    Converts continuous STFT patches directly
    into Residual VQ token IDs.

    Input:
        [B, S, patch_dimension]

    Output:
        [B, S, number_of_quantizers]
    """

    encoded = (
        tokenizer.encoder(
            patches
        )
    )

    (
        quantized,
        token_ids,
        commitment_loss
    ) = (
        tokenizer.quantizer(
            encoded
        )
    )

    return (
        token_ids
    )

def pretokenize_split(
    dataset,
    tokenizer,
    output_directory,
    split_name,
    batch_size,
    number_of_quantizers,
    codebook_size,
    device
):
    """
    Converts an entire dataset split from
    continuous STFT patches to discrete
    Residual VQ token IDs.
    """

    if len(
        dataset
    ) == 0:

        raise ValueError(
            f"{split_name} dataset is empty."
        )

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        example_context,
        example_target
    ) = (
        dataset[
            0
        ]
    )

    sequence_length = (
        example_context.shape[
            0
        ]
    )

    print(
        f"\n{split_name.capitalize()} split"
    )

    print(
        "=" * 70
    )

    print(
        "Examples:",
        len(
            dataset
        )
    )

    print(
        "STFT patch sequence length:",
        sequence_length
    )

    print(
        "STFT patch dimension:",
        example_context.shape[
            1
        ]
    )

    context_output_path = (
        output_directory
        / f"{split_name}_context_tokens.npy"
    )

    target_output_path = (
        output_directory
        / f"{split_name}_target_tokens.npy"
    )

    context_output = (
        np.lib.format.open_memmap(
            context_output_path,
            mode="w+",
            dtype=np.uint16,
            shape=(
                len(
                    dataset
                ),
                sequence_length,
                number_of_quantizers
            )
        )
    )

    target_output = (
        np.lib.format.open_memmap(
            target_output_path,
            mode="w+",
            dtype=np.uint16,
            shape=(
                len(
                    dataset
                ),
                sequence_length,
                number_of_quantizers
            )
        )
    )

    data_loader = (
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    )

    current_index = 0

    tokenizer.eval()

    with torch.no_grad():

        for (
            context_batch,
            target_batch
        ) in tqdm(
            data_loader,
            desc=(
                f"Pretokenizing "
                f"{split_name}"
            )
        ):

            context_batch = (
                context_batch.to(
                    device=device,
                    dtype=torch.float32
                )
            )

            target_batch = (
                target_batch.to(
                    device=device,
                    dtype=torch.float32
                )
            )

            current_batch_size = (
                context_batch.shape[
                    0
                ]
            )

            # Combine on the batch dimension so
            # the tokenizer only needs one
            # forward operation.
            combined_batch = torch.cat(
                [
                    context_batch,
                    target_batch
                ],
                dim=0
            )

            combined_token_ids = (
                convert_patches_to_token_ids(
                    tokenizer,
                    combined_batch
                )
            )

            context_token_ids = (
                combined_token_ids[
                    :current_batch_size
                ]
            )

            target_token_ids = (
                combined_token_ids[
                    current_batch_size:
                ]
            )

            if (
                context_token_ids.ndim
                != 3
            ):

                raise ValueError(
                    "Context token IDs do not "
                    "have shape [B, S, Q]."
                )

            if (
                target_token_ids.ndim
                != 3
            ):

                raise ValueError(
                    "Target token IDs do not "
                    "have shape [B, S, Q]."
                )

            if (
                context_token_ids.shape[
                    2
                ]
                != number_of_quantizers
            ):

                raise ValueError(
                    "Unexpected number of "
                    "context quantizers."
                )

            if (
                target_token_ids.shape[
                    2
                ]
                != number_of_quantizers
            ):

                raise ValueError(
                    "Unexpected number of "
                    "target quantizers."
                )

            minimum_token = min(
                context_token_ids.min().item(),
                target_token_ids.min().item()
            )

            maximum_token = max(
                context_token_ids.max().item(),
                target_token_ids.max().item()
            )

            if (
                minimum_token < 0
                or maximum_token
                >= codebook_size
            ):

                raise ValueError(
                    "Tokenizer returned token IDs "
                    "outside the codebook range."
                )

            batch_start = (
                current_index
            )

            batch_end = (
                current_index
                + current_batch_size
            )

            context_numpy = (
                context_token_ids
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.uint16,
                    copy=False
                )
            )

            target_numpy = (
                target_token_ids
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.uint16,
                    copy=False
                )
            )

            context_output[
                batch_start:
                batch_end
            ] = (
                context_numpy
            )

            target_output[
                batch_start:
                batch_end
            ] = (
                target_numpy
            )

            current_index = (
                batch_end
            )

    context_output.flush()

    target_output.flush()

    if (
        current_index
        != len(
            dataset
        )
    ):

        raise ValueError(
            "Not all dataset examples "
            "were written."
        )

    print(
        "\nPretokenization complete"
    )

    print(
        "Context file:",
        context_output_path
    )

    print(
        "Target file:",
        target_output_path
    )

    print(
        "Saved examples:",
        current_index
    )

    print(
        "Saved shape:",
        (
            current_index,
            sequence_length,
            number_of_quantizers
        )
    )

    return {
        "context_path":
            str(
                context_output_path
            ),

        "target_path":
            str(
                target_output_path
            ),

        "number_of_examples":
            current_index,

        "sequence_length":
            sequence_length,

        "number_of_quantizers":
            number_of_quantizers
    }

def verify_pretokenized_split(
    context_path,
    target_path,
    codebook_size,
    number_of_quantizers
):
    """
    Checks saved token arrays before they are
    used for Transformer training.
    """

    dataset = (
        PretokenizedDrumDataset(
            context_path=(
                context_path
            ),
            target_path=(
                target_path
            )
        )
    )

    print(
        "\nPretokenized dataset verification"
    )

    print(
        "=" * 70
    )

    print(
        "Examples:",
        len(
            dataset
        )
    )

    (
        context_tokens,
        target_tokens
    ) = (
        dataset[
            0
        ]
    )

    print(
        "Context shape:",
        context_tokens.shape
    )

    print(
        "Target shape:",
        target_tokens.shape
    )

    print(
        "Context dtype:",
        context_tokens.dtype
    )

    print(
        "Target dtype:",
        target_tokens.dtype
    )

    print(
        "Context min/max:",
        context_tokens.min().item(),
        context_tokens.max().item()
    )

    print(
        "Target min/max:",
        target_tokens.min().item(),
        target_tokens.max().item()
    )

    if (
        context_tokens.shape[
            1
        ]
        != number_of_quantizers
    ):

        raise ValueError(
            "Incorrect number of context "
            "quantizers."
        )

    if (
        target_tokens.shape[
            1
        ]
        != number_of_quantizers
    ):

        raise ValueError(
            "Incorrect number of target "
            "quantizers."
        )

    if (
        context_tokens.min()
        < 0
        or context_tokens.max()
        >= codebook_size
    ):

        raise ValueError(
            "Context IDs outside valid range."
        )

    if (
        target_tokens.min()
        < 0
        or target_tokens.max()
        >= codebook_size
    ):

        raise ValueError(
            "Target IDs outside valid range."
        )

    print(
        "Verification passed:",
        True
    )

def calculate_token_statistics(
    context_path,
    target_path,
    codebook_size,
    number_of_quantizers,
    split_name
):

    context_tokens = np.load(
        context_path,
        mmap_mode="r"
    )

    target_tokens = np.load(
        target_path,
        mmap_mode="r"
    )

    if (
        context_tokens.shape
        != target_tokens.shape
    ):

        raise ValueError(
            "Context and target token "
            "shapes must match."
        )

    print(
        f"\n{split_name} Token Statistics"
    )

    print(
        "=" * 70
    )

    print(
        "Context shape:",
        context_tokens.shape
    )

    print(
        "Target shape:",
        target_tokens.shape
    )

    statistics = []

    for quantizer_index in range(
        number_of_quantizers
    ):

        context_quantizer_tokens = (
            context_tokens[
                :,
                :,
                quantizer_index
            ]
            .reshape(
                -1
            )
        )

        target_quantizer_tokens = (
            target_tokens[
                :,
                :,
                quantizer_index
            ]
            .reshape(
                -1
            )
        )

        combined_tokens = np.concatenate(
            [
                context_quantizer_tokens,
                target_quantizer_tokens
            ]
        )

        token_counts = np.bincount(
            combined_tokens,
            minlength=codebook_size
        )

        total_tokens = (
            token_counts.sum()
        )

        used_mask = (
            token_counts > 0
        )

        used_tokens = (
            used_mask.sum()
        )

        unused_tokens = (
            codebook_size
            - used_tokens
        )

        usage_percentage = (
            used_tokens
            / codebook_size
            * 100.0
        )

        probabilities = (
            token_counts[
                used_mask
            ]
            / total_tokens
        )

        entropy = -np.sum(
            probabilities
            * np.log(
                probabilities
            )
        )

        maximum_entropy = (
            np.log(
                codebook_size
            )
        )

        normalized_entropy = (
            entropy
            / maximum_entropy
        )

        perplexity = (
            np.exp(
                entropy
            )
        )

        most_common_token = (
            token_counts.argmax()
        )

        most_common_count = (
            token_counts[
                most_common_token
            ]
        )

        most_common_percentage = (
            most_common_count
            / total_tokens
            * 100.0
        )

        top_10_token_ids = (
            np.argsort(
                token_counts
            )[
                -10:
            ][
                ::-1
            ]
        )

        top_10_token_counts = (
            token_counts[
                top_10_token_ids
            ]
        )

        print(
            f"\nQuantizer {quantizer_index + 1}"
        )

        print(
            "-" * 70
        )

        print(
            "Total token assignments:",
            total_tokens
        )

        print(
            "Used codebook entries:",
            used_tokens
        )

        print(
            "Unused codebook entries:",
            unused_tokens
        )

        print(
            "Codebook usage:",
            f"{usage_percentage:.2f}%"
        )

        print(
            "Entropy:",
            f"{entropy:.4f}"
        )

        print(
            "Normalized entropy:",
            f"{normalized_entropy:.4f}"
        )

        print(
            "Perplexity:",
            f"{perplexity:.2f}"
        )

        print(
            "Most common token:",
            most_common_token
        )

        print(
            "Most common token percentage:",
            f"{most_common_percentage:.2f}%"
        )

        print(
            "Top 10 token IDs:",
            top_10_token_ids.tolist()
        )

        print(
            "Top 10 token counts:",
            top_10_token_counts.tolist()
        )

        statistics.append({
            "quantizer":
                quantizer_index
                + 1,

            "total_tokens":
                int(
                    total_tokens
                ),

            "used_tokens":
                int(
                    used_tokens
                ),

            "unused_tokens":
                int(
                    unused_tokens
                ),

            "usage_percentage":
                float(
                    usage_percentage
                ),

            "entropy":
                float(
                    entropy
                ),

            "normalized_entropy":
                float(
                    normalized_entropy
                ),

            "perplexity":
                float(
                    perplexity
                ),

            "most_common_token":
                int(
                    most_common_token
                ),

            "most_common_percentage":
                float(
                    most_common_percentage
                ),

            "top_10_token_ids":
                top_10_token_ids.tolist(),

            "top_10_token_counts":
                top_10_token_counts.tolist()
        })

    return statistics

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    print(
        "\nCreating original datasets..."
    )

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
                SET_LIMIT
            ),
            maximum_tracks=(
                MAXIMUM_TRACKS
            )
        )
    )

    test_dataset = (
        create_test_slakh_dataset(
            test_path=(
                SLAKH2100_REDUX_16K_TEST
            ),
            set_limit=(
                SET_LIMIT
            ),
            maximum_tracks=(
                MAXIMUM_TRACKS
            )
        )
    )

    print(
        "\nDataset Information"
    )

    print(
        "=" * 70
    )

    print(
        "Training examples:",
        len(
            training_dataset
        )
    )

    print(
        "Validation examples:",
        len(
            validation_dataset
        )
    )

    print(
        "Test examples:",
        len(
            test_dataset
        )
    )

    print(
        "\nDataset Information"
    )

    print(
        "=" * 70
    )

    print(
        "Training examples:",
        len(
            training_dataset
        )
    )

    print(
        "Validation examples:",
        len(
            validation_dataset
        )
    )

    print(
        "Test examples:",
        len(
            test_dataset
        )
    )

    tokenizer = (
        load_tokenizer(
            checkpoint_path=(
                TOKENIZER_CHECKPOINT_PATH
            ),
            device=device
        )
    )

    training_results = (
        pretokenize_split(
            dataset=(
                training_dataset
            ),
            tokenizer=(
                tokenizer
            ),
            output_directory=(
                OUTPUT_DIRECTORY
            ),
            split_name="training",
            batch_size=(
                BATCH_SIZE
            ),
            number_of_quantizers=(
                NUMBER_OF_QUANTIZERS
            ),
            codebook_size=(
                CODEBOOK_SIZE
            ),
            device=device
        )
    )

    validation_results = (
        pretokenize_split(
            dataset=(
                validation_dataset
            ),
            tokenizer=(
                tokenizer
            ),
            output_directory=(
                OUTPUT_DIRECTORY
            ),
            split_name="validation",
            batch_size=(
                BATCH_SIZE
            ),
            number_of_quantizers=(
                NUMBER_OF_QUANTIZERS
            ),
            codebook_size=(
                CODEBOOK_SIZE
            ),
            device=device
        )
    )

    test_results = (
        pretokenize_split(
            dataset=(
                test_dataset
            ),
            tokenizer=(
                tokenizer
            ),
            output_directory=(
                OUTPUT_DIRECTORY
            ),
            split_name="test",
            batch_size=(
                BATCH_SIZE
            ),
            number_of_quantizers=(
                NUMBER_OF_QUANTIZERS
            ),
            codebook_size=(
                CODEBOOK_SIZE
            ),
            device=device
        )
    )

    print(
        "\nVerifying training split"
    )

    verify_pretokenized_split(
        context_path=(
            training_results[
                "context_path"
            ]
        ),
        target_path=(
            training_results[
                "target_path"
            ]
        ),
        codebook_size=(
            CODEBOOK_SIZE
        ),
        number_of_quantizers=(
            NUMBER_OF_QUANTIZERS
        )
    )

    print(
        "\nVerifying validation split"
    )

    verify_pretokenized_split(
        context_path=(
            validation_results[
                "context_path"
            ]
        ),
        target_path=(
            validation_results[
                "target_path"
            ]
        ),
        codebook_size=(
            CODEBOOK_SIZE
        ),
        number_of_quantizers=(
            NUMBER_OF_QUANTIZERS
        )
    )

    print(
        "\nVerifying test split"
    )

    verify_pretokenized_split(
        context_path=(
            test_results[
                "context_path"
            ]
        ),
        target_path=(
            test_results[
                "target_path"
            ]
        ),
        codebook_size=(
            CODEBOOK_SIZE
        ),
        number_of_quantizers=(
            NUMBER_OF_QUANTIZERS
        )
    )

    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "PRETOKENIZATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nSaved files:"
    )

    print(
        "Training context:",
        training_results[
            "context_path"
        ]
    )

    print(
        "Training target:",
        training_results[
            "target_path"
        ]
    )

    print(
        "Validation context:",
        validation_results[
            "context_path"
        ]
    )

    print(
        "Validation target:",
        validation_results[
            "target_path"
        ]
    )

    print(
        "Test context:",
        test_results[
            "context_path"
        ]
    )

    print(
        "Test target:",
        test_results[
            "target_path"
        ]
    )

    training_token_statistics = (
        calculate_token_statistics(
            context_path=(
                training_results[
                    "context_path"
                ]
            ),
            target_path=(
                training_results[
                    "target_path"
                ]
            ),
            codebook_size=(
                CODEBOOK_SIZE
            ),
            number_of_quantizers=(
                NUMBER_OF_QUANTIZERS
            ),
            split_name="Training"
        )
    )

    validation_token_statistics = (
        calculate_token_statistics(
            context_path=(
                validation_results[
                    "context_path"
                ]
            ),
            target_path=(
                validation_results[
                    "target_path"
                ]
            ),
            codebook_size=(
                CODEBOOK_SIZE
            ),
            number_of_quantizers=(
                NUMBER_OF_QUANTIZERS
            ),
            split_name="Validation"
        )
    )

    test_token_statistics = (
        calculate_token_statistics(
            context_path=(
                test_results[
                    "context_path"
                ]
            ),
            target_path=(
                test_results[
                    "target_path"
                ]
            ),
            codebook_size=(
                CODEBOOK_SIZE
            ),
            number_of_quantizers=(
                NUMBER_OF_QUANTIZERS
            ),
            split_name="Test"
        )
    )

    metadata = {
        "tokenizer_checkpoint":
            TOKENIZER_CHECKPOINT_PATH,

        "codebook_size":
            CODEBOOK_SIZE,

        "number_of_quantizers":
            NUMBER_OF_QUANTIZERS,

        "storage_dtype":
            "uint16",

        "training":
            training_results,

        "validation":
            validation_results,

        "test":
            test_results,

        "token_statistics": {
            "training":
                training_token_statistics,

            "validation":
                validation_token_statistics,

            "test":
                test_token_statistics
        }
    }

    metadata_path = (
        Path(
            OUTPUT_DIRECTORY
        )
        / "metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as metadata_file:

        json.dump(
            metadata,
            metadata_file,
            indent=4
        )
