import numpy as np

TRAINING_CONTEXT_PATH = (
    "pretokenized_dataset_retrained/"
    "training_context_tokens.npy"
)

TRAINING_TARGET_PATH = (
    "pretokenized_dataset_retrained/"
    "training_target_tokens.npy"
)

VALIDATION_CONTEXT_PATH = (
    "pretokenized_dataset_retrained/"
    "validation_context_tokens.npy"
)

VALIDATION_TARGET_PATH = (
    "pretokenized_dataset_retrained/"
    "validation_target_tokens.npy"
)

TEST_CONTEXT_PATH = (
    "pretokenized_dataset_retrained/"
    "test_context_tokens.npy"
)

TEST_TARGET_PATH = (
    "pretokenized_dataset_retrained/"
    "test_target_tokens.npy"
)

CODEBOOK_SIZE = 1024

NUMBER_OF_QUANTIZERS = 8

# Small smoothing value prevents zero
# probabilities for unseen tokens/transitions.
SMOOTHING_ALPHA = 1e-3

# Process the large memmapped datasets
# in manageable chunks.
CHUNK_SIZE = 256

def load_token_arrays(
    context_path,
    target_path
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

    if (
        context_tokens.ndim
        != 3
    ):

        raise ValueError(
            "Token arrays must have shape "
            "[examples, sequence, quantizers]."
        )

    return (
        context_tokens,
        target_tokens
    )

def calculate_unigram_probabilities(
    training_target_path,
    codebook_size,
    number_of_quantizers,
    smoothing_alpha
):
    """
    Learns:

        P(next_token)

    for every quantizer using only the
    training target tokens.
    """

    training_tokens = np.load(
        training_target_path,
        mmap_mode="r"
    )

    unigram_probabilities = (
        np.zeros(
            (
                number_of_quantizers,
                codebook_size
            ),
            dtype=np.float64
        )
    )

    most_common_tokens = []

    print(
        "\nTraining Unigram Model"
    )

    print(
        "=" * 70
    )

    for quantizer_index in range(
        number_of_quantizers
    ):

        quantizer_tokens = (
            training_tokens[
                :,
                :,
                quantizer_index
            ]
            .reshape(
                -1
            )
        )

        token_counts = np.bincount(
            quantizer_tokens,
            minlength=codebook_size
        ).astype(
            np.float64
        )

        most_common_token = (
            token_counts.argmax()
        )

        most_common_tokens.append(
            int(
                most_common_token
            )
        )

        smoothed_counts = (
            token_counts
            + smoothing_alpha
        )

        probabilities = (
            smoothed_counts
            / smoothed_counts.sum()
        )

        unigram_probabilities[
            quantizer_index
        ] = (
            probabilities
        )

        print(
            f"Q{quantizer_index + 1}: "
            f"most common token = "
            f"{most_common_token}"
        )

    return (
        unigram_probabilities,
        most_common_tokens
    )

def calculate_bigram_probabilities(
    training_context_path,
    training_target_path,
    codebook_size,
    number_of_quantizers,
    smoothing_alpha,
    chunk_size
):
    """
    Learns:

        P(next_token | current_token)

    for every quantizer.

    Training transitions include:

        context internal transitions,
        final context -> first target,
        target internal transitions.

    Examples are handled independently so
    transitions never cross between songs/pairs.
    """

    (
        context_tokens,
        target_tokens
    ) = (
        load_token_arrays(
            training_context_path,
            training_target_path
        )
    )

    number_of_examples = (
        context_tokens.shape[
            0
        ]
    )

    bigram_probabilities = (
        np.zeros(
            (
                number_of_quantizers,
                codebook_size,
                codebook_size
            ),
            dtype=np.float64
        )
    )

    print(
        "\nTraining Bigram Model"
    )

    print(
        "=" * 70
    )

    for quantizer_index in range(
        number_of_quantizers
    ):

        transition_counts = (
            np.zeros(
                (
                    codebook_size,
                    codebook_size
                ),
                dtype=np.int64
            )
        )

        for start_index in range(
            0,
            number_of_examples,
            chunk_size
        ):

            end_index = min(
                start_index
                + chunk_size,
                number_of_examples
            )

            context_chunk = (
                context_tokens[
                    start_index:
                    end_index,
                    :,
                    quantizer_index
                ]
            )

            target_chunk = (
                target_tokens[
                    start_index:
                    end_index,
                    :,
                    quantizer_index
                ]
            )

            full_sequence = np.concatenate(
                [
                    context_chunk,
                    target_chunk
                ],
                axis=1
            )

            current_tokens = (
                full_sequence[
                    :,
                    :-1
                ]
                .reshape(
                    -1
                )
                .astype(
                    np.int64,
                    copy=False
                )
            )

            next_tokens = (
                full_sequence[
                    :,
                    1:
                ]
                .reshape(
                    -1
                )
                .astype(
                    np.int64,
                    copy=False
                )
            )

            flattened_transition_ids = (
                current_tokens
                * codebook_size
                + next_tokens
            )

            chunk_counts = np.bincount(
                flattened_transition_ids,
                minlength=(
                    codebook_size
                    * codebook_size
                )
            )

            chunk_counts = (
                chunk_counts.reshape(
                    codebook_size,
                    codebook_size
                )
            )

            transition_counts += (
                chunk_counts
            )

        smoothed_counts = (
            transition_counts.astype(
                np.float64
            )
            + smoothing_alpha
        )

        row_sums = (
            smoothed_counts.sum(
                axis=1,
                keepdims=True
            )
        )

        probabilities = (
            smoothed_counts
            / row_sums
        )

        bigram_probabilities[
            quantizer_index
        ] = (
            probabilities
        )

        print(
            f"Q{quantizer_index + 1}: "
            f"bigram table complete"
        )

    return bigram_probabilities

def create_target_prediction_pairs(
    context_tokens,
    target_tokens,
    quantizer_index
):
    """
    Creates the exact prediction positions used
    by the Transformer:

        C_last -> T0
        T0     -> T1
        ...
        T(N-2) -> T(N-1)

    This produces exactly target_length
    predictions for every example.
    """

    context_quantizer = (
        context_tokens[
            :,
            :,
            quantizer_index
        ]
    )

    target_quantizer = (
        target_tokens[
            :,
            :,
            quantizer_index
        ]
    )

    last_context_token = (
        context_quantizer[
            :,
            -1:
        ]
    )

    current_tokens = np.concatenate(
        [
            last_context_token,
            target_quantizer[
                :,
                :-1
            ]
        ],
        axis=1
    )

    expected_tokens = (
        target_quantizer
    )

    return (
        current_tokens,
        expected_tokens
    )

def evaluate_unigram_baseline(
    context_path,
    target_path,
    unigram_probabilities,
    most_common_tokens,
    number_of_quantizers,
    split_name
):
    """
    Evaluates:

        P(next_token)

    The hard prediction is the most common
    training token.

    CrossEntropy is calculated from the full
    training unigram probability distribution.
    """

    (
        context_tokens,
        target_tokens
    ) = (
        load_token_arrays(
            context_path,
            target_path
        )
    )

    quantizer_accuracies = []

    quantizer_losses = []

    total_correct = 0

    total_predictions = 0

    print(
        f"\n{split_name} Unigram Baseline"
    )

    print(
        "=" * 70
    )

    for quantizer_index in range(
        number_of_quantizers
    ):

        (
            current_tokens,
            expected_tokens
        ) = (
            create_target_prediction_pairs(
                context_tokens,
                target_tokens,
                quantizer_index
            )
        )

        predicted_token = (
            most_common_tokens[
                quantizer_index
            ]
        )

        correct_predictions = (
            expected_tokens
            == predicted_token
        )

        correct_count = (
            correct_predictions.sum()
        )

        number_of_predictions = (
            correct_predictions.size
        )

        accuracy = (
            correct_count
            / number_of_predictions
        )

        probabilities = (
            unigram_probabilities[
                quantizer_index
            ][
                expected_tokens
            ]
        )

        cross_entropy_loss = (
            -np.log(
                probabilities
            )
            .mean()
        )

        quantizer_accuracies.append(
            float(
                accuracy
            )
        )

        quantizer_losses.append(
            float(
                cross_entropy_loss
            )
        )

        total_correct += (
            correct_count
        )

        total_predictions += (
            number_of_predictions
        )

        print(
            f"Q{quantizer_index + 1}: "
            f"accuracy = "
            f"{accuracy * 100:.3f}% | "
            f"CE = "
            f"{cross_entropy_loss:.6f}"
        )

    overall_accuracy = (
        total_correct
        / total_predictions
    )

    mean_cross_entropy = (
        np.mean(
            quantizer_losses
        )
    )

    print(
        "\nOverall accuracy:",
        f"{overall_accuracy * 100:.3f}%"
    )

    print(
        "Mean CrossEntropy:",
        f"{mean_cross_entropy:.6f}"
    )

    return {
        "accuracy":
            float(
                overall_accuracy
            ),

        "cross_entropy":
            float(
                mean_cross_entropy
            ),

        "quantizer_accuracies":
            quantizer_accuracies,

        "quantizer_losses":
            quantizer_losses
    }

def evaluate_bigram_baseline(
    context_path,
    target_path,
    bigram_probabilities,
    number_of_quantizers,
    split_name,
    chunk_size
):
    """
    Evaluates:

        P(next_token | current_token)

    Hard accuracy uses the most likely next
    token for each current token.

    CrossEntropy uses the complete bigram
    probability distribution.
    """

    (
        context_tokens,
        target_tokens
    ) = (
        load_token_arrays(
            context_path,
            target_path
        )
    )

    number_of_examples = (
        context_tokens.shape[
            0
        ]
    )

    quantizer_accuracies = []

    quantizer_losses = []

    total_correct = 0

    total_predictions = 0

    print(
        f"\n{split_name} Bigram Baseline"
    )

    print(
        "=" * 70
    )

    for quantizer_index in range(
        number_of_quantizers
    ):

        transition_probabilities = (
            bigram_probabilities[
                quantizer_index
            ]
        )

        most_likely_next_tokens = (
            transition_probabilities.argmax(
                axis=1
            )
        )

        quantizer_correct = 0

        quantizer_predictions = 0

        negative_log_likelihood_sum = 0.0

        for start_index in range(
            0,
            number_of_examples,
            chunk_size
        ):

            end_index = min(
                start_index
                + chunk_size,
                number_of_examples
            )

            context_chunk = (
                context_tokens[
                    start_index:
                    end_index
                ]
            )

            target_chunk = (
                target_tokens[
                    start_index:
                    end_index
                ]
            )

            (
                current_tokens,
                expected_tokens
            ) = (
                create_target_prediction_pairs(
                    context_chunk,
                    target_chunk,
                    quantizer_index
                )
            )

            predicted_tokens = (
                most_likely_next_tokens[
                    current_tokens
                ]
            )

            correct_predictions = (
                predicted_tokens
                == expected_tokens
            )

            quantizer_correct += (
                correct_predictions.sum()
            )

            quantizer_predictions += (
                correct_predictions.size
            )

            actual_probabilities = (
                transition_probabilities[
                    current_tokens,
                    expected_tokens
                ]
            )

            negative_log_likelihood_sum += (
                -np.log(
                    actual_probabilities
                )
                .sum()
            )

        accuracy = (
            quantizer_correct
            / quantizer_predictions
        )

        cross_entropy_loss = (
            negative_log_likelihood_sum
            / quantizer_predictions
        )

        quantizer_accuracies.append(
            float(
                accuracy
            )
        )

        quantizer_losses.append(
            float(
                cross_entropy_loss
            )
        )

        total_correct += (
            quantizer_correct
        )

        total_predictions += (
            quantizer_predictions
        )

        print(
            f"Q{quantizer_index + 1}: "
            f"accuracy = "
            f"{accuracy * 100:.3f}% | "
            f"CE = "
            f"{cross_entropy_loss:.6f}"
        )

    overall_accuracy = (
        total_correct
        / total_predictions
    )

    mean_cross_entropy = (
        np.mean(
            quantizer_losses
        )
    )

    print(
        "\nOverall accuracy:",
        f"{overall_accuracy * 100:.3f}%"
    )

    print(
        "Mean CrossEntropy:",
        f"{mean_cross_entropy:.6f}"
    )

    return {
        "accuracy":
            float(
                overall_accuracy
            ),

        "cross_entropy":
            float(
                mean_cross_entropy
            ),

        "quantizer_accuracies":
            quantizer_accuracies,

        "quantizer_losses":
            quantizer_losses
    }

if __name__ == "__main__":

    print(
        "\n"
    )

    print(
        "#" * 70
    )

    print(
        "TOKEN PREDICTION BASELINES"
    )

    print(
        "#" * 70
    )

    # 1. Train unigram probability model
    (
        unigram_probabilities,
        most_common_tokens
    ) = (
        calculate_unigram_probabilities(
            training_target_path=(
                TRAINING_TARGET_PATH
            ),
            codebook_size=(
                CODEBOOK_SIZE
            ),
            number_of_quantizers=(
                NUMBER_OF_QUANTIZERS
            ),
            smoothing_alpha=(
                SMOOTHING_ALPHA
            )
        )
    )

    # 2. Train bigram probability model
    bigram_probabilities = (
        calculate_bigram_probabilities(
            training_context_path=(
                TRAINING_CONTEXT_PATH
            ),
            training_target_path=(
                TRAINING_TARGET_PATH
            ),
            codebook_size=(
                CODEBOOK_SIZE
            ),
            number_of_quantizers=(
                NUMBER_OF_QUANTIZERS
            ),
            smoothing_alpha=(
                SMOOTHING_ALPHA
            ),
            chunk_size=(
                CHUNK_SIZE
            )
        )
    )

    # 3. Training baselines
    training_unigram_results = (
        evaluate_unigram_baseline(
            context_path=(
                TRAINING_CONTEXT_PATH
            ),
            target_path=(
                TRAINING_TARGET_PATH
            ),
            unigram_probabilities=(
                unigram_probabilities
            ),
            most_common_tokens=(
                most_common_tokens
            ),
            number_of_quantizers=(
                NUMBER_OF_QUANTIZERS
            ),
            split_name="Training"
        )
    )

    training_bigram_results = (
        evaluate_bigram_baseline(
            context_path=(
                TRAINING_CONTEXT_PATH
            ),
            target_path=(
                TRAINING_TARGET_PATH
            ),
            bigram_probabilities=(
                bigram_probabilities
            ),
            number_of_quantizers=(
                NUMBER_OF_QUANTIZERS
            ),
            split_name="Training",
            chunk_size=(
                CHUNK_SIZE
            )
        )
    )

    # 4. Validation baselines
    validation_unigram_results = (
        evaluate_unigram_baseline(
            context_path=(
                VALIDATION_CONTEXT_PATH
            ),
            target_path=(
                VALIDATION_TARGET_PATH
            ),
            unigram_probabilities=(
                unigram_probabilities
            ),
            most_common_tokens=(
                most_common_tokens
            ),
            number_of_quantizers=(
                NUMBER_OF_QUANTIZERS
            ),
            split_name="Validation"
        )
    )

    validation_bigram_results = (
        evaluate_bigram_baseline(
            context_path=(
                VALIDATION_CONTEXT_PATH
            ),
            target_path=(
                VALIDATION_TARGET_PATH
            ),
            bigram_probabilities=(
                bigram_probabilities
            ),
            number_of_quantizers=(
                NUMBER_OF_QUANTIZERS
            ),
            split_name="Validation",
            chunk_size=(
                CHUNK_SIZE
            )
        )
    )

    # 5. Test baselines
    test_unigram_results = (
        evaluate_unigram_baseline(
            context_path=(
                TEST_CONTEXT_PATH
            ),
            target_path=(
                TEST_TARGET_PATH
            ),
            unigram_probabilities=(
                unigram_probabilities
            ),
            most_common_tokens=(
                most_common_tokens
            ),
            number_of_quantizers=(
                NUMBER_OF_QUANTIZERS
            ),
            split_name="Test"
        )
    )

    test_bigram_results = (
        evaluate_bigram_baseline(
            context_path=(
                TEST_CONTEXT_PATH
            ),
            target_path=(
                TEST_TARGET_PATH
            ),
            bigram_probabilities=(
                bigram_probabilities
            ),
            number_of_quantizers=(
                NUMBER_OF_QUANTIZERS
            ),
            split_name="Test",
            chunk_size=(
                CHUNK_SIZE
            )
        )
    )

    # 6. Summary
    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "BASELINE SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "\nTraining"
    )

    print(
        "Unigram accuracy:",
        f"{training_unigram_results['accuracy'] * 100:.3f}%"
    )

    print(
        "Unigram CE:",
        f"{training_unigram_results['cross_entropy']:.6f}"
    )

    print(
        "Bigram accuracy:",
        f"{training_bigram_results['accuracy'] * 100:.3f}%"
    )

    print(
        "Bigram CE:",
        f"{training_bigram_results['cross_entropy']:.6f}"
    )

    print(
        "\nValidation"
    )

    print(
        "Unigram accuracy:",
        f"{validation_unigram_results['accuracy'] * 100:.3f}%"
    )

    print(
        "Unigram CE:",
        f"{validation_unigram_results['cross_entropy']:.6f}"
    )

    print(
        "Bigram accuracy:",
        f"{validation_bigram_results['accuracy'] * 100:.3f}%"
    )

    print(
        "Bigram CE:",
        f"{validation_bigram_results['cross_entropy']:.6f}"
    )

    print(
        "\nTest"
    )

    print(
        "Unigram accuracy:",
        f"{test_unigram_results['accuracy'] * 100:.3f}%"
    )

    print(
        "Unigram CE:",
        f"{test_unigram_results['cross_entropy']:.6f}"
    )

    print(
        "Bigram accuracy:",
        f"{test_bigram_results['accuracy'] * 100:.3f}%"
    )

    print(
        "Bigram CE:",
        f"{test_bigram_results['cross_entropy']:.6f}"
    )