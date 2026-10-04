import torch

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.utils.data import (
    DataLoader,
    Subset
)

import soundfile as sf

from autoregressive_transformer import (
    AutoregressiveDrumTransformer
)

from pretokenise_dataset import (
    PretokenizedDrumDataset
)

from stft_tokeniser import (
    STFTTokenizer
)

from data_processing_pipeline import (
    reconstruct_audio_from_stft_tokens,
    TARGET_SAMPLE_RATE
)


# Dataset paths
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

# Tokenizer configuration
TOKENIZER_CHECKPOINT_PATH = (
    "tokenizer_checkpoints/"
    "stft_tokenizer_2_retrained_best.pt"
)

# Transformer configuration
CODEBOOK_SIZE = 1024
NUMBER_OF_QUANTIZERS = 8
EMBEDDING_DIMENSION = 256
NUMBER_OF_HEADS = 8
NUMBER_OF_LAYERS = 4
FEED_FORWARD_DIMENSION = 1024
DROPOUT = 0.1
N_FREQUENCY_PATCHES = 16

# Training configuration
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
PATIENCE = 10
CHECKPOINT_INTERVAL = 5
NUMBER_OF_GENERATION_EXAMPLES = 5

CHECKPOINT_DIRECTORY = (
    "transformer_checkpoints_retrained"
)

EXAMPLE_OUTPUT_DIRECTORY = (
    "generated_examples_retrained"
)

MODEL_NAME = (
    "autoregressive_drum_transformer_discrete_retrained"
)

SET_LIMIT = False
MAXIMUM_TRAINING_EXAMPLES = 100
MAXIMUM_VALIDATION_EXAMPLES = 50


# Training Pipeline
class TrainModelPipeline:

    def __init__(
        self,
        tokenizer_checkpoint_path,
        codebook_size=CODEBOOK_SIZE,
        number_of_quantizers=NUMBER_OF_QUANTIZERS,
        embedding_dimension=EMBEDDING_DIMENSION,
        number_of_heads=NUMBER_OF_HEADS,
        number_of_layers=NUMBER_OF_LAYERS,
        feed_forward_dimension=FEED_FORWARD_DIMENSION,
        dropout=DROPOUT,
        n_frequency_patches=N_FREQUENCY_PATCHES,
        gradient_clip_norm=GRADIENT_CLIP_NORM,
        weight_decay=WEIGHT_DECAY,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        checkpoint_directory=CHECKPOINT_DIRECTORY,
        example_output_directory=EXAMPLE_OUTPUT_DIRECTORY,
        device=None
    ):

        if checkpoint_interval < 1:

            raise ValueError(
                "checkpoint_interval must "
                "be at least 1."
            )

        if gradient_clip_norm <= 0:

            raise ValueError(
                "gradient_clip_norm must "
                "be greater than 0."
            )

        self.tokenizer_checkpoint_path = Path(
            tokenizer_checkpoint_path
        )

        self.codebook_size = (
            codebook_size
        )

        self.number_of_quantizers = (
            number_of_quantizers
        )

        self.embedding_dimension = (
            embedding_dimension
        )

        self.number_of_heads = (
            number_of_heads
        )

        self.number_of_layers = (
            number_of_layers
        )

        self.feed_forward_dimension = (
            feed_forward_dimension
        )

        self.dropout = (
            dropout
        )

        self.n_frequency_patches = (
            n_frequency_patches
        )

        self.gradient_clip_norm = (
            gradient_clip_norm
        )

        self.weight_decay = (
            weight_decay
        )

        self.checkpoint_interval = (
            checkpoint_interval
        )

        self.checkpoint_directory = Path(
            checkpoint_directory
        )

        self.checkpoint_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.example_output_directory = Path(
            example_output_directory
        )

        self.example_output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = torch.device(
                device
            )

        # Tokenizer is NOT used during training.
        # It is loaded only so discrete token IDs
        # can be decoded into audio examples.
        self.tokenizer = (
            self.load_tokenizer()
        )

    def load_tokenizer(
        self
    ):

        tokenizer = (
            STFTTokenizer()
            .to(
                self.device
            )
        )

        checkpoint = torch.load(
            self.tokenizer_checkpoint_path,
            map_location=self.device,
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
            "\nTokenizer loaded for "
            "checkpoint reconstruction"
        )

        print(
            "=" * 70
        )

        print(
            "Checkpoint:",
            self.tokenizer_checkpoint_path
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

    def create_model(
        self
    ):

        model = (
            AutoregressiveDrumTransformer(
                codebook_size=(
                    self.codebook_size
                ),
                number_of_quantizers=(
                    self.number_of_quantizers
                ),
                embedding_dimension=(
                    self.embedding_dimension
                ),
                number_of_heads=(
                    self.number_of_heads
                ),
                number_of_layers=(
                    self.number_of_layers
                ),
                feed_forward_dimension=(
                    self.feed_forward_dimension
                ),
                dropout=(
                    self.dropout
                ),
                n_frequency_patches=(
                    self.n_frequency_patches
                )
            )
        )

        model = (
            model.to(
                self.device
            )
        )

        return model

    def create_optimizer(
        self,
        model,
        learning_rate
    ):

        optimizer = (
            torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=(
                    self.weight_decay
                )
            )
        )

        return optimizer

    def create_loss_function(
        self
    ):

        return (
            torch.nn.CrossEntropyLoss()
        )

    def prepare_autoregressive_batch(
        self,
        context_tokens,
        target_tokens
    ):

        if (
            context_tokens.ndim != 3
            or target_tokens.ndim != 3
        ):

            raise ValueError(
                "Context and target must "
                "have shape [B, S, Q]."
            )

        if (
            context_tokens.shape
            != target_tokens.shape
        ):

            raise ValueError(
                "Context and target shapes "
                "must match."
            )

        context_length = (
            context_tokens.shape[
                1
            ]
        )

        full_sequence = torch.cat(
            [
                context_tokens,
                target_tokens
            ],
            dim=1
        )

        model_input = (
            full_sequence[
                :,
                :-1,
                :
            ]
        )

        expected_output = (
            full_sequence[
                :,
                1:,
                :
            ]
        )

        return (
            model_input,
            expected_output,
            context_length
        )

    def select_target_continuation(
        self,
        logits,
        expected_output,
        context_length
    ):

        target_logits = (
            logits[
                :,
                context_length - 1:,
                :,
                :
            ]
        )

        target_expected = (
            expected_output[
                :,
                context_length - 1:,
                :
            ]
        )

        return (
            target_logits,
            target_expected
        )

    def calculate_loss(
        self,
        target_logits,
        target_expected,
        loss_function
    ):

        quantizer_losses = []

        for quantizer_index in range(
            self.number_of_quantizers
        ):

            quantizer_logits = (
                target_logits[
                    :,
                    :,
                    quantizer_index,
                    :
                ]
            )

            quantizer_targets = (
                target_expected[
                    :,
                    :,
                    quantizer_index
                ]
            )

            quantizer_loss = (
                loss_function(
                    quantizer_logits.reshape(
                        -1,
                        self.codebook_size
                    ),
                    quantizer_targets.reshape(
                        -1
                    )
                )
            )

            quantizer_losses.append(
                quantizer_loss
            )

        quantizer_losses = (
            torch.stack(
                quantizer_losses
            )
        )

        total_loss = (
            quantizer_losses.mean()
        )

        return (
            total_loss,
            quantizer_losses
        )

    def calculate_accuracy(
        self,
        target_logits,
        target_expected
    ):

        predicted_tokens = (
            torch.argmax(
                target_logits,
                dim=-1
            )
        )

        accuracy = (
            (
                predicted_tokens
                == target_expected
            )
            .float()
            .mean()
            .item()
        )

        return accuracy

    def decode_token_ids(
        self,
        token_ids
    ):
        """
        Converts RVQ token IDs:

            [S, Q]
            or
            [B, S, Q]

        into reconstructed STFT patches:

            [S, patch_dimension]
        """

        if token_ids.ndim == 2:

            token_ids = (
                token_ids.unsqueeze(
                    0
                )
            )

        token_ids = (
            token_ids.to(
                device=self.device,
                dtype=torch.long
            )
        )

        self.tokenizer.eval()

        with torch.no_grad():

            quantized_latents = (
                self.tokenizer
                .quantizer
                .get_output_from_indices(
                    token_ids
                )
            )

            reconstructed_patches = (
                self.tokenizer.decoder(
                    quantized_latents
                )
            )

        reconstructed_patches = (
            reconstructed_patches[
                0
            ]
            .detach()
            .cpu()
        )

        return reconstructed_patches

    def save_reference_examples(
        self,
        example_context_tokens,
        example_target_tokens
    ):

        print(
            "\nSaving reference examples"
        )

        print(
            "=" * 70
        )

        for example_index in range(
            len(
                example_context_tokens
            )
        ):

            context_token_ids = (
                example_context_tokens[
                    example_index
                ]
            )

            target_token_ids = (
                example_target_tokens[
                    example_index
                ]
            )

            context_patches = (
                self.decode_token_ids(
                    context_token_ids
                )
            )

            target_patches = (
                self.decode_token_ids(
                    target_token_ids
                )
            )

            context_audio = (
                reconstruct_audio_from_stft_tokens(
                    context_patches
                )
            )

            target_audio = (
                reconstruct_audio_from_stft_tokens(
                    target_patches
                )
            )

            context_path = (
                self.example_output_directory
                / (
                    f"example_"
                    f"{example_index + 1}_"
                    f"context.wav"
                )
            )

            target_path = (
                self.example_output_directory
                / (
                    f"example_"
                    f"{example_index + 1}_"
                    f"target.wav"
                )
            )

            sf.write(
                context_path,
                context_audio,
                TARGET_SAMPLE_RATE
            )

            sf.write(
                target_path,
                target_audio,
                TARGET_SAMPLE_RATE
            )

            print(
                f"Example {example_index + 1}"
            )

            print(
                "Context:",
                context_path
            )

            print(
                "Target:",
                target_path
            )

    def save_generated_example(
        self,
        model,
        context_token_ids,
        number_of_target_tokens,
        epoch,
        model_name,
        example_index
    ):
        """
        Generates an autoregressive continuation,
        decodes the generated RVQ IDs using the
        frozen tokenizer and saves a WAV file.
        """

        model.eval()

        context_token_ids = (
            context_token_ids.to(
                device=self.device,
                dtype=torch.long
            )
        )

        if context_token_ids.ndim == 2:

            context_token_ids = (
                context_token_ids.unsqueeze(
                    0
                )
            )

        print(
            f"\nGenerating checkpoint example "
            f"{example_index + 1}..."
        )

        generated_token_ids = (
            model.generate_continuation(
                context_token_ids=(
                    context_token_ids
                ),
                number_of_target_tokens=(
                    number_of_target_tokens
                )
            )
        )

        print(
            "Generated token shape:",
            generated_token_ids.shape
        )

        generated_patches = (
            self.decode_token_ids(
                generated_token_ids
            )
        )

        print(
            "Decoded STFT patch shape:",
            generated_patches.shape
        )

        generated_audio = (
            reconstruct_audio_from_stft_tokens(
                generated_patches
            )
        )

        file_name = (
            f"{model_name}_"
            f"epoch_{epoch}_"
            f"example_{example_index + 1}_"
            f"generated.wav"
        )

        file_path = (
            self.example_output_directory
            / file_name
        )

        sf.write(
            file_path,
            generated_audio,
            TARGET_SAMPLE_RATE
        )

        print(
            "Generated example:",
            file_path
        )

        return file_path

    def train_epoch(
        self,
        model,
        data_loader,
        optimizer,
        loss_function
    ):

        model.train()

        total_loss = 0.0

        total_accuracy = 0.0

        quantizer_loss_totals = (
            torch.zeros(
                self.number_of_quantizers,
                dtype=torch.float64
            )
        )

        number_of_batches = 0

        for (
            context_tokens,
            target_tokens
        ) in tqdm(
            data_loader,
            desc="Training Transformer",
            leave=False
        ):

            context_tokens = (
                context_tokens.to(
                    device=self.device,
                    dtype=torch.long
                )
            )

            target_tokens = (
                target_tokens.to(
                    device=self.device,
                    dtype=torch.long
                )
            )

            (
                model_input,
                expected_output,
                context_length
            ) = (
                self.prepare_autoregressive_batch(
                    context_tokens,
                    target_tokens
                )
            )

            optimizer.zero_grad()

            logits = (
                model(
                    model_input
                )
            )

            (
                target_logits,
                target_expected
            ) = (
                self.select_target_continuation(
                    logits,
                    expected_output,
                    context_length
                )
            )

            (
                loss,
                quantizer_losses
            ) = (
                self.calculate_loss(
                    target_logits,
                    target_expected,
                    loss_function
                )
            )

            if not torch.isfinite(
                loss
            ):

                raise ValueError(
                    "Training loss became "
                    "NaN or infinite."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=(
                    self.gradient_clip_norm
                )
            )

            optimizer.step()

            accuracy = (
                self.calculate_accuracy(
                    target_logits,
                    target_expected
                )
            )

            total_loss += (
                loss.item()
            )

            total_accuracy += (
                accuracy
            )

            quantizer_loss_totals += (
                quantizer_losses
                .detach()
                .cpu()
                .to(
                    torch.float64
                )
            )

            number_of_batches += 1

        results = {
            "loss":
                total_loss
                / number_of_batches,

            "accuracy":
                total_accuracy
                / number_of_batches,

            "quantizer_losses":
                (
                    quantizer_loss_totals
                    / number_of_batches
                ).tolist()
        }

        return results

    def validate_epoch(
        self,
        model,
        data_loader,
        loss_function
    ):

        model.eval()

        total_loss = 0.0

        total_accuracy = 0.0

        quantizer_loss_totals = (
            torch.zeros(
                self.number_of_quantizers,
                dtype=torch.float64
            )
        )

        number_of_batches = 0

        with torch.no_grad():

            for (
                context_tokens,
                target_tokens
            ) in tqdm(
                data_loader,
                desc="Validating Transformer",
                leave=False
            ):

                context_tokens = (
                    context_tokens.to(
                        device=self.device,
                        dtype=torch.long
                    )
                )

                target_tokens = (
                    target_tokens.to(
                        device=self.device,
                        dtype=torch.long
                    )
                )

                (
                    model_input,
                    expected_output,
                    context_length
                ) = (
                    self.prepare_autoregressive_batch(
                        context_tokens,
                        target_tokens
                    )
                )

                logits = (
                    model(
                        model_input
                    )
                )

                (
                    target_logits,
                    target_expected
                ) = (
                    self.select_target_continuation(
                        logits,
                        expected_output,
                        context_length
                    )
                )

                (
                    loss,
                    quantizer_losses
                ) = (
                    self.calculate_loss(
                        target_logits,
                        target_expected,
                        loss_function
                    )
                )

                accuracy = (
                    self.calculate_accuracy(
                        target_logits,
                        target_expected
                    )
                )

                total_loss += (
                    loss.item()
                )

                total_accuracy += (
                    accuracy
                )

                quantizer_loss_totals += (
                    quantizer_losses
                    .detach()
                    .cpu()
                    .to(
                        torch.float64
                    )
                )

                number_of_batches += 1

        results = {
            "loss":
                total_loss
                / number_of_batches,

            "accuracy":
                total_accuracy
                / number_of_batches,

            "quantizer_losses":
                (
                    quantizer_loss_totals
                    / number_of_batches
                ).tolist()
        }

        return results

    def copy_model_state(
        self,
        model
    ):

        return {
            key:
                value
                .detach()
                .cpu()
                .clone()
            for key, value
            in model.state_dict().items()
        }

    def save_checkpoint(
        self,
        model,
        optimizer,
        epoch,
        model_name,
        best_validation_loss,
        best_epoch,
        checkpoint_type
    ):

        checkpoint = {
            "model_name":
                model_name,

            "checkpoint_type":
                checkpoint_type,

            "epoch":
                epoch,

            "model_state_dict":
                self.copy_model_state(
                    model
                ),

            "best_validation_loss":
                best_validation_loss,

            "best_epoch":
                best_epoch,

            "model_configuration": {
                "codebook_size":
                    self.codebook_size,

                "number_of_quantizers":
                    self.number_of_quantizers,

                "embedding_dimension":
                    self.embedding_dimension,

                "number_of_heads":
                    self.number_of_heads,

                "number_of_layers":
                    self.number_of_layers,

                "feed_forward_dimension":
                    self.feed_forward_dimension,

                "dropout":
                    self.dropout,

                "n_frequency_patches":
                    self.n_frequency_patches
            },

            "tokenizer_checkpoint":
                str(
                    self.tokenizer_checkpoint_path
                ),

            "saved_at":
                datetime.now().isoformat(
                    timespec="seconds"
                )
        }

        if checkpoint_type == "latest":

            checkpoint[
                "optimizer_state_dict"
            ] = (
                optimizer.state_dict()
            )

            file_name = (
                f"{model_name}_latest.pt"
            )

        elif checkpoint_type == "best":

            file_name = (
                f"{model_name}_best.pt"
            )

        else:

            raise ValueError(
                "checkpoint_type must be "
                "'latest' or 'best'."
            )

        file_path = (
            self.checkpoint_directory
            / file_name
        )

        torch.save(
            checkpoint,
            file_path
        )

        return file_path

    def train_model(
        self,
        training_loader,
        validation_loader,
        example_context_tokens,
        example_target_tokens,
        epochs,
        learning_rate,
        model_name,
        patience=PATIENCE
    ):

        model = (
            self.create_model()
        )

        optimizer = (
            self.create_optimizer(
                model,
                learning_rate
            )
        )

        loss_function = (
            self.create_loss_function()
        )

        # Save fixed context and ground-truth target
        # so checkpoint generations can be compared
        # against the same example.
        self.save_reference_examples(
            example_context_tokens=(
                example_context_tokens
            ),
            example_target_tokens=(
                example_target_tokens
            )
        )

        best_validation_loss = float(
            "inf"
        )

        best_epoch = None

        best_model_state = None

        patience_count = 0

        history = []

        for epoch in range(
            1,
            epochs + 1
        ):

            training_results = (
                self.train_epoch(
                    model=model,
                    data_loader=(
                        training_loader
                    ),
                    optimizer=(
                        optimizer
                    ),
                    loss_function=(
                        loss_function
                    )
                )
            )

            validation_results = (
                self.validate_epoch(
                    model=model,
                    data_loader=(
                        validation_loader
                    ),
                    loss_function=(
                        loss_function
                    )
                )
            )

            history.append({
                "epoch":
                    epoch,

                "training_loss":
                    training_results[
                        "loss"
                    ],

                "training_accuracy":
                    training_results[
                        "accuracy"
                    ],

                "training_quantizer_losses":
                    training_results[
                        "quantizer_losses"
                    ],

                "validation_loss":
                    validation_results[
                        "loss"
                    ],

                "validation_accuracy":
                    validation_results[
                        "accuracy"
                    ],

                "validation_quantizer_losses":
                    validation_results[
                        "quantizer_losses"
                    ]
            })

            print(
                f"\nEpoch {epoch}"
            )

            print(
                "Training loss:",
                f"{training_results['loss']:.6f}"
            )

            print(
                "Training token accuracy:",
                f"{training_results['accuracy'] * 100:.3f}%"
            )

            print(
                "Validation loss:",
                f"{validation_results['loss']:.6f}"
            )

            print(
                "Validation token accuracy:",
                f"{validation_results['accuracy'] * 100:.3f}%"
            )

            print(
                "\nTraining quantizer losses"
            )

            for (
                quantizer_index,
                quantizer_loss
            ) in enumerate(
                training_results[
                    "quantizer_losses"
                ]
            ):

                print(
                    f"Q{quantizer_index + 1}: "
                    f"{quantizer_loss:.6f}"
                )

            print(
                "\nValidation quantizer losses"
            )

            for (
                quantizer_index,
                quantizer_loss
            ) in enumerate(
                validation_results[
                    "quantizer_losses"
                ]
            ):

                print(
                    f"Q{quantizer_index + 1}: "
                    f"{quantizer_loss:.6f}"
                )

            if (
                validation_results[
                    "loss"
                ]
                < best_validation_loss
            ):

                best_validation_loss = (
                    validation_results[
                        "loss"
                    ]
                )

                best_epoch = (
                    epoch
                )

                best_model_state = (
                    self.copy_model_state(
                        model
                    )
                )

                patience_count = 0

                self.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    model_name=model_name,
                    best_validation_loss=(
                        best_validation_loss
                    ),
                    best_epoch=(
                        best_epoch
                    ),
                    checkpoint_type="best"
                )

            else:

                patience_count += 1

            # Checkpoint + generated audio
            if (
                epoch
                % self.checkpoint_interval
                == 0
            ):

                checkpoint_path = (
                    self.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        model_name=model_name,
                        best_validation_loss=(
                            best_validation_loss
                        ),
                        best_epoch=(
                            best_epoch
                        ),
                        checkpoint_type="latest"
                    )
                )

                print(
                    "\nLatest checkpoint:",
                    checkpoint_path
                )

                for example_index in range(
                    len(
                        example_context_tokens
                    )
                ):

                    context_tokens = (
                        example_context_tokens[
                            example_index
                        ]
                    )

                    target_tokens = (
                        example_target_tokens[
                            example_index
                        ]
                    )

                    number_of_target_tokens = (
                        target_tokens.shape[
                            0
                        ]
                    )

                    self.save_generated_example(
                        model=model,
                        context_token_ids=(
                            context_tokens
                        ),
                        number_of_target_tokens=(
                            number_of_target_tokens
                        ),
                        epoch=epoch,
                        model_name=model_name,
                        example_index=(
                            example_index
                        )
                    )

            if (
                patience_count
                >= patience
            ):

                print(
                    "\nValidation loss early "
                    "stop activated."
                )

                break

        if best_model_state is not None:

            model.load_state_dict(
                best_model_state
            )

        return {
            "model":
                model,

            "optimizer":
                optimizer,

            "history":
                history,

            "best_validation_loss":
                best_validation_loss,

            "best_epoch":
                best_epoch
        }
    

if __name__ == "__main__":

    # Load datasets
    training_dataset = (
        PretokenizedDrumDataset(
            context_path=(
                TRAINING_CONTEXT_PATH
            ),
            target_path=(
                TRAINING_TARGET_PATH
            )
        )
    )

    validation_dataset = (
        PretokenizedDrumDataset(
            context_path=(
                VALIDATION_CONTEXT_PATH
            ),
            target_path=(
                VALIDATION_TARGET_PATH
            )
        )
    )

    test_dataset = (
        PretokenizedDrumDataset(
            context_path=(
                TEST_CONTEXT_PATH
            ),
            target_path=(
                TEST_TARGET_PATH
            )
        )
    )

    print(
        "\nFull Dataset Information"
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

     # Fixed validation examples
    example_context_tokens = []

    example_target_tokens = []

    for example_index in range(
        NUMBER_OF_GENERATION_EXAMPLES
    ):

        (
            context_tokens,
            target_tokens
        ) = (
            validation_dataset[
                example_index
            ]
        )

        example_context_tokens.append(
            context_tokens
        )

        example_target_tokens.append(
            target_tokens
        )

    print(
        "\nFixed validation examples"
    )

    print(
        "=" * 70
    )

    print(
        "Number of examples:",
        len(
            example_context_tokens
        )
    )

    for example_index in range(
        len(
            example_context_tokens
        )
    ):

        print(
            f"Example {example_index + 1} "
            f"context shape:",
            example_context_tokens[
                example_index
            ].shape
        )

        print(
            f"Example {example_index + 1} "
            f"target shape:",
            example_target_tokens[
                example_index
            ].shape
        )

    # Small-run limits
    if SET_LIMIT:

        training_limit = min(
            MAXIMUM_TRAINING_EXAMPLES,
            len(
                training_dataset
            )
        )

        validation_limit = min(
            MAXIMUM_VALIDATION_EXAMPLES,
            len(
                validation_dataset
            )
        )

        training_dataset = (
            Subset(
                training_dataset,
                range(
                    training_limit
                )
            )
        )

        validation_dataset = (
            Subset(
                validation_dataset,
                range(
                    validation_limit
                )
            )
        )

    print(
        "\nTraining Run Dataset Information"
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
        "Test examples reserved:",
        len(
            test_dataset
        )
    )

    # DataLoaders
    training_loader = (
        DataLoader(
            training_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
    )

    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
    )

    # Pipeline
    training_pipeline = (
        TrainModelPipeline(
            tokenizer_checkpoint_path=(
                TOKENIZER_CHECKPOINT_PATH
            ),
            codebook_size=(
                CODEBOOK_SIZE
            ),
            number_of_quantizers=(
                NUMBER_OF_QUANTIZERS
            ),
            embedding_dimension=(
                EMBEDDING_DIMENSION
            ),
            number_of_heads=(
                NUMBER_OF_HEADS
            ),
            number_of_layers=(
                NUMBER_OF_LAYERS
            ),
            feed_forward_dimension=(
                FEED_FORWARD_DIMENSION
            ),
            dropout=(
                DROPOUT
            ),
            n_frequency_patches=(
                N_FREQUENCY_PATCHES
            ),
            gradient_clip_norm=(
                GRADIENT_CLIP_NORM
            ),
            weight_decay=(
                WEIGHT_DECAY
            ),
            checkpoint_interval=(
                CHECKPOINT_INTERVAL
            ),
            checkpoint_directory=(
                CHECKPOINT_DIRECTORY
            ),
            example_output_directory=(
                EXAMPLE_OUTPUT_DIRECTORY
            )
        )
    )

    print(
        "\nTraining device:",
        training_pipeline.device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    # ========================================================
    # Train
    # ========================================================

    print(
        "\n"
    )

    print(
        "#" * 70
    )

    print(
        "DISCRETE AUTOREGRESSIVE "
        "DRUM TRANSFORMER TRAINING"
    )

    print(
        "#" * 70
    )

    results = (
        training_pipeline
        .train_model(
            training_loader=(
                training_loader
            ),
            validation_loader=(
                validation_loader
            ),
            example_context_tokens=(
                example_context_tokens
            ),
            example_target_tokens=(
                example_target_tokens
            ),
            epochs=(
                EPOCHS
            ),
            learning_rate=(
                LEARNING_RATE
            ),
            model_name=(
                MODEL_NAME
            ),
            patience=(
                PATIENCE
            )
        )
    )

    # Training history
    print(
        "\nTraining History"
    )

    print(
        "=" * 70
    )

    for epoch_results in (
        results[
            "history"
        ]
    ):

        print(
            f"Epoch "
            f"{epoch_results['epoch']} | "
            f"Train loss: "
            f"{epoch_results['training_loss']:.6f} | "
            f"Train accuracy: "
            f"{epoch_results['training_accuracy'] * 100:.3f}% | "
            f"Validation loss: "
            f"{epoch_results['validation_loss']:.6f} | "
            f"Validation accuracy: "
            f"{epoch_results['validation_accuracy'] * 100:.3f}%"
        )

    print(
        "\nBest validation loss:",
        results[
            "best_validation_loss"
        ]
    )

    print(
        "Best epoch:",
        results[
            "best_epoch"
        ]
    )