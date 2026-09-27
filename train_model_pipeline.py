import torch
import copy

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

import soundfile as sf

from data_processing_pipeline import (
    reconstruct_audio_from_stft_tokens,
    TARGET_SAMPLE_RATE
)

from autoregressive_transformer import (
    AutoregressiveDrumTransformer
)

from construct_dataset import (
    create_slakh_datasets,
    SLAKH2100_REDUX_16K_TRAIN,
    SLAKH2100_REDUX_16K_VALIDATION
)

# Training configuration
BATCH_SIZE = 16
TOKEN_DIMENSION = 512
EMBEDDING_DIMENSION = 256
NUMBER_OF_HEADS = 8
NUMBER_OF_LAYERS = 4
FEED_FORWARD_DIMENSION = 1024
DROPOUT = 0.1
N_FREQUENCY_PATCHES = 16
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SMOOTH_L1_BETA = 3.0
GRADIENT_CLIP_NORM = 1.0
PATIENCE = 10

CHECKPOINT_INTERVAL = 5
CHECKPOINT_DIRECTORY = (
    "checkpoints"
)

SET_LIMIT = False
MAXIMUM_TRACKS = 1

MODEL_NAME = (
    "autoregressive_drum_transformer_MSE_Loss"
)

EXAMPLE_OUTPUT_DIRECTORY = (
    "generated_examples"
)

class TrainModelPipeline:
    """
    Trains the autoregressive drum continuation
    Transformer.
    """

    def __init__(
        self,
        token_dimension=512,
        embedding_dimension=256,
        number_of_heads=8,
        number_of_layers=4,
        feed_forward_dimension=1024,
        dropout=0.1,
        n_frequency_patches=16,
        smooth_l1_beta=3.0,
        gradient_clip_norm=1.0,
        weight_decay=1e-4,
        checkpoint_interval=5,
        checkpoint_directory="checkpoints",
        example_output_directory="generated_examples",
        device=None
    ):

        if checkpoint_interval < 1:

            raise ValueError(
                "checkpoint_interval must "
                "be at least 1."
            )

        if smooth_l1_beta <= 0:

            raise ValueError(
                "smooth_l1_beta must "
                "be greater than 0."
            )

        if gradient_clip_norm <= 0:

            raise ValueError(
                "gradient_clip_norm must "
                "be greater than 0."
            )

        self.token_dimension = (
            token_dimension
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

        self.smooth_l1_beta = (
            smooth_l1_beta
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

    def create_model(
        self
    ):
        """
        Creates a new autoregressive Transformer.
        """

        model = (
            AutoregressiveDrumTransformer(
                token_dimension=(
                    self.token_dimension
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

    def _copy_value(
        self,
        value
    ):
        """
        Recursively creates detached CPU copies
        of model-state values.
        """

        if isinstance(
            value,
            torch.Tensor
        ):

            return (
                value
                .detach()
                .cpu()
                .clone()
            )

        if isinstance(
            value,
            list
        ):

            return [
                self._copy_value(
                    item
                )
                for item in value
            ]

        if isinstance(
            value,
            tuple
        ):

            return tuple(
                self._copy_value(
                    item
                )
                for item in value
            )

        if isinstance(
            value,
            dict
        ):

            return {
                key:
                    self._copy_value(
                        item
                    )
                for key, item
                in value.items()
            }

        return copy.deepcopy(
            value
        )

    def copy_model_state(
        self,
        model
    ):
        """
        Returns an independent CPU copy of the
        current model parameters.
        """

        return (
            self._copy_value(
                model.state_dict()
            )
        )

    def _create_optimizer(
        self,
        model,
        learning_rate
    ):
        """
        Creates the AdamW optimizer.
        """

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

    def _create_loss_function(
        self
    ):
        """
        Creates the robust STFT regression loss.
        """

        loss_function = (
            torch.nn.MSELoss()
        )

        return loss_function

    def prepare_autoregressive_batch(
        self,
        context_batch,
        target_batch
    ):
        """
        Combines context and target into a full
        sequence and shifts it for next-token
        prediction.

        Context:
            [B, 304, 512]

        Target:
            [B, 304, 512]

        Full sequence:
            [B, 608, 512]

        Model input:
            [B, 607, 512]

        Expected output:
            [B, 607, 512]
        """

        if (
            context_batch.ndim != 3
            or target_batch.ndim != 3
        ):

            raise ValueError(
                "Context and target must have "
                "shape [batch, sequence, token]."
            )

        if (
            context_batch.shape
            != target_batch.shape
        ):

            raise ValueError(
                "Context and target batch "
                "shapes must match."
            )

        context_length = (
            context_batch.shape[1]
        )

        full_sequence = torch.cat(
            [
                context_batch,
                target_batch
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
        predictions,
        expected_output,
        context_length
    ):
        """
        Selects only the continuation section used
        for loss calculation.

        The first prediction is:

            final context token -> first target token

        Followed by:

            target token -> next target token
        """

        target_predictions = (
            predictions[
                :,
                context_length - 1:,
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

        if (
            target_predictions.shape
            != target_expected.shape
        ):

            raise ValueError(
                "Target predictions and expected "
                "targets have different shapes."
            )

        return (
            target_predictions,
            target_expected
        )

    def _save_checkpoint(
        self,
        model,
        optimizer,
        epoch,
        model_name,
        best_validation_loss,
        best_epoch,
        best_model_state,
        checkpoint_type="latest"
    ):
        """
        Saves a .pt checkpoint.

        latest:
            Saves current model and optimizer state
            so training can be resumed.

        best:
            Saves the model with the lowest
            validation loss.
        """

        if checkpoint_type not in (
            "latest",
            "best"
        ):

            raise ValueError(
                "checkpoint_type must be "
                "'latest' or 'best'."
            )

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

            "best_model_state_dict":
                self._copy_value(
                    best_model_state
                ),

            "model_configuration": {
                "token_dimension":
                    self.token_dimension,

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

            "training_configuration": {
                "smooth_l1_beta":
                    self.smooth_l1_beta,

                "gradient_clip_norm":
                    self.gradient_clip_norm,

                "weight_decay":
                    self.weight_decay
            },

            "saved_at":
                datetime.now().isoformat(
                    timespec="seconds"
                )
        }

        if checkpoint_type == "latest":

            checkpoint[
                "optimizer_state_dict"
            ] = (
                self._copy_value(
                    optimizer.state_dict()
                )
            )

            file_name = (
                f"{model_name}_latest.pt"
            )

        else:

            file_name = (
                f"{model_name}_best.pt"
            )

        file_path = (
            self.checkpoint_directory
            / file_name
        )

        temporary_path = (
            file_path.with_suffix(
                ".tmp"
            )
        )

        torch.save(
            checkpoint,
            temporary_path
        )

        temporary_path.replace(
            file_path
        )

        return file_path

    def save_generated_example(
        self,
        model,
        context_tokens,
        number_of_target_tokens,
        epoch,
        model_name
    ):
        """
        Generates and saves an example drum
        continuation for checkpoint evaluation.
        """

        generated_tokens = (
            model.generate_continuation(
                context_tokens=(
                    context_tokens
                ),
                number_of_target_tokens=(
                    number_of_target_tokens
                )
            )
        )

        generated_tokens = (
            generated_tokens
            .squeeze(
                0
            )
            .detach()
            .cpu()
        )

        generated_audio = (
            reconstruct_audio_from_stft_tokens(
                generated_tokens
            )
        )

        file_name = (
            f"{model_name}_"
            f"epoch_{epoch}_"
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

        return file_path

    def train_epoch(
        self,
        model,
        data_loader,
        optimizer,
        loss_function
    ):
        """
        Runs one autoregressive training epoch.
        """

        model.train()

        total_loss = 0.0

        number_of_batches = 0

        for (
            context_batch,
            target_batch
        ) in data_loader:

            context_batch = (
                context_batch.to(
                    device=self.device,
                    dtype=torch.float32
                )
            )

            target_batch = (
                target_batch.to(
                    device=self.device,
                    dtype=torch.float32
                )
            )

            optimizer.zero_grad()

            (
                model_input,
                expected_output,
                context_length
            ) = (
                self.prepare_autoregressive_batch(
                    context_batch,
                    target_batch
                )
            )

            predictions = (
                model(
                    model_input
                )
            )

            (
                target_predictions,
                target_expected
            ) = (
                self.select_target_continuation(
                    predictions,
                    expected_output,
                    context_length
                )
            )

            loss = (
                loss_function(
                    target_predictions,
                    target_expected
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

            total_loss += (
                loss.item()
            )

            number_of_batches += 1

        if number_of_batches == 0:

            raise ValueError(
                "Training DataLoader "
                "contains no batches."
            )

        average_loss = (
            total_loss
            / number_of_batches
        )

        return average_loss

    def validate_epoch(
        self,
        model,
        data_loader,
        loss_function
    ):
        """
        Runs one autoregressive validation epoch.
        """

        model.eval()

        total_loss = 0.0

        number_of_batches = 0

        with torch.no_grad():

            for (
                context_batch,
                target_batch
            ) in data_loader:

                context_batch = (
                    context_batch.to(
                        device=self.device,
                        dtype=torch.float32
                    )
                )

                target_batch = (
                    target_batch.to(
                        device=self.device,
                        dtype=torch.float32
                    )
                )

                (
                    model_input,
                    expected_output,
                    context_length
                ) = (
                    self.prepare_autoregressive_batch(
                        context_batch,
                        target_batch
                    )
                )

                predictions = (
                    model(
                        model_input
                    )
                )

                (
                    target_predictions,
                    target_expected
                ) = (
                    self.select_target_continuation(
                        predictions,
                        expected_output,
                        context_length
                    )
                )

                loss = (
                    loss_function(
                        target_predictions,
                        target_expected
                    )
                )

                if not torch.isfinite(
                    loss
                ):

                    raise ValueError(
                        "Validation loss became "
                        "NaN or infinite."
                    )

                total_loss += (
                    loss.item()
                )

                number_of_batches += 1

        if number_of_batches == 0:

            raise ValueError(
                "Validation DataLoader "
                "contains no batches."
            )

        average_loss = (
            total_loss
            / number_of_batches
        )

        return average_loss

    def train_model(
        self,
        training_loader,
        validation_loader,
        example_context_tokens,
        example_target_length,
        epochs,
        learning_rate,
        model_name,
        optimizer=None,
        start_epoch=1,
        best_validation_loss=float(
            "inf"
        ),
        best_epoch=None,
        best_model_state=None,
        patience=PATIENCE
    ):
        """
        Trains the autoregressive Transformer.

        Latest checkpoints are saved at the
        configured interval.

        The model with the lowest validation
        Smooth L1 loss is restored at the end.
        """

        model = (
            self.create_model()
        )

        if optimizer is None:

            optimizer = (
                self._create_optimizer(
                    model=model,
                    learning_rate=(
                        learning_rate
                    )
                )
            )

        loss_function = (
            self._create_loss_function()
        )

        history = []

        latest_checkpoint_paths = []

        if best_model_state is None:

            best_model_state = (
                self.copy_model_state(
                    model
                )
            )

        patience_count = 0

        for epoch in tqdm(
            range(
                start_epoch,
                epochs + 1
            )
        ):

            training_loss = (
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

            validation_loss = (
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

            if (
                validation_loss
                < best_validation_loss
            ):

                best_validation_loss = (
                    validation_loss
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

            else:

                patience_count += 1

            history.append({
                "epoch":
                    epoch,

                "training_loss":
                    training_loss,

                "validation_loss":
                    validation_loss,

                "best_validation_loss":
                    best_validation_loss,

                "best_epoch":
                    best_epoch
            })

            print(
                f"\nEpoch "
                f"{epoch} | "
                f"Training loss: "
                f"{training_loss:.6f} | "
                f"Validation loss: "
                f"{validation_loss:.6f}"
            )

            if (
                epoch
                % self.checkpoint_interval
                == 0
            ):

                checkpoint_path = (
                    self._save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        model_name=(
                            model_name
                        ),
                        best_validation_loss=(
                            best_validation_loss
                        ),
                        best_epoch=(
                            best_epoch
                        ),
                        best_model_state=(
                            best_model_state
                        ),
                        checkpoint_type=(
                            "latest"
                        )
                    )
                )

                latest_checkpoint_paths.append(
                    checkpoint_path
                )

                example_output_path = (
                    self.save_generated_example(
                        model=model,
                        context_tokens=(
                            example_context_tokens
                        ),
                        number_of_target_tokens=(
                            example_target_length
                        ),
                        epoch=epoch,
                        model_name=(
                            model_name
                        )
                    )
                )

                print(
                    "\nGenerated example:",
                    example_output_path
                )

            if (
                patience_count
                >= patience
            ):

                print(
                    "Validation loss early "
                    "stop activated."
                )

                break

        model.load_state_dict(
            best_model_state
        )

        best_checkpoint_path = (
            self._save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=best_epoch,
                model_name=(
                    model_name
                ),
                best_validation_loss=(
                    best_validation_loss
                ),
                best_epoch=(
                    best_epoch
                ),
                best_model_state=(
                    best_model_state
                ),
                checkpoint_type=(
                    "best"
                )
            )
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
                best_epoch,

            "latest_checkpoint_paths":
                latest_checkpoint_paths,

            "best_checkpoint_path":
                best_checkpoint_path
        }


if __name__ == "__main__":

    # Create datasets
    print(
        "Creating datasets..."
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

    # Get fixed example from validation dataset
    (
        example_context_tokens,
        example_target_tokens
    ) = (
        validation_dataset[0]
    )

    example_target_length = (
        example_target_tokens.shape[0]
    )

    example_context_audio = (
        reconstruct_audio_from_stft_tokens(
            example_context_tokens
        )
    )

    example_target_audio = (
        reconstruct_audio_from_stft_tokens(
            example_target_tokens
        )
    )

    example_output_directory = Path(
        EXAMPLE_OUTPUT_DIRECTORY
    )

    example_output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    sf.write(
        example_output_directory
        / "example_context.wav",
        example_context_audio,
        TARGET_SAMPLE_RATE
    )

    sf.write(
        example_output_directory
        / "example_target.wav",
        example_target_audio,
        TARGET_SAMPLE_RATE
    )

    print(
        "\nSaved fixed validation example:"
    )

    print(
        "Context:",
        example_output_directory
        / "example_context.wav"
    )

    print(
        "Target:",
        example_output_directory
        / "example_target.wav"
    )

    # Dataset information
    print(
        "\nDataset Information"
    )

    print(
        "=" * 70
    )

    print(
        "Training dataset size:",
        len(
            training_dataset
        )
    )

    print(
        "Validation dataset size:",
        len(
            validation_dataset
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

    # Create training pipeline
    training_pipeline = (
        TrainModelPipeline(
            token_dimension=(
                TOKEN_DIMENSION
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
            smooth_l1_beta=(
                SMOOTH_L1_BETA
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
            )
        )
    )

    # Device information
    print(
        "\nTraining device:",
        training_pipeline.device
    )

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    # Train model
    print("\n")

    print(
        "#" * 70
    )

    print(
        "AUTOREGRESSIVE DRUM "
        "CONTINUATION TRAINING"
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
            example_target_length=(
                example_target_length
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

    # Retrieve trained model
    trained_model = (
        results[
            "model"
        ]
    )

    # Print training history
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
            f"Training loss: "
            f"{epoch_results['training_loss']:.6f} | "
            f"Validation loss: "
            f"{epoch_results['validation_loss']:.6f}"
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

    print(
        "Best checkpoint:",
        results[
            "best_checkpoint_path"
        ]
    )