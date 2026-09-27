import torch
import soundfile as sf

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from stft_tokeniser import (
    STFTTokenizer
)

from construct_dataset import (
    create_slakh_datasets,
    SLAKH2100_REDUX_16K_TRAIN,
    SLAKH2100_REDUX_16K_VALIDATION
)

from data_processing_pipeline import (
    reconstruct_audio_from_stft_tokens,
    TARGET_SAMPLE_RATE,
    TIME_PATCH_SIZE,
    FREQUENCY_PATCH_SIZE
)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 16

PATCH_DIMENSION = (
    2
    * FREQUENCY_PATCH_SIZE
    * TIME_PATCH_SIZE
)
LATENT_DIMENSION = 256
CODEBOOK_SIZE = 1024

NUMBER_OF_QUANTIZERS = 8

EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 10

CHECKPOINT_INTERVAL = 5

CHECKPOINT_DIRECTORY = (
    "tokenizer_checkpoints"
)

EXAMPLE_OUTPUT_DIRECTORY = (
    "tokenizer_examples"
)

MODEL_NAME = (
    "stft_tokenizer_2"
)

SET_LIMIT = False
MAXIMUM_TRACKS = 100

WEIGHTED_MSE_ALPHA = 4.0

QUIET_LOSS_WEIGHT = 2.0

TRANSIENT_LOSS_WEIGHT = 3.0

QUIET_THRESHOLD = 0.05

RESUME_CHECKPOINT = (
    "tokenizer_checkpoints/stft_tokenizer_2_latest.pt"
)


# ============================================================
# Training Pipeline
# ============================================================

class TrainTokenizerPipeline:

    def __init__(
        self,
        patch_dimension=PATCH_DIMENSION,
        latent_dimension=LATENT_DIMENSION,
        codebook_size=CODEBOOK_SIZE,
        number_of_quantizers=NUMBER_OF_QUANTIZERS,
        weight_decay=WEIGHT_DECAY,
        weighted_mse_alpha=WEIGHTED_MSE_ALPHA,
        quiet_loss_weight=QUIET_LOSS_WEIGHT,
        transient_loss_weight=TRANSIENT_LOSS_WEIGHT,
        quiet_threshold=QUIET_THRESHOLD,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        checkpoint_directory=CHECKPOINT_DIRECTORY,
        example_output_directory=EXAMPLE_OUTPUT_DIRECTORY,
        device=None
    ):

        self.patch_dimension = (
            patch_dimension
        )

        self.latent_dimension = (
            latent_dimension
        )

        self.codebook_size = (
            codebook_size
        )

        self.number_of_quantizers = (
            number_of_quantizers
        )

        self.weight_decay = (
            weight_decay
        )

        self.weighted_mse_alpha = (
            weighted_mse_alpha
        )

        self.quiet_loss_weight = (
            quiet_loss_weight
        )

        self.transient_loss_weight = (
            transient_loss_weight
        )

        self.quiet_threshold = (
            quiet_threshold
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

    def create_tokenizer(
        self
    ):

        tokenizer = STFTTokenizer(
            patch_dimension=(
                self.patch_dimension
            ),
            latent_dimension=(
                self.latent_dimension
            ),
            codebook_size=(
                self.codebook_size
            )
        )

        tokenizer = tokenizer.to(
            self.device
        )

        return tokenizer

    def create_optimizer(
        self,
        tokenizer,
        learning_rate
    ):

        optimizer = torch.optim.AdamW(
            tokenizer.parameters(),
            lr=learning_rate,
            weight_decay=(
                self.weight_decay
            )
        )

        return optimizer

    def prepare_patch_batch(
        self,
        context_batch,
        target_batch
    ):

        if (
            context_batch.ndim != 3
            or target_batch.ndim != 3
        ):

            raise ValueError(
                "Context and target must have shape "
                "[batch, sequence, patch_dimension]."
            )

        if (
            context_batch.shape
            != target_batch.shape
        ):

            raise ValueError(
                "Context and target shapes "
                "must match."
            )

        patches = torch.cat(
            [
                context_batch,
                target_batch
            ],
            dim=1
        )

        return patches

    def calculate_loss(
        self,
        reconstructed_patches,
        original_patches,
        commitment_loss
    ):

        # Restore patch structure:
        #
        # [B, S, patch_dimension]
        #
        # becomes:
        #
        # [B, S, 2, frequency, time]
        reconstructed_structured = (
            reconstructed_patches.reshape(
                reconstructed_patches.shape[0],
                reconstructed_patches.shape[1],
                2,
                FREQUENCY_PATCH_SIZE,
                TIME_PATCH_SIZE
            )
        )

        original_structured = (
            original_patches.reshape(
                original_patches.shape[0],
                original_patches.shape[1],
                2,
                FREQUENCY_PATCH_SIZE,
                TIME_PATCH_SIZE
            )
        )

        # ====================================================
        # Real and imaginary components
        # ====================================================

        reconstructed_real = (
            reconstructed_structured[
                :,
                :,
                0,
                :,
                :
            ]
        )

        reconstructed_imaginary = (
            reconstructed_structured[
                :,
                :,
                1,
                :,
                :
            ]
        )

        original_real = (
            original_structured[
                :,
                :,
                0,
                :,
                :
            ]
        )

        original_imaginary = (
            original_structured[
                :,
                :,
                1,
                :,
                :
            ]
        )

        # ====================================================
        # Magnitudes
        # ====================================================

        reconstructed_magnitude = (
            torch.sqrt(
                reconstructed_real.pow(
                    2
                )
                + reconstructed_imaginary.pow(
                    2
                )
                + 1e-8
            )
        )

        original_magnitude = (
            torch.sqrt(
                original_real.pow(
                    2
                )
                + original_imaginary.pow(
                    2
                )
                + 1e-8
            )
        )

        # ====================================================
        # Normalize original magnitude
        # ====================================================

        maximum_magnitude = (
            original_magnitude
            .amax(
                dim=(
                    1,
                    2,
                    3
                ),
                keepdim=True
            )
        )

        normalized_magnitude = (
            original_magnitude
            / (
                maximum_magnitude
                + 1e-8
            )
        )

        # ====================================================
        # 1. Weighted complex MSE
        # ====================================================

        magnitude_weights = (
            1.0
            + self.weighted_mse_alpha
            * normalized_magnitude
        )

        # Apply the same magnitude-derived weight
        # to the real and imaginary channels.
        magnitude_weights = (
            magnitude_weights.unsqueeze(
                2
            )
        )

        squared_error = (
            reconstructed_structured
            - original_structured
        ).pow(
            2
        )

        weighted_reconstruction_loss = (
            magnitude_weights
            * squared_error
        ).mean()

        # ====================================================
        # 2. Quiet-region loss
        # ====================================================

        quiet_mask = (
            normalized_magnitude
            < self.quiet_threshold
        ).float()

        # Penalize reconstructed energy in regions
        # where the original STFT is quiet.
        quiet_error = (
            reconstructed_magnitude.pow(
                2
            )
            * quiet_mask
        )

        quiet_loss = (
            quiet_error.sum()
            / (
                quiet_mask.sum()
                + 1e-8
            )
        )

        # ====================================================
        # 3. Transient / onset loss
        # ====================================================

        # Measure frame-to-frame magnitude changes
        # within each time-frequency patch.
        original_difference = (
            original_magnitude[
                :,
                :,
                :,
                1:
            ]
            - original_magnitude[
                :,
                :,
                :,
                :-1
            ]
        )

        reconstructed_difference = (
            reconstructed_magnitude[
                :,
                :,
                :,
                1:
            ]
            - reconstructed_magnitude[
                :,
                :,
                :,
                :-1
            ]
        )

        transient_loss = (
            torch.nn.functional.l1_loss(
                reconstructed_difference,
                original_difference
            )
        )

        # ====================================================
        # 4. Commitment loss
        # ====================================================

        commitment_loss = (
            commitment_loss.mean()
        )

        # ====================================================
        # Total loss
        # ====================================================

        total_loss = (
            weighted_reconstruction_loss
            + self.quiet_loss_weight
            * quiet_loss
            + self.transient_loss_weight
            * transient_loss
            + commitment_loss
        )

        return (
            total_loss,
            weighted_reconstruction_loss,
            quiet_loss,
            transient_loss,
            commitment_loss
        )

    def train_epoch(
        self,
        tokenizer,
        data_loader,
        optimizer
    ):

        tokenizer.train()

        total_loss = 0.0
        total_reconstruction_loss = 0.0
        total_quiet_loss = 0.0
        total_transient_loss = 0.0
        total_commitment_loss = 0.0

        number_of_batches = 0

        for (
            context_batch,
            target_batch
        ) in tqdm(
            data_loader,
            desc="Training tokenizer",
            leave=False
        ):

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

            patches = (
                self.prepare_patch_batch(
                    context_batch,
                    target_batch
                )
            )

            optimizer.zero_grad()

            (
                reconstructed_patches,
                token_ids,
                commitment_loss
            ) = (
                tokenizer(
                    patches
                )
            )

            (
                loss,
                reconstruction_loss,
                quiet_loss,
                transient_loss,
                commitment_loss
            ) = (
                self.calculate_loss(
                    reconstructed_patches,
                    patches,
                    commitment_loss
                )
            )

            if not torch.isfinite(
                loss
            ):

                raise ValueError(
                    "Tokenizer training loss became "
                    "NaN or infinite."
                )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

            total_reconstruction_loss += (
                reconstruction_loss.item()
            )

            total_quiet_loss += (
                quiet_loss.item()
            )

            total_transient_loss += (
                transient_loss.item()
            )

            total_commitment_loss += (
                commitment_loss.item()
            )

            number_of_batches += 1

        if number_of_batches == 0:

            raise ValueError(
                "Training DataLoader contains "
                "no batches."
            )

        results = {
            "loss":
                total_loss
                / number_of_batches,

            "reconstruction_loss":
                total_reconstruction_loss
                / number_of_batches,

            "quiet_loss":
                total_quiet_loss 
                / number_of_batches,

            "transient_loss":
                total_transient_loss 
                / number_of_batches,

            "commitment_loss":
                total_commitment_loss
                / number_of_batches
        }

        return results

    def validate_epoch(
        self,
        tokenizer,
        data_loader
    ):

        tokenizer.eval()

        total_loss = 0.0
        total_reconstruction_loss = 0.0
        total_quiet_loss = 0.0
        total_transient_loss = 0.0
        total_commitment_loss = 0.0

        number_of_batches = 0

        with torch.no_grad():

            for (
                context_batch,
                target_batch
            ) in tqdm(
                data_loader,
                desc="Validating tokenizer",
                leave=False
            ):

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

                patches = (
                    self.prepare_patch_batch(
                        context_batch,
                        target_batch
                    )
                )

                (
                    reconstructed_patches,
                    token_ids,
                    commitment_loss
                ) = (
                    tokenizer(
                        patches
                    )
                )

                (
                    loss,
                    reconstruction_loss,
                    quiet_loss,
                    transient_loss,
                    commitment_loss
                ) = (
                    self.calculate_loss(
                        reconstructed_patches,
                        patches,
                        commitment_loss
                    )
                )

                if not torch.isfinite(
                    loss
                ):

                    raise ValueError(
                        "Tokenizer validation loss became "
                        "NaN or infinite."
                    )

                total_loss += (
                    loss.item()
                )

                total_reconstruction_loss += (
                    reconstruction_loss.item()
                )

                total_quiet_loss += (
                    quiet_loss.item()
                )
                
                total_transient_loss += (
                    transient_loss.item()
                )

                total_commitment_loss += (
                    commitment_loss.item()
                )

                number_of_batches += 1

        if number_of_batches == 0:

            raise ValueError(
                "Validation DataLoader contains "
                "no batches."
            )

        results = {
            "loss":
                total_loss
                / number_of_batches,

            "reconstruction_loss":
                total_reconstruction_loss
                / number_of_batches,

            "quiet_loss":
                total_quiet_loss 
                / number_of_batches,
            
            "transient_loss":
                total_transient_loss 
                / number_of_batches,

            "commitment_loss":
                total_commitment_loss
                / number_of_batches
        }

        return results

    def save_checkpoint(
        self,
        tokenizer,
        optimizer,
        epoch,
        model_name,
        best_validation_loss,
        best_epoch,
        checkpoint_type="latest"
    ):

        checkpoint = {
            "model_name":
                model_name,

            "epoch":
                epoch,

            "checkpoint_type":
                checkpoint_type,

            "model_state_dict":
                tokenizer.state_dict(),

            "best_validation_loss":
                best_validation_loss,

            "best_epoch":
                best_epoch,

            "model_configuration": {
                "patch_dimension":
                    self.patch_dimension,

                "latent_dimension":
                    self.latent_dimension,

                "codebook_size":
                    self.codebook_size,

                "number_of_quantizers":
                    self.number_of_quantizers,

                "weighted_mse_alpha":
                    self.weighted_mse_alpha,

                "quiet_loss_weight":
                    self.quiet_loss_weight,

                "transient_loss_weight":
                    self.transient_loss_weight,

                "quiet_threshold":
                    self.quiet_threshold
            }
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

        else:

            file_name = (
                f"{model_name}_best.pt"
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

    def load_checkpoint(
        self,
        tokenizer,
        optimizer,
        checkpoint_path
    ):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        tokenizer.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        if (
            "optimizer_state_dict"
            in checkpoint
        ):

            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        start_epoch = (
            checkpoint[
                "epoch"
            ]
            + 1
        )

        best_validation_loss = (
            checkpoint.get(
                "best_validation_loss",
                float(
                    "inf"
                )
            )
        )

        best_epoch = (
            checkpoint.get(
                "best_epoch",
                None
            )
        )

        print(
            "\nCheckpoint loaded"
        )

        print(
            "=" * 70
        )

        print(
            "Checkpoint:",
            checkpoint_path
        )

        print(
            "Completed epoch:",
            checkpoint[
                "epoch"
            ]
        )

        print(
            "Resuming from epoch:",
            start_epoch
        )

        print(
            "Best validation loss:",
            best_validation_loss
        )

        print(
            "Best epoch:",
            best_epoch
        )

        return (
            start_epoch,
            best_validation_loss,
            best_epoch
        )

    def save_reconstruction_example(
        self,
        tokenizer,
        example_patches,
        epoch,
        model_name
    ):

        tokenizer.eval()

        example_patches = (
            example_patches
            .unsqueeze(
                0
            )
            .to(
                device=self.device,
                dtype=torch.float32
            )
        )

        with torch.no_grad():

            (
                reconstructed_patches,
                token_ids,
                commitment_loss
            ) = (
                tokenizer(
                    example_patches
                )
            )

        reconstructed_patches = (
            reconstructed_patches[
                0
            ]
            .detach()
            .cpu()
        )

        token_ids = (
            token_ids[
                0
            ]
            .detach()
            .cpu()
        )

        reconstructed_audio = (
            reconstruct_audio_from_stft_tokens(
                reconstructed_patches
            )
        )

        file_name = (
            f"{model_name}_"
            f"epoch_{epoch}_"
            f"reconstructed.wav"
        )

        file_path = (
            self.example_output_directory
            / file_name
        )

        sf.write(
            file_path,
            reconstructed_audio,
            TARGET_SAMPLE_RATE
        )

        print(
            "\nTokenizer reconstruction:"
        )

        print(
            "Audio:",
            file_path
        )

        print(
            "Token ID shape:",
            token_ids.shape
        )

        if token_ids.ndim == 2:

            for quantizer_index in range(
                token_ids.shape[
                    1
                ]
            ):

                quantizer_ids = (
                    token_ids[
                        :,
                        quantizer_index
                    ]
                )

                print(
                    f"Quantizer "
                    f"{quantizer_index + 1} "
                    f"unique token IDs:",
                    torch.unique(
                        quantizer_ids
                    ).numel()
                )

        return file_path

    def train_tokenizer(
        self,
        training_loader,
        validation_loader,
        example_patches,
        epochs,
        learning_rate,
        model_name,
        patience=10
    ):

        tokenizer = (
            self.create_tokenizer()
        )

        optimizer = (
            self.create_optimizer(
                tokenizer,
                learning_rate
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
                    tokenizer,
                    training_loader,
                    optimizer
                )
            )

            validation_results = (
                self.validate_epoch(
                    tokenizer,
                    validation_loader
                )
            )

            history.append({
                "epoch":
                    epoch,

                "training_loss":
                    training_results[
                        "loss"
                    ],

                "training_reconstruction_loss":
                    training_results[
                        "reconstruction_loss"
                    ],

                "training_quiet_loss":
                    training_results[
                        "quiet_loss"
                    ],

                "training_transient_loss":
                    training_results[
                        "transient_loss"
                    ],

                "training_commitment_loss":
                    training_results[
                        "commitment_loss"
                    ],

                "validation_loss":
                    validation_results[
                        "loss"
                    ],

                "validation_reconstruction_loss":
                    validation_results[
                        "reconstruction_loss"
                    ],

                "validation_quiet_loss":
                    validation_results[
                        "quiet_loss"
                    ],
                
                "validation_transient_loss":
                    validation_results[
                        "transient_loss"
                    ],

                "validation_commitment_loss":
                    validation_results[
                        "commitment_loss"
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
                "Training reconstruction loss:",
                f"{training_results['reconstruction_loss']:.6f}"
            )

            print(
                "Training quiet loss:",
                f"{training_results['quiet_loss']:.6f}"
            )

            print(
                "Training transient loss:",
                f"{training_results['transient_loss']:.6f}"
            )

            print(
                "Training commitment loss:",
                f"{training_results['commitment_loss']:.6f}"
            )

            print(
                "Validation loss:",
                f"{validation_results['loss']:.6f}"
            )

            print(
                "Validation reconstruction loss:",
                f"{validation_results['reconstruction_loss']:.6f}"
            )

            print(
                "Validation quiet loss:",
                f"{validation_results['quiet_loss']:.6f}"
            )

            print(
                "Validation transient loss:",
                f"{validation_results['transient_loss']:.6f}"
            )

            print(
                "Validation commitment loss:",
                f"{validation_results['commitment_loss']:.6f}"
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

                best_model_state = {
                    key:
                        value
                        .detach()
                        .cpu()
                        .clone()
                    for key, value
                    in tokenizer
                    .state_dict()
                    .items()
                }

                patience_count = 0

                self.save_checkpoint(
                    tokenizer=tokenizer,
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

            if (
                epoch
                % self.checkpoint_interval
                == 0
            ):

                self.save_checkpoint(
                    tokenizer=tokenizer,
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

                self.save_reconstruction_example(
                    tokenizer=tokenizer,
                    example_patches=(
                        example_patches
                    ),
                    epoch=epoch,
                    model_name=(
                        model_name
                    )
                )

            if (
                patience_count
                >= patience
            ):

                print(
                    "\nEarly stopping activated."
                )

                break

        if best_model_state is not None:

            tokenizer.load_state_dict(
                best_model_state
            )

        return {
            "tokenizer":
                tokenizer,

            "history":
                history,

            "best_validation_loss":
                best_validation_loss,

            "best_epoch":
                best_epoch
        }

    def resume_tokenizer_training(
        self,
        training_loader,
        validation_loader,
        example_patches,
        checkpoint_path,
        epochs,
        learning_rate,
        model_name,
        patience=10
    ):

        # Create tokenizer.
        tokenizer = (
            self.create_tokenizer()
        )

        # Create optimizer.
        optimizer = (
            self.create_optimizer(
                tokenizer,
                learning_rate
            )
        )

        # Load checkpoint.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        tokenizer.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        if (
            "optimizer_state_dict"
            not in checkpoint
        ):

            raise ValueError(
                "Checkpoint does not contain "
                "optimizer_state_dict."
            )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        completed_epoch = (
            checkpoint[
                "epoch"
            ]
        )

        start_epoch = (
            completed_epoch
            + 1
        )

        best_validation_loss = (
            checkpoint.get(
                "best_validation_loss",
                float(
                    "inf"
                )
            )
        )

        best_epoch = (
            checkpoint.get(
                "best_epoch",
                None
            )
        )

        best_model_state = None

        patience_count = 0

        history = []

        print(
            "\nResuming tokenizer training"
        )

        print(
            "=" * 70
        )

        print(
            "Checkpoint:",
            checkpoint_path
        )

        print(
            "Completed epoch:",
            completed_epoch
        )

        print(
            "Starting epoch:",
            start_epoch
        )

        print(
            "Target epoch:",
            epochs
        )

        print(
            "Best validation loss:",
            best_validation_loss
        )

        print(
            "Best epoch:",
            best_epoch
        )

        if (
            start_epoch
            > epochs
        ):

            raise ValueError(
                "Checkpoint epoch is already "
                "greater than or equal to the "
                "requested final epoch."
            )

        for epoch in range(
            start_epoch,
            epochs + 1
        ):

            training_results = (
                self.train_epoch(
                    tokenizer,
                    training_loader,
                    optimizer
                )
            )

            validation_results = (
                self.validate_epoch(
                    tokenizer,
                    validation_loader
                )
            )

            history.append({
                "epoch":
                    epoch,

                "training_loss":
                    training_results[
                        "loss"
                    ],

                "training_reconstruction_loss":
                    training_results[
                        "reconstruction_loss"
                    ],

                "training_quiet_loss":
                    training_results[
                        "quiet_loss"
                    ],

                "training_transient_loss":
                    training_results[
                        "transient_loss"
                    ],

                "training_commitment_loss":
                    training_results[
                        "commitment_loss"
                    ],

                "validation_loss":
                    validation_results[
                        "loss"
                    ],

                "validation_reconstruction_loss":
                    validation_results[
                        "reconstruction_loss"
                    ],

                "validation_quiet_loss":
                    validation_results[
                        "quiet_loss"
                    ],

                "validation_transient_loss":
                    validation_results[
                        "transient_loss"
                    ],

                "validation_commitment_loss":
                    validation_results[
                        "commitment_loss"
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
                "Training reconstruction loss:",
                f"{training_results['reconstruction_loss']:.6f}"
            )

            print(
                "Training quiet loss:",
                f"{training_results['quiet_loss']:.6f}"
            )

            print(
                "Training transient loss:",
                f"{training_results['transient_loss']:.6f}"
            )

            print(
                "Training commitment loss:",
                f"{training_results['commitment_loss']:.6f}"
            )

            print(
                "Validation loss:",
                f"{validation_results['loss']:.6f}"
            )

            print(
                "Validation reconstruction loss:",
                f"{validation_results['reconstruction_loss']:.6f}"
            )

            print(
                "Validation quiet loss:",
                f"{validation_results['quiet_loss']:.6f}"
            )

            print(
                "Validation transient loss:",
                f"{validation_results['transient_loss']:.6f}"
            )

            print(
                "Validation commitment loss:",
                f"{validation_results['commitment_loss']:.6f}"
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

                best_model_state = {
                    key:
                        value
                        .detach()
                        .cpu()
                        .clone()
                    for key, value
                    in tokenizer
                    .state_dict()
                    .items()
                }

                patience_count = 0

                self.save_checkpoint(
                    tokenizer=tokenizer,
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

            if (
                epoch
                % self.checkpoint_interval
                == 0
            ):

                self.save_checkpoint(
                    tokenizer=tokenizer,
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

                self.save_reconstruction_example(
                    tokenizer=tokenizer,
                    example_patches=(
                        example_patches
                    ),
                    epoch=epoch,
                    model_name=(
                        model_name
                    )
                )

            if (
                patience_count
                >= patience
            ):

                print(
                    "\nEarly stopping activated."
                )

                break

        if best_model_state is not None:

            tokenizer.load_state_dict(
                best_model_state
            )

        return {
            "tokenizer":
                tokenizer,

            "history":
                history,

            "best_validation_loss":
                best_validation_loss,

            "best_epoch":
                best_epoch
        }

    def calculate_codebook_usage(
        self,
        tokenizer,
        data_loader
    ):

        tokenizer.eval()

        token_counts = torch.zeros(
            self.number_of_quantizers,
            self.codebook_size,
            dtype=torch.long
        )

        total_tokens_per_quantizer = 0

        with torch.no_grad():

            for (
                context_batch,
                target_batch
            ) in tqdm(
                data_loader,
                desc="Calculating codebook usage",
                leave=False
            ):

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

                patches = torch.cat(
                    [
                        context_batch,
                        target_batch
                    ],
                    dim=1
                )

                (
                    reconstructed_patches,
                    token_ids,
                    commitment_loss
                ) = (
                    tokenizer(
                        patches
                    )
                )

                token_ids = (
                    token_ids
                    .detach()
                    .cpu()
                )

                if token_ids.ndim != 3:

                    raise ValueError(
                        "ResidualVQ token IDs must have "
                        "shape [B, S, num_quantizers]."
                    )

                for quantizer_index in range(
                    self.number_of_quantizers
                ):

                    quantizer_token_ids = (
                        token_ids[
                            :,
                            :,
                            quantizer_index
                        ]
                        .reshape(
                            -1
                        )
                    )

                    batch_counts = (
                        torch.bincount(
                            quantizer_token_ids,
                            minlength=(
                                self.codebook_size
                            )
                        )
                    )

                    token_counts[
                        quantizer_index
                    ] += (
                        batch_counts
                    )

                total_tokens_per_quantizer += (
                    token_ids.shape[
                        0
                    ]
                    * token_ids.shape[
                        1
                    ]
                )

        results = []

        for quantizer_index in range(
            self.number_of_quantizers
        ):

            counts = (
                token_counts[
                    quantizer_index
                ]
            )

            used_mask = (
                counts > 0
            )

            number_of_used_tokens = (
                used_mask
                .sum()
                .item()
            )

            usage_percentage = (
                number_of_used_tokens
                / self.codebook_size
                * 100.0
            )

            probabilities = (
                counts[
                    used_mask
                ]
                .to(
                    dtype=torch.float64
                )
                / total_tokens_per_quantizer
            )

            entropy = -(
                probabilities
                * torch.log(
                    probabilities
                )
            ).sum()

            perplexity = (
                torch.exp(
                    entropy
                )
                .item()
            )

            (
                most_common_counts,
                most_common_ids
            ) = (
                torch.topk(
                    counts,
                    k=min(
                        10,
                        self.codebook_size
                    )
                )
            )

            most_common_percentage = (
                most_common_counts[
                    0
                ].item()
                / total_tokens_per_quantizer
                * 100.0
            )

            results.append({
                "quantizer":
                    quantizer_index
                    + 1,

                "total_tokens":
                    total_tokens_per_quantizer,

                "used_tokens":
                    number_of_used_tokens,

                "unused_tokens":
                    (
                        self.codebook_size
                        - number_of_used_tokens
                    ),

                "usage_percentage":
                    usage_percentage,

                "perplexity":
                    perplexity,

                "most_common_token":
                    most_common_ids[
                        0
                    ].item(),

                "most_common_percentage":
                    most_common_percentage,

                "top_10_token_ids":
                    most_common_ids.tolist(),

                "top_10_token_counts":
                    most_common_counts.tolist()
            })

        return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

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

    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    (
        example_context_patches,
        example_target_patches
    ) = (
        validation_dataset[
            0
        ]
    )

    original_context_audio = (
        reconstruct_audio_from_stft_tokens(
            example_context_patches
        )
    )

    example_output_directory = Path(
        EXAMPLE_OUTPUT_DIRECTORY
    )

    example_output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    original_context_path = (
        example_output_directory
        / "original_validation_context.wav"
    )

    sf.write(
        original_context_path,
        original_context_audio,
        TARGET_SAMPLE_RATE
    )

    print(
        "\nSaved original validation context:"
    )

    print(
        original_context_path
    )

    training_pipeline = (
        TrainTokenizerPipeline(
            patch_dimension=(
                PATCH_DIMENSION
            ),
            latent_dimension=(
                LATENT_DIMENSION
            ),
            codebook_size=(
                CODEBOOK_SIZE
            ),
            number_of_quantizers=(
                NUMBER_OF_QUANTIZERS
            ),
            weight_decay=(
                WEIGHT_DECAY
            ),
            weighted_mse_alpha=(
                WEIGHTED_MSE_ALPHA
            ),
            quiet_loss_weight=(
                QUIET_LOSS_WEIGHT
            ),
            transient_loss_weight=(
                TRANSIENT_LOSS_WEIGHT
            ),
            quiet_threshold=(
                QUIET_THRESHOLD
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

    print(
        "\n"
    )

    print(
        "#" * 70
    )

    print(
        "STFT TOKENIZER TRAINING"
    )

    print(
        "#" * 70
    )

    # results = (
    #     training_pipeline
    #     .train_tokenizer(
    #         training_loader=(
    #             training_loader
    #         ),
    #         validation_loader=(
    #             validation_loader
    #         ),
    #         example_patches=(
    #             example_context_patches
    #         ),
    #         epochs=(
    #             EPOCHS
    #         ),
    #         learning_rate=(
    #             LEARNING_RATE
    #         ),
    #         model_name=(
    #             MODEL_NAME
    #         ),
    #         patience=(
    #             PATIENCE
    #         )
    #     )
    # )

    results = (
        training_pipeline
        .resume_tokenizer_training(
            training_loader=(
                training_loader
            ),
            validation_loader=(
                validation_loader
            ),
            example_patches=(
                example_context_patches
            ),
            checkpoint_path=RESUME_CHECKPOINT,
            epochs=(
                30
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
            f"Train: "
            f"{epoch_results['training_loss']:.6f} | "
            f"Validation: "
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

    codebook_results = (
        training_pipeline
        .calculate_codebook_usage(
            tokenizer=(
                results[
                    "tokenizer"
                ]
            ),
            data_loader=(
                validation_loader
            )
        )
    )

    print(
        "\nResidual VQ Codebook Usage"
    )

    print(
        "=" * 70
    )

    for quantizer_results in (
        codebook_results
    ):

        print(
            "\nQuantizer:",
            quantizer_results[
                "quantizer"
            ]
        )

        print(
            "Total token assignments:",
            quantizer_results[
                "total_tokens"
            ]
        )

        print(
            "Used codebook entries:",
            quantizer_results[
                "used_tokens"
            ]
        )

        print(
            "Unused codebook entries:",
            quantizer_results[
                "unused_tokens"
            ]
        )

        print(
            "Codebook usage:",
            f"{quantizer_results['usage_percentage']:.2f}%"
        )

        print(
            "Codebook perplexity:",
            f"{quantizer_results['perplexity']:.2f}"
        )

        print(
            "Most common token:",
            quantizer_results[
                "most_common_token"
            ]
        )

        print(
            "Most common token percentage:",
            f"{quantizer_results['most_common_percentage']:.2f}%"
        )

        print(
            "Top 10 token IDs:",
            quantizer_results[
                "top_10_token_ids"
            ]
        )

        print(
            "Top 10 token counts:",
            quantizer_results[
                "top_10_token_counts"
            ]
        )

        print(
            "-" * 70
        )