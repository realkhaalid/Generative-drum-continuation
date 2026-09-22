import torch
import torch.nn as nn

from positional_encoding_functions import (
    add_sequence_positional_encoding
)

class AutoregressiveDrumTransformer(
    nn.Module
):
    """
    Autoregressive causal Transformer for
    drum audio continuation.

    Input:
        [batch_size, sequence_length, token_dimension]

    Output:
        [batch_size, sequence_length, token_dimension]

    Each output position predicts the next
    STFT token in the sequence.
    """

    def __init__(
        self,
        token_dimension=512,
        embedding_dimension=256,
        number_of_heads=8,
        number_of_layers=4,
        feed_forward_dimension=1024,
        dropout=0.1,
        n_frequency_patches=16
    ):
        super().__init__()

        self.token_dimension = (
            token_dimension
        )

        self.embedding_dimension = (
            embedding_dimension
        )

        self.n_frequency_patches = (
            n_frequency_patches
        )

        self.token_embedding = nn.Linear(
            token_dimension,
            embedding_dimension
        )

        transformer_layer = (
            nn.TransformerEncoderLayer(
                d_model=(
                    embedding_dimension
                ),
                nhead=(
                    number_of_heads
                ),
                dim_feedforward=(
                    feed_forward_dimension
                ),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer=(
                    transformer_layer
                ),
                num_layers=(
                    number_of_layers
                ),
                norm=nn.LayerNorm(
                    embedding_dimension
                )
            )
        )

        self.output_projection = nn.Linear(
            embedding_dimension,
            token_dimension
        )

        self._initialize_weights()

    def _initialize_weights(
        self
    ):
        """
        Initializes trainable matrices using
        Xavier uniform initialization.
        """

        for parameter in (
            self.parameters()
        ):

            if parameter.dim() > 1:

                nn.init.xavier_uniform_(
                    parameter
                )

    def create_causal_mask(
        self,
        sequence_length,
        device
    ):
        """
        Creates a causal self-attention mask.

        True values represent positions that
        cannot be attended to.
        """

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        return causal_mask

    def forward(
        self,
        tokens
    ):
        """
        Performs causal next-token prediction.
        """

        if tokens.ndim != 3:

            raise ValueError(
                "Expected token tensor shape "
                "[batch_size, sequence_length, "
                "token_dimension], but received "
                f"{tokens.shape}."
            )

        if (
            tokens.shape[2]
            != self.token_dimension
        ):

            raise ValueError(
                f"Expected token dimension "
                f"{self.token_dimension}, "
                f"but received "
                f"{tokens.shape[2]}."
            )

        sequence_length = (
            tokens.shape[1]
        )

        # 1. Token embedding
        embeddings = (
            self.token_embedding(
                tokens
            )
        )

        # 2. 2D positional encoding
        embeddings = (
            add_sequence_positional_encoding(
                embeddings,
                self.n_frequency_patches
            )
        )

        # 3. Causal attention mask
        causal_mask = (
            self.create_causal_mask(
                sequence_length,
                embeddings.device
            )
        )

        # 4. Transformer
        transformer_output = (
            self.transformer(
                embeddings,
                mask=causal_mask,
                is_causal=True
            )
        )

        # 5. Predict STFT token
        predictions = (
            self.output_projection(
                transformer_output
            )
        )

        return predictions

if __name__ == "__main__":

    # Testing
    from construct_dataset import (
        create_slakh_datasets,
        SLAKH2100_REDUX_16K_TRAIN,
        SLAKH2100_REDUX_16K_VALIDATION,
        SLAKH2100_REDUX_16K_TEST
    )

    SET_LIMIT = True
    MAX_TRACKS = 1

    # 1. Create datasets
    (
        training_dataset,
        validation_dataset,
        test_dataset
    ) = create_slakh_datasets(
        training_path=(
            SLAKH2100_REDUX_16K_TRAIN
        ),
        validation_path=(
            SLAKH2100_REDUX_16K_VALIDATION
        ),
        test_path=(
            SLAKH2100_REDUX_16K_TEST
        ),
        set_limit=SET_LIMIT,
        maximum_tracks=MAX_TRACKS
    )

    print("\nTraining dataset size:")
    print(
        len(training_dataset)
    )

    # 2. Retrieve one example
    (
        context_tokens,
        target_tokens
    ) = training_dataset[0]

    print("\nIndividual sample")
    print("=" * 60)

    print(
        "Context shape:",
        context_tokens.shape
    )

    print(
        "Target shape:",
        target_tokens.shape
    )

    # Add batch dimension.
    context_tokens = (
        context_tokens.unsqueeze(0)
    )

    target_tokens = (
        target_tokens.unsqueeze(0)
    )

    print("\nAfter adding batch dimension")
    print("=" * 60)

    print(
        "Context shape:",
        context_tokens.shape
    )

    print(
        "Target shape:",
        target_tokens.shape
    )

    # 3. Combine context and target
    full_sequence = torch.cat(
        [
            context_tokens,
            target_tokens
        ],
        dim=1
    )

    print("\nFull sequence")
    print("=" * 60)

    print(
        "Full sequence shape:",
        full_sequence.shape
    )

    # 4. Shift for autoregressive training
    model_input = full_sequence[
        :,
        :-1,
        :
    ]

    expected_output = full_sequence[
        :,
        1:,
        :
    ]

    print("\nAutoregressive shift")
    print("=" * 60)

    print(
        "Model input shape:",
        model_input.shape
    )

    print(
        "Expected output shape:",
        expected_output.shape
    )

    # 5. Create Transformer
    model = (
        AutoregressiveDrumTransformer(
            token_dimension=512,
            embedding_dimension=256,
            number_of_heads=8,
            number_of_layers=4,
            feed_forward_dimension=1024,
            dropout=0.1,
            n_frequency_patches=16
        )
    )

    print("\nModel created")
    print("=" * 60)

    # 6. One forward pass
    model.eval()

    with torch.no_grad():

        predictions = (
            model(
                model_input
            )
        )

    print("\nForward pass")
    print("=" * 60)

    print(
        "Input shape:",
        model_input.shape
    )

    print(
        "Prediction shape:",
        predictions.shape
    )

    print(
        "Expected output shape:",
        expected_output.shape
    )

    print(
        "Prediction dtype:",
        predictions.dtype
    )

    # 7. Check continuation section
    context_length = (
        context_tokens.shape[1]
    )

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

    print("\nTarget continuation section")
    print("=" * 60)

    print(
        "Target predictions shape:",
        target_predictions.shape
    )

    print(
        "Target expected shape:",
        target_expected.shape
    )

    print("\nNumerical ranges")
    print("=" * 60)

    print(
        "Input min/max:",
        model_input.min().item(),
        model_input.max().item()
    )

    print(
        "Target min/max:",
        target_expected.min().item(),
        target_expected.max().item()
    )

    print(
        "Prediction min/max:",
        predictions.min().item(),
        predictions.max().item()
    )

    print(
        "Input mean/std:",
        model_input.mean().item(),
        model_input.std().item()
    )

    print(
        "Target mean/std:",
        target_expected.mean().item(),
        target_expected.std().item()
    )

    # 8. Calculate example loss
    mse_loss_function = (
        nn.MSELoss()
    )

    smooth_l1_loss_function = (
        nn.SmoothL1Loss(
            beta=3.0
        )
    )

    mse_loss = (
        mse_loss_function(
            target_predictions,
            target_expected
        )
    )

    smooth_l1_loss = (
        smooth_l1_loss_function(
            target_predictions,
            target_expected
        )
    )

    print("\nLoss comparison")
    print("=" * 60)

    print(
        "MSE loss:",
        mse_loss.item()
    )

    print(
        "Smooth L1 loss:",
        smooth_l1_loss.item()
    )