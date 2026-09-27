import torch
import torch.nn as nn

from vector_quantize_pytorch import (
    ResidualVQ
)

from data_processing_pipeline import (
    TIME_PATCH_SIZE,
    FREQUENCY_PATCH_SIZE
)

PATCH_DIMENSION = (
    2
    * FREQUENCY_PATCH_SIZE
    * TIME_PATCH_SIZE
)

class STFTTokenizer(
    nn.Module
):

    def __init__(
        self,
        patch_dimension=PATCH_DIMENSION,
        latent_dimension=256,
        codebook_size=1024
    ):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(
                patch_dimension,
                512
            ),

            nn.GELU(),

            nn.Linear(
                512,
                latent_dimension
            )
        )

        self.quantizer = ResidualVQ(
            dim=latent_dimension,
            num_quantizers=8,
            codebook_size=codebook_size,
            kmeans_init=True,
            kmeans_iters=10,
            threshold_ema_dead_code=2
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                latent_dimension,
                512
            ),

            nn.GELU(),

            nn.Linear(
                512,
                patch_dimension
            )
        )

    def forward(
        self,
        patches
    ):

        encoded = (
            self.encoder(
                patches
            )
        )

        (
            quantized,
            token_ids,
            commitment_loss
        ) = (
            self.quantizer(
                encoded
            )
        )

        # print(
        #     "Encoded shape:",
        #     encoded.shape
        # )

        # print(
        #     "Quantized shape:",
        #     quantized.shape
        # )

        reconstructed_patches = (
            self.decoder(
                quantized
            )
        )

        # reconstruction_loss = (
        #     torch.nn.functional.mse_loss(
        #         reconstructed_patches,
        #         patches
        #     )
        # )

        commitment_loss = (
            commitment_loss.mean()
        )

        # total_loss = (
        #     reconstruction_loss
        #     + commitment_loss
        # )

        return (
            reconstructed_patches,
            token_ids,
            #reconstruction_loss,
            commitment_loss,
            #total_loss
        )

if __name__ == "__main__":


    # Testing REMEMBER TO UNCOMMENT CODE ABOVE
    from construct_dataset import (
        create_slakh_datasets,
        SLAKH2100_REDUX_16K_TRAIN,
        SLAKH2100_REDUX_16K_VALIDATION
    )

    SET_LIMIT = True
    MAX_TRACKS = 1

    # 1. Create datasets
    (
        training_dataset,
        validation_dataset
    ) = create_slakh_datasets(
        training_path=(
            SLAKH2100_REDUX_16K_TRAIN
        ),
        validation_path=(
            SLAKH2100_REDUX_16K_VALIDATION
        ),
        set_limit=SET_LIMIT,
        maximum_tracks=MAX_TRACKS
    )

    print(
        "\nTraining dataset size:"
    )

    print(
        len(
            training_dataset
        )
    )

    # 2. Retrieve one example
    (
        context_tokens,
        target_tokens
    ) = (
        training_dataset[
            0
        ]
    )

    print(
        "\nIndividual sample"
    )

    print(
        "=" * 60
    )

    print(
        "Context shape:",
        context_tokens.shape
    )

    print(
        "Target shape:",
        target_tokens.shape
    )

    # 3. Add batch dimension
    context_tokens = (
        context_tokens.unsqueeze(
            0
        )
    )

    target_tokens = (
        target_tokens.unsqueeze(
            0
        )
    )

    print(
        "\nAfter adding batch dimension"
    )

    print(
        "=" * 60
    )

    print(
        "Context shape:",
        context_tokens.shape
    )

    print(
        "Target shape:",
        target_tokens.shape
    )

    # 4. Create tokenizer
    tokenizer = (
        STFTTokenizer(
            patch_dimension=PATCH_DIMENSION,
            latent_dimension=256,
            codebook_size=1024
        )
    )

    print(
        "\nTokenizer created"
    )

    print(
        "=" * 60
    )

    # 5. Put tokenizer in training mode
    tokenizer.train()

    # 6. Forward pass in training mode
    (
        reconstructed_context,
        context_token_ids,
        reconstruction_loss,
        commitment_loss,
        total_loss
    ) = (
        tokenizer(
            context_tokens
        )
    )

    print(
        "\nTokenizer training-mode forward pass"
    )

    print(
        "=" * 60
    )

    print(
        "Input context shape:",
        context_tokens.shape
    )

    print(
        "Reconstructed context shape:",
        reconstructed_context.shape
    )

    print(
        "Token IDs shape:",
        context_token_ids.shape
    )

    print(
        "Token ID dtype:",
        context_token_ids.dtype
    )

    print(
        "Reconstruction loss:",
        reconstruction_loss.item()
    )

    print(
        "Commitment loss:",
        commitment_loss.item()
    )

    print(
        "Total loss:",
        total_loss.item()
    )

    # 7. Check gradient tracking
    print(
        "\nGradient tracking"
    )

    print(
        "=" * 60
    )

    print(
        "Reconstructed context "
        "requires gradient:",
        reconstructed_context.requires_grad
    )

    print(
        "Reconstruction loss "
        "requires gradient:",
        reconstruction_loss.requires_grad
    )

    print(
        "Commitment loss "
        "requires gradient:",
        commitment_loss.requires_grad
    )

    print(
        "Total loss "
        "requires gradient:",
        total_loss.requires_grad
    )

    # 8. Check token ID values
    print(
        "\nToken ID information"
    )

    print(
        "=" * 60
    )

    print(
        "Minimum token ID:",
        context_token_ids
        .min()
        .item()
    )

    print(
        "Maximum token ID:",
        context_token_ids
        .max()
        .item()
    )

    unique_token_ids = (
        torch.unique(
            context_token_ids
        )
    )

    print(
        "Unique token IDs:",
        unique_token_ids.numel()
    )

    print(
        "First 20 token IDs:"
    )

    print(
        context_token_ids[
            0,
            :20
        ]
    )

    # 9. Clear any existing gradients
    tokenizer.zero_grad()

    # 10. Backpropagation test
    total_loss.backward()

    print(
        "\nBackpropagation test"
    )

    print(
        "=" * 60
    )

    print(
        "Encoder layer 1 weight gradient exists:",
        tokenizer.encoder[
            0
        ].weight.grad
        is not None
    )

    print(
        "Encoder layer 1 bias gradient exists:",
        tokenizer.encoder[
            0
        ].bias.grad
        is not None
    )

    print(
        "Encoder layer 2 weight gradient exists:",
        tokenizer.encoder[
            2
        ].weight.grad
        is not None
    )

    print(
        "Encoder layer 2 bias gradient exists:",
        tokenizer.encoder[
            2
        ].bias.grad
        is not None
    )

    print(
        "Decoder layer 1 weight gradient exists:",
        tokenizer.decoder[
            0
        ].weight.grad
        is not None
    )

    print(
        "Decoder layer 1 bias gradient exists:",
        tokenizer.decoder[
            0
        ].bias.grad
        is not None
    )

    print(
        "Decoder layer 2 weight gradient exists:",
        tokenizer.decoder[
            2
        ].weight.grad
        is not None
    )

    print(
        "Decoder layer 2 bias gradient exists:",
        tokenizer.decoder[
            2
        ].bias.grad
        is not None
    )

    # 11. Inspect gradient magnitudes
    print(
        "\nGradient magnitudes"
    )

    print(
        "=" * 60
    )

    print(
        "Encoder layer 1 gradient mean:",
        tokenizer.encoder[
            0
        ]
        .weight
        .grad
        .abs()
        .mean()
        .item()
    )

    print(
        "Encoder layer 1 gradient max:",
        tokenizer.encoder[
            0
        ]
        .weight
        .grad
        .abs()
        .max()
        .item()
    )

    print(
        "Encoder layer 2 gradient mean:",
        tokenizer.encoder[
            2
        ]
        .weight
        .grad
        .abs()
        .mean()
        .item()
    )

    print(
        "Encoder layer 2 gradient max:",
        tokenizer.encoder[
            2
        ]
        .weight
        .grad
        .abs()
        .max()
        .item()
    )

    print(
        "Decoder layer 1 gradient mean:",
        tokenizer.decoder[
            0
        ]
        .weight
        .grad
        .abs()
        .mean()
        .item()
    )

    print(
        "Decoder layer 1 gradient max:",
        tokenizer.decoder[
            0
        ]
        .weight
        .grad
        .abs()
        .max()
        .item()
    )

    print(
        "Decoder layer 2 gradient mean:",
        tokenizer.decoder[
            2
        ]
        .weight
        .grad
        .abs()
        .mean()
        .item()
    )

    print(
        "Decoder layer 2 gradient max:",
        tokenizer.decoder[
            2
        ]
        .weight
        .grad
        .abs()
        .max()
        .item()
    )

    # 12. Final checks
    print(
        "\nFinal checks"
    )

    print(
        "=" * 60
    )

    print(
        "Input and reconstruction "
        "shapes match:",
        context_tokens.shape
        == reconstructed_context.shape
    )

    print(
        "Token count matches "
        "input sequence length:",
        context_token_ids.shape[1]
        == context_tokens.shape[1]
    )

    print(
        "All reconstructed values finite:",
        torch.isfinite(
            reconstructed_context
        )
        .all()
        .item()
    )

    print(
        "Total loss finite:",
        torch.isfinite(
            total_loss
        )
        .item()
    )