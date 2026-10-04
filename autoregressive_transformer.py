import torch
import torch.nn as nn

from transformers import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper
)

from positional_encoding_functions import (
    add_sequence_positional_encoding
)

#Config
EMBEDDING_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 4
FF_DIM = 1024
DROPOUT = 0.1
N_FREQUENCY_PATCHES = 16


class AutoregressiveDrumTransformer(
    nn.Module
):
    """
    Autoregressive causal Transformer for
    discrete drum-token continuation.

    Input:
        [batch_size, sequence_length, number_of_quantizers]

    Output:
        [batch_size,
         sequence_length,
         number_of_quantizers,
         codebook_size]

    Each sequence position represents one
    STFT time-frequency patch.

    Each patch contains one token ID from
    every Residual VQ quantizer.
    """

    def __init__(
        self,
        codebook_size=1024,
        number_of_quantizers=8,
        embedding_dimension=EMBEDDING_DIM,
        number_of_heads=NUM_HEADS,
        number_of_layers=NUM_LAYERS,
        feed_forward_dimension=FF_DIM,
        dropout=DROPOUT,
        n_frequency_patches=N_FREQUENCY_PATCHES
    ):

        super().__init__()

        self.codebook_size = (
            codebook_size
        )

        self.number_of_quantizers = (
            number_of_quantizers
        )

        self.embedding_dimension = (
            embedding_dimension
        )

        self.n_frequency_patches = (
            n_frequency_patches
        )

        # Each Residual VQ level has its
        # own independent codebook.
        #
        # Therefore each quantizer needs
        # its own token embedding table.
        self.token_embeddings = (
            nn.ModuleList(
                [
                    nn.Embedding(
                        codebook_size,
                        embedding_dimension
                    )
                    for _ in range(
                        number_of_quantizers
                    )
                ]
            )
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

        # Each Residual VQ level also has
        # its own output classification head.
        #
        # Every head predicts one of
        # codebook_size possible IDs.
        self.output_heads = (
            nn.ModuleList(
                [
                    nn.Linear(
                        embedding_dimension,
                        codebook_size
                    )
                    for _ in range(
                        number_of_quantizers
                    )
                ]
            )
        )

        self._initialize_weights()

    def _initialize_weights(
        self
    ):
        """
        Initializes trainable weight matrices.
        """

        for module in (
            self.modules()
        ):

            if isinstance(
                module,
                nn.Linear
            ):

                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.Embedding
            ):

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02
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

    def embed_tokens(
        self,
        token_ids
    ):
        """
        Embeds all Residual VQ token IDs.
        Embeddings from all Residual VQ
        levels are summed to create one
        representation for each STFT patch.
        """

        embedded_tokens = None

        for quantizer_index in range(
            self.number_of_quantizers
        ):

            quantizer_token_ids = (
                token_ids[
                    :,
                    :,
                    quantizer_index
                ]
            )

            quantizer_embeddings = (
                self.token_embeddings[
                    quantizer_index
                ](
                    quantizer_token_ids
                )
            )

            if embedded_tokens is None:

                embedded_tokens = (
                    quantizer_embeddings
                )

            else:

                embedded_tokens = (
                    embedded_tokens
                    + quantizer_embeddings
                )

        return embedded_tokens

    def project_to_logits(
        self,
        transformer_output
    ):
        """
        Converts Transformer hidden states
        into categorical logits for every
        Residual VQ quantizer.
        """

        quantizer_logits = []

        for quantizer_index in range(
            self.number_of_quantizers
        ):

            logits = (
                self.output_heads[
                    quantizer_index
                ](
                    transformer_output
                )
            )

            quantizer_logits.append(
                logits
            )

        logits = torch.stack(
            quantizer_logits,
            dim=2
        )

        return logits

    def forward(
        self,
        token_ids
    ):
        """
        Performs causal next-token prediction.
        """

        if token_ids.ndim != 3:

            raise ValueError(
                "Expected token ID tensor shape "
                "[batch_size, sequence_length, "
                "number_of_quantizers], but received "
                f"{token_ids.shape}."
            )

        if (
            token_ids.shape[
                2
            ]
            != self.number_of_quantizers
        ):

            raise ValueError(
                f"Expected "
                f"{self.number_of_quantizers} "
                f"quantizers, but received "
                f"{token_ids.shape[2]}."
            )

        if (
            token_ids.dtype
            != torch.long
        ):

            token_ids = (
                token_ids.long()
            )

        if (
            token_ids.min()
            < 0
            or token_ids.max()
            >= self.codebook_size
        ):

            raise ValueError(
                "Token IDs must be between "
                f"0 and {self.codebook_size - 1}."
            )

        sequence_length = (
            token_ids.shape[
                1
            ]
        )

        embeddings = (
            self.embed_tokens(
                token_ids
            )
        )

        embeddings = (
            add_sequence_positional_encoding(
                embeddings,
                self.n_frequency_patches
            )
        )

        causal_mask = (
            self.create_causal_mask(
                sequence_length,
                embeddings.device
            )
        )

        transformer_output = (
            self.transformer(
                embeddings,
                mask=causal_mask,
                is_causal=True
            )
        )

        logits = (
            self.project_to_logits(
                transformer_output
            )
        )

        return logits

    def generate_continuation(
        self,
        context_token_ids,
        number_of_target_tokens
    ):
        """
        Autoregressively generates future
        Residual VQ token IDs.
        """

        self.eval()

        if (
            context_token_ids.ndim
            == 2
        ):

            context_token_ids = (
                context_token_ids.unsqueeze(
                    0
                )
            )

        if (
            context_token_ids.ndim
            != 3
        ):

            raise ValueError(
                "Context token IDs must have shape "
                "[sequence_length, number_of_quantizers] "
                "or "
                "[batch_size, sequence_length, "
                "number_of_quantizers]."
            )

        if (
            context_token_ids.shape[
                2
            ]
            != self.number_of_quantizers
        ):

            raise ValueError(
                f"Expected "
                f"{self.number_of_quantizers} "
                f"quantizers, but received "
                f"{context_token_ids.shape[2]}."
            )

        device = next(
            self.parameters()
        ).device

        generated_sequence = (
            context_token_ids.to(
                device=device,
                dtype=torch.long
            )
        )

        with torch.no_grad():

            for _ in range(
                number_of_target_tokens
            ):

                logits = (
                    self(
                        generated_sequence
                    )
                )

                # Last patch position:
                #
                # [B, Q, codebook_size]
                next_token_logits = (
                    logits[
                        :,
                        -1,
                        :,
                        :
                    ]
                )

                # Greedy token selection
                next_token_ids = (
                    torch.argmax(
                        next_token_logits,
                        dim=-1
                    )
                )

                # Add sequence dimension:
                next_token_ids = (
                    next_token_ids.unsqueeze(
                        1
                    )
                )

                generated_sequence = (
                    torch.cat(
                        [
                            generated_sequence,
                            next_token_ids
                        ],
                        dim=1
                    )
                )

        generated_target = (
            generated_sequence[
                :,
                -number_of_target_tokens:,
                :
            ]
        )

        return generated_target

    def sample_inference_tokens(
        self,
        next_token_logits,
        temperature=0.8,
        top_k=20,
        top_p=0.9
    ):

        batch_size = (
            next_token_logits.shape[
                0
            ]
        )

        number_of_quantizers = (
            next_token_logits.shape[
                1
            ]
        )

        logits = (
            next_token_logits.reshape(
                batch_size
                * number_of_quantizers,
                self.codebook_size
            )
        )

        dummy_input_ids = torch.zeros(
            (
                batch_size
                * number_of_quantizers,
                1
            ),
            device=logits.device,
            dtype=torch.long
        )

        temperature_warper = (
            TemperatureLogitsWarper(
                temperature
            )
        )

        top_k_warper = (
            TopKLogitsWarper(
                top_k
            )
        )

        top_p_warper = (
            TopPLogitsWarper(
                top_p
            )
        )

        logits = (
            temperature_warper(
                dummy_input_ids,
                logits
            )
        )

        logits = (
            top_k_warper(
                dummy_input_ids,
                logits
            )
        )

        logits = (
            top_p_warper(
                dummy_input_ids,
                logits
            )
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=-1
            )
        )

        sampled_token_ids = (
            torch.multinomial(
                probabilities,
                num_samples=1
            )
        )

        sampled_token_ids = (
            sampled_token_ids.reshape(
                batch_size,
                number_of_quantizers
            )
        )

        return sampled_token_ids

    def generate_continuation_inference(
        self,
        context_token_ids,
        number_of_target_tokens,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        progress_callback=None
    ):

        self.eval()

        if (
            context_token_ids.ndim
            == 2
        ):

            context_token_ids = (
                context_token_ids.unsqueeze(
                    0
                )
            )

        if (
            context_token_ids.ndim
            != 3
        ):

            raise ValueError(
                "Context token IDs must have shape "
                "[sequence_length, number_of_quantizers] "
                "or "
                "[batch_size, sequence_length, "
                "number_of_quantizers]."
            )

        if (
            context_token_ids.shape[
                2
            ]
            != self.number_of_quantizers
        ):

            raise ValueError(
                f"Expected "
                f"{self.number_of_quantizers} "
                f"quantizers, but received "
                f"{context_token_ids.shape[2]}."
            )

        device = next(
            self.parameters()
        ).device

        generated_sequence = (
            context_token_ids.to(
                device=device,
                dtype=torch.long
            )
        )

        with torch.no_grad():

            for token_index in range(
                number_of_target_tokens
            ):

                logits = (
                    self(
                        generated_sequence
                    )
                )

                next_token_logits = (
                    logits[
                        :,
                        -1,
                        :,
                        :
                    ]
                )

                next_token_ids = (
                    self.sample_inference_tokens(
                        next_token_logits=(
                            next_token_logits
                        ),
                        temperature=(
                            temperature
                        ),
                        top_k=(
                            top_k
                        ),
                        top_p=(
                            top_p
                        )
                    )
                )

                next_token_ids = (
                    next_token_ids.unsqueeze(
                        1
                    )
                )

                generated_sequence = (
                    torch.cat(
                        [
                            generated_sequence,
                            next_token_ids
                        ],
                        dim=1
                    )
                )

                if (
                    progress_callback is not None
                    and (
                        token_index % 16 == 0
                        or token_index
                        == number_of_target_tokens - 1
                    )
                ):

                    progress_callback(
                        token_index + 1,
                        number_of_target_tokens
                    )

        generated_target = (
            generated_sequence[
                :,
                -number_of_target_tokens:,
                :
            ]
        )

        return generated_target

if __name__ == "__main__":


    # Testing REMEMBER TO UNCOMMENT CODE ABOVE
    from construct_dataset import (
        create_slakh_datasets,
        SLAKH2100_REDUX_16K_TRAIN,
        SLAKH2100_REDUX_16K_VALIDATION
    )

    from stft_tokeniser import STFTTokenizer

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

    TOKENIZER_CHECKPOINT_PATH = (
        "tokenizer_checkpoints/"
        "stft_tokenizer_2_best.pt"
    )

    (
        context_patches,
        target_patches
    ) = (
        training_dataset[
            0
        ]
    )

    print(
        "\nSTFT patch tensors"
    )

    print(
        "=" * 60
    )

    print(
        "Context patches shape:",
        context_patches.shape
    )

    print(
        "Target patches shape:",
        target_patches.shape
    )

    # Add batch dimension.
    context_patches = (
        context_patches.unsqueeze(
            0
        )
    )

    target_patches = (
        target_patches.unsqueeze(
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
        "Context patches shape:",
        context_patches.shape
    )

    print(
        "Target patches shape:",
        target_patches.shape
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = (
        STFTTokenizer()
        .to(
            device
        )
    )

    tokenizer_checkpoint = torch.load(
        TOKENIZER_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    tokenizer.load_state_dict(
        tokenizer_checkpoint[
            "model_state_dict"
        ]
    )

    tokenizer.eval()

    print(
        "\nTokenizer loaded"
    )

    print(
        "=" * 60
    )

    print(
        "Tokenizer checkpoint:",
        TOKENIZER_CHECKPOINT_PATH
    )

    print(
        "Device:",
        device
    )

    context_patches = (
        context_patches.to(
            device=device,
            dtype=torch.float32
        )
    )

    target_patches = (
        target_patches.to(
            device=device,
            dtype=torch.float32
        )
    )

    with torch.no_grad():

        (
            context_reconstruction,
            context_token_ids,
            context_commitment_loss
        ) = (
            tokenizer(
                context_patches
            )
        )

        (
            target_reconstruction,
            target_token_ids,
            target_commitment_loss
        ) = (
            tokenizer(
                target_patches
            )
        )

    print(
        "\nTokenizer output"
    )

    print(
        "=" * 60
    )

    print(
        "Context token IDs shape:",
        context_token_ids.shape
    )

    print(
        "Target token IDs shape:",
        target_token_ids.shape
    )

    print(
        "Context token dtype:",
        context_token_ids.dtype
    )

    print(
        "Target token dtype:",
        target_token_ids.dtype
    )

    print(
        "Context token min/max:",
        context_token_ids.min().item(),
        context_token_ids.max().item()
    )

    print(
        "Target token min/max:",
        target_token_ids.min().item(),
        target_token_ids.max().item()
    )

    full_sequence = torch.cat(
        [
            context_token_ids,
            target_token_ids
        ],
        dim=1
    )

    print(
        "\nFull discrete token sequence"
    )

    print(
        "=" * 60
    )

    print(
        "Full sequence shape:",
        full_sequence.shape
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

    print(
        "\nAutoregressive shift"
    )

    print(
        "=" * 60
    )

    print(
        "Model input shape:",
        model_input.shape
    )

    print(
        "Expected output shape:",
        expected_output.shape
    )

    model = (
        AutoregressiveDrumTransformer()
        .to(
            device
        )
    )

    print(
        "\nTransformer created"
    )

    print(
        "=" * 60
    )

    model.eval()

    with torch.no_grad():

        logits = (
            model(
                model_input
            )
        )

    print(
        "\nTransformer forward pass"
    )

    print(
        "=" * 60
    )

    print(
        "Model input shape:",
        model_input.shape
    )

    print(
        "Logits shape:",
        logits.shape
    )

    print(
        "Expected output shape:",
        expected_output.shape
    )

    print(
        "Expected logits shape:",
        (
            model_input.shape[0],
            model_input.shape[1],
            8,
            1024
        )
    )

    print(
        "Logits dtype:",
        logits.dtype
    )

    context_length = (
        context_token_ids.shape[
            1
        ]
    )

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

    print(
        "\nTarget continuation section"
    )

    print(
        "=" * 60
    )

    print(
        "Target logits shape:",
        target_logits.shape
    )

    print(
        "Target expected shape:",
        target_expected.shape
    )

    predicted_token_ids = (
        torch.argmax(
            target_logits,
            dim=-1
        )
    )

    print(
        "\nPredicted token IDs"
    )

    print(
        "=" * 60
    )

    print(
        "Predicted IDs shape:",
        predicted_token_ids.shape
    )

    print(
        "Predicted IDs dtype:",
        predicted_token_ids.dtype
    )

    print(
        "Predicted IDs min/max:",
        predicted_token_ids.min().item(),
        predicted_token_ids.max().item()
    )

    print(
        "First predicted patch:"
    )

    print(
        predicted_token_ids[
            0,
            0
        ]
    )

    print(
        "First expected patch:"
    )

    print(
        target_expected[
            0,
            0
        ]
    )

    cross_entropy_loss = (
        nn.CrossEntropyLoss()
    )

    quantizer_losses = []

    for quantizer_index in range(
        8
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
            cross_entropy_loss(
                quantizer_logits.reshape(
                    -1,
                    1024
                ),
                quantizer_targets.reshape(
                    -1
                )
            )
        )

        quantizer_losses.append(
            quantizer_loss
        )

        print(
            f"Quantizer {quantizer_index + 1} "
            f"CrossEntropy loss:",
            quantizer_loss.item()
        )

    total_loss = (
        torch.stack(
            quantizer_losses
        )
        .mean()
    )

    print(
        "\nLoss test"
    )

    print(
        "=" * 60
    )

    print(
        "Mean CrossEntropy loss:",
        total_loss.item()
    )

    print(
        "Loss finite:",
        torch.isfinite(
            total_loss
        ).item()
    )

    model.train()

    model.zero_grad()

    logits = (
        model(
            model_input
        )
    )

    target_logits = (
        logits[
            :,
            context_length - 1:,
            :,
            :
        ]
    )

    quantizer_losses = []

    for quantizer_index in range(
        8
    ):

        quantizer_loss = (
            cross_entropy_loss(
                target_logits[
                    :,
                    :,
                    quantizer_index,
                    :
                ]
                .reshape(
                    -1,
                    1024
                ),
                target_expected[
                    :,
                    :,
                    quantizer_index
                ]
                .reshape(
                    -1
                )
            )
        )

        quantizer_losses.append(
            quantizer_loss
        )

    total_loss = (
        torch.stack(
            quantizer_losses
        )
        .mean()
    )

    total_loss.backward()

    print(
        "\nBackpropagation test"
    )

    print(
        "=" * 60
    )

    print(
        "Embedding 1 gradient exists:",
        model.token_embeddings[
            0
        ].weight.grad
        is not None
    )

    print(
        "Embedding 8 gradient exists:",
        model.token_embeddings[
            7
        ].weight.grad
        is not None
    )

    print(
        "Output head 1 gradient exists:",
        model.output_heads[
            0
        ].weight.grad
        is not None
    )

    print(
        "Output head 8 gradient exists:",
        model.output_heads[
            7
        ].weight.grad
        is not None
    )

    print(
        "Embedding 1 gradient mean:",
        model.token_embeddings[
            0
        ]
        .weight
        .grad
        .abs()
        .mean()
        .item()
    )

    print(
        "Output head 1 gradient mean:",
        model.output_heads[
            0
        ]
        .weight
        .grad
        .abs()
        .mean()
        .item()
    )

    print(
        "\nFinal checks"
    )

    print(
        "=" * 60
    )

    print(
        "Input is torch.long:",
        model_input.dtype
        == torch.long
    )

    print(
        "Correct number of quantizers:",
        model_input.shape[
            2
        ]
        == 8
    )

    print(
        "Prediction sequence length correct:",
        logits.shape[
            1
        ]
        == model_input.shape[
            1
        ]
    )

    print(
        "Prediction quantizer count correct:",
        logits.shape[
            2
        ]
        == 8
    )

    print(
        "Prediction vocabulary size correct:",
        logits.shape[
            3
        ]
        == 1024
    )

    print(
        "All logits finite:",
        torch.isfinite(
            logits
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