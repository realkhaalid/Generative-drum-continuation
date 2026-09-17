import torch
import math

def positional_encoding(
    num_positions,
    embedding_dim,
    device,
    dtype
):
    """
    Creates standard sinusoidal positional encoding
    for one positional dimension.
    """

    pos_encoding = torch.zeros(
        num_positions,
        embedding_dim,
        device=device,
        dtype=dtype
    )

    position = torch.arange(
        num_positions,
        device=device,
        dtype=dtype
    ).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(
            0,
            embedding_dim,
            2,
            device=device,
            dtype=dtype
        )
        * (
            -math.log(10000.0)
            / embedding_dim
        )
    )

    pos_encoding[:, 0::2] = torch.sin(
        position * div_term
    )

    pos_encoding[:, 1::2] = torch.cos(
        position
        * div_term[
            :pos_encoding[:, 1::2].shape[1]
        ]
    )

    return pos_encoding

def frequency_time_positional_encoding(
    n_frequency_patches,
    n_time_patches,
    embedding_dim,
    device,
    dtype
):
    """
    Creates 2D sinusoidal positional encoding
    for frequency-time STFT tokens.
    """

    frequency_encoding = positional_encoding(
        n_frequency_patches,
        embedding_dim,
        device,
        dtype
    )

    time_encoding = positional_encoding(
        n_time_patches,
        embedding_dim,
        device,
        dtype
    )

    positional_encodings = []

    for frequency_index in range(
        n_frequency_patches
    ):
        for time_index in range(
            n_time_patches
        ):

            encoding = (
                frequency_encoding[
                    frequency_index
                ]
                +
                time_encoding[
                    time_index
                ]
            )

            positional_encodings.append(
                encoding
            )

    positional_encodings = torch.stack(
        positional_encodings
    )

    return positional_encodings.unsqueeze(0)

def add_positional_encoding(
    embeddings,
    n_frequency_patches,
    n_time_patches
):
    """
    Adds frequency and temporal positional
    information to STFT token embeddings.
    """

    _, num_tokens, embedding_dim = (
        embeddings.shape
    )

    expected_tokens = (
        n_frequency_patches
        * n_time_patches
    )

    if num_tokens != expected_tokens:
        raise ValueError(
            f"Expected {expected_tokens} tokens, "
            f"but received {num_tokens}."
        )

    pos_encoding = frequency_time_positional_encoding(
        n_frequency_patches,
        n_time_patches,
        embedding_dim,
        embeddings.device,
        embeddings.dtype
    )

    return embeddings + pos_encoding