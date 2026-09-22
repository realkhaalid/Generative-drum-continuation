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

def add_sequence_positional_encoding(
    embeddings,
    n_frequency_patches
):
    """
    Adds continuous 2D time-frequency positional
    encoding to an arbitrary-length token sequence.
    """

    _, sequence_length, embedding_dim = (
        embeddings.shape
    )

    token_positions = torch.arange(
        sequence_length,
        device=embeddings.device
    )

    frequency_indices = (
        token_positions
        % n_frequency_patches
    )

    time_indices = (
        token_positions
        // n_frequency_patches
    )

    n_time_positions = (
        int(
            time_indices.max().item()
        )
        + 1
    )

    frequency_encoding = positional_encoding(
        n_frequency_patches,
        embedding_dim,
        embeddings.device,
        embeddings.dtype
    )

    time_encoding = positional_encoding(
        n_time_positions,
        embedding_dim,
        embeddings.device,
        embeddings.dtype
    )

    pos_encoding = (
        frequency_encoding[
            frequency_indices
        ]
        +
        time_encoding[
            time_indices
        ]
    )

    pos_encoding = (
        pos_encoding.unsqueeze(0)
    )

    return (
        embeddings
        + pos_encoding
    )