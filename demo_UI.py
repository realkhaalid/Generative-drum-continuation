from io import BytesIO
from pathlib import Path

import hashlib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import tempfile
import time

from data_processing_pipeline import (
    load_and_validate_audio,
    find_audio_context_target_pairs,
    calculate_loudness_db,
    convert_to_stft_spectrogram,
    TARGET_SAMPLE_RATE,
    CLIP_DURATION_SECONDS,
    HOPLENGTH,
    LOUDNESS_THRESHOLD_DB
)

from inference_pipeline import (
    DrumContinuationInferencePipeline
)


# Page Configuration
st.set_page_config(
    page_title=(
        "Generative Drum Continuation"
    ),
    page_icon="🥁",
    layout="wide"
)


# Configuration
TOKENIZER_CHECKPOINT_PATH = (
    "tokenizer_checkpoints/"
    "stft_tokenizer_2_retrained_best.pt"
)

TRANSFORMER_CHECKPOINT_PATH = (
    "transformer_checkpoints_retrained/"
    "autoregressive_drum_transformer_discrete_retrained_best.pt"
)

DEFAULT_TEMPERATURE = (
    0.8
)

DEFAULT_TOP_K = (
    20
)

DEFAULT_TOP_P = (
    0.9
)


# Save Uploaded File
def save_uploaded_file(
    uploaded_file
):

    file_bytes = (
        uploaded_file.getvalue()
    )

    file_hash = (
        hashlib.sha256(
            file_bytes
        )
        .hexdigest()
    )

    suffix = (
        Path(
            uploaded_file.name
        )
        .suffix
        .lower()
    )

    temporary_directory = (
        Path(
            tempfile.gettempdir()
        )
        / "drum_continuation_demo"
    )

    temporary_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = (
        temporary_directory
        / (
            f"{file_hash}"
            f"{suffix}"
        )
    )

    if not file_path.exists():

        file_path.write_bytes(
            file_bytes
        )

    return file_path


# Convert Audio to WAV Bytes
def audio_to_bytes(
    audio,
    sample_rate
):

    buffer = (
        BytesIO()
    )

    sf.write(
        buffer,
        audio,
        sample_rate,
        format="WAV"
    )

    buffer.seek(
        0
    )

    return (
        buffer.getvalue()
    )


# Process Uploaded Audio
@st.cache_data(
    show_spinner=False
)
def process_uploaded_audio(
    file_path_string
):

    file_path = (
        Path(
            file_path_string
        )
    )

    (
        audio,
        sample_rate
    ) = (
        load_and_validate_audio(
            file_path
        )
    )

    (
        pair_start_samples,
        statistics
    ) = (
        find_audio_context_target_pairs(
            audio,
            sample_rate
        )
    )

    clip_length = int(
        sample_rate
        * CLIP_DURATION_SECONDS
    )

    audio_pairs = []

    for (
        pair_index,
        start_sample
    ) in enumerate(
        pair_start_samples,
        start=1
    ):

        context_start = (
            start_sample
        )

        context_end = (
            context_start
            + clip_length
        )

        target_start = (
            context_end
        )

        target_end = (
            target_start
            + clip_length
        )

        if (
            target_end
            > len(
                audio
            )
        ):

            continue

        context_audio = (
            audio[
                context_start:
                context_end
            ]
        )

        target_audio = (
            audio[
                target_start:
                target_end
            ]
        )

        context_loudness = (
            calculate_loudness_db(
                context_audio
            )
        )

        target_loudness = (
            calculate_loudness_db(
                target_audio
            )
        )

        audio_pairs.append({
            "pair_index":
                pair_index,

            "start_sample":
                start_sample,

            "context_start":
                context_start,

            "context_end":
                context_end,

            "target_start":
                target_start,

            "target_end":
                target_end,

            "context_audio":
                context_audio,

            "target_audio":
                target_audio,

            "context_loudness":
                context_loudness,

            "target_loudness":
                target_loudness
        })

    return (
        audio,
        sample_rate,
        audio_pairs,
        statistics
    )


# Load Inference Pipeline
@st.cache_resource(
    show_spinner=False
)
def get_inference_pipeline():

    inference_pipeline = (
        DrumContinuationInferencePipeline(
            tokenizer_checkpoint_path=(
                TOKENIZER_CHECKPOINT_PATH
            ),
            transformer_checkpoint_path=(
                TRANSFORMER_CHECKPOINT_PATH
            )
        )
    )

    return inference_pipeline


# Create STFT Figure
def create_stft_figure(
    audio,
    sample_rate,
    title
):

    (
        real_tensor,
        imaginary_tensor
    ) = (
        convert_to_stft_spectrogram(
            audio
        )
    )

    real = (
        real_tensor
        .detach()
        .cpu()
        .numpy()
    )

    imaginary = (
        imaginary_tensor
        .detach()
        .cpu()
        .numpy()
    )

    magnitude = (
        np.sqrt(
            real ** 2
            + imaginary ** 2
        )
    )

    maximum_magnitude = (
        np.max(
            magnitude
        )
    )

    if maximum_magnitude <= 0:

        maximum_magnitude = (
            1.0
        )

    magnitude_db = (
        librosa.amplitude_to_db(
            magnitude,
            ref=maximum_magnitude
        )
    )

    (
        figure,
        axis
    ) = (
        plt.subplots(
            figsize=(
                12,
                4
            )
        )
    )

    image = (
        librosa.display.specshow(
            magnitude_db,
            sr=sample_rate,
            hop_length=HOPLENGTH,
            x_axis="time",
            y_axis="linear",
            ax=axis
        )
    )

    axis.set_title(
        title
    )

    figure.colorbar(
        image,
        ax=axis,
        format="%+2.0f dB"
    )

    figure.tight_layout()

    return figure


# Page Title
st.title(
    "Generative Drum Continuation"
)

st.write(
    "Upload drum audio, select a valid "
    "5-second context and target pair, and "
    "generate a new 5-second continuation "
    "using the trained autoregressive Transformer."
)


# Upload Audio
uploaded_file = (
    st.file_uploader(
        "Select Audio File",
        type=[
            "wav",
            "flac"
        ]
    )
)


if uploaded_file is None:

    st.info(
        "Upload a WAV or FLAC audio file "
        "to begin."
    )

    st.stop()


audio_file_path = (
    save_uploaded_file(
        uploaded_file
    )
)


# Load Audio
try:

    (
        audio,
        sample_rate,
        audio_pairs,
        processing_statistics
    ) = (
        process_uploaded_audio(
            str(
                audio_file_path
            )
        )
    )

except Exception as error:

    st.error(
        f"Audio processing failed: "
        f"{error}"
    )

    st.stop()


duration = (
    len(
        audio
    )
    / sample_rate
)


# Audio Preview
st.subheader(
    "Audio Preview"
)

(
    preview_col1,
    preview_col2
) = (
    st.columns(
        [
            2,
            1
        ]
    )
)


with preview_col1:

    st.audio(
        uploaded_file.getvalue()
    )


with preview_col2:

    st.write(
        f"**File:** "
        f"{uploaded_file.name}"
    )

    st.write(
        f"**Duration:** "
        f"{duration:.2f} seconds"
    )

    st.write(
        f"**Sample Rate:** "
        f"{sample_rate} Hz"
    )

    st.write(
        f"**Valid Pairs:** "
        f"{len(audio_pairs)}"
    )


# Data Processing Demo
st.header(
    "Data Processing Demo"
)

st.write(
    "The uploaded audio is divided into "
    "contiguous 5-second context and "
    "5-second target pairs. Pairs containing "
    "audio below the loudness threshold are "
    "removed before inference."
)


# Processing Overview
(
    pipeline_col1,
    arrow_col1,
    pipeline_col2,
    arrow_col2,
    pipeline_col3
) = (
    st.columns(
        [
            3,
            1,
            3,
            1,
            3
        ]
    )
)


with pipeline_col1:

    st.markdown(
        f"""
        ### Audio File

        **{duration:.2f} seconds**
        """
    )


with arrow_col1:

    st.markdown(
        "### →"
    )


with pipeline_col2:

    st.markdown(
        f"""
        ### Candidate Pairs

        **{processing_statistics["total_pairs"]}**
        """
    )


with arrow_col2:

    st.markdown(
        "### →"
    )


with pipeline_col3:

    st.markdown(
        f"""
        ### Valid Pairs

        **{processing_statistics["valid_pairs"]}**
        """
    )


# Processing Statistics
st.subheader(
    "Processing Statistics"
)

(
    stats_col1,
    stats_col2,
    stats_col3
) = (
    st.columns(
        3
    )
)


with stats_col1:

    st.metric(
        "Candidate Pairs",
        processing_statistics[
            "total_pairs"
        ]
    )


with stats_col2:

    st.metric(
        "Valid Pairs",
        processing_statistics[
            "valid_pairs"
        ]
    )


with stats_col3:

    st.metric(
        "Rejected Quiet Pairs",
        processing_statistics[
            "rejected_quiet_pairs"
        ]
    )


st.caption(
    "Pairs are rejected when either the "
    "context or target falls below "
    f"{LOUDNESS_THRESHOLD_DB:.2f} dBFS."
)


# Context Target Pair Selection
st.subheader(
    "Context-Target Pair"
)


if len(
    audio_pairs
) == 0:

    st.warning(
        "No valid context-target pairs "
        "were found."
    )

    st.stop()


pair_options = {}


for pair in (
    audio_pairs
):

    start_time = (
        pair[
            "start_sample"
        ]
        / sample_rate
    )

    context_end_time = (
        start_time
        + CLIP_DURATION_SECONDS
    )

    target_end_time = (
        context_end_time
        + CLIP_DURATION_SECONDS
    )

    label = (
        f"Pair {pair['pair_index']} | "
        f"Context: "
        f"{start_time:.1f}s - "
        f"{context_end_time:.1f}s | "
        f"Target: "
        f"{context_end_time:.1f}s - "
        f"{target_end_time:.1f}s"
    )

    pair_options[
        label
    ] = (
        pair
    )


selected_pair_label = (
    st.selectbox(
        "Select Context-Target Pair",
        list(
            pair_options.keys()
        )
    )
)


selected_pair = (
    pair_options[
        selected_pair_label
    ]
)


context_audio = (
    selected_pair[
        "context_audio"
    ]
)

target_audio = (
    selected_pair[
        "target_audio"
    ]
)


# Selected Audio Pair
(
    context_col,
    target_col
) = (
    st.columns(
        2
    )
)


with context_col:

    st.write(
        "**Context Audio**"
    )

    st.audio(
        audio_to_bytes(
            context_audio,
            sample_rate
        ),
        format="audio/wav"
    )

    st.caption(
        f"Loudness: "
        f"{selected_pair['context_loudness']:.2f} "
        f"dBFS"
    )


with target_col:

    st.write(
        "**Original Target Audio**"
    )

    st.audio(
        audio_to_bytes(
            target_audio,
            sample_rate
        ),
        format="audio/wav"
    )

    st.caption(
        f"Loudness: "
        f"{selected_pair['target_loudness']:.2f} "
        f"dBFS"
    )


# STFT Visualisation
st.subheader(
    "STFT Visualisation"
)

st.write(
    "The magnitude of the complex STFT is "
    "displayed for the selected context and "
    "original target audio."
)


(
    stft_col1,
    stft_col2
) = (
    st.columns(
        2
    )
)


with stft_col1:

    context_figure = (
        create_stft_figure(
            context_audio,
            sample_rate,
            "Context STFT"
        )
    )

    st.pyplot(
        context_figure,
        use_container_width=True
    )

    plt.close(
        context_figure
    )


with stft_col2:

    target_figure = (
        create_stft_figure(
            target_audio,
            sample_rate,
            "Original Target STFT"
        )
    )

    st.pyplot(
        target_figure,
        use_container_width=True
    )

    plt.close(
        target_figure
    )


# Inference
st.header(
    "Inference"
)

st.write(
    "Generate the next 5 seconds of drum "
    "audio conditioned on the selected "
    "5-second context."
)


# Sampling Configuration
(
    parameter_col1,
    parameter_col2,
    parameter_col3
) = (
    st.columns(
        3
    )
)


with parameter_col1:

    temperature = (
        st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.5,
            value=(
                DEFAULT_TEMPERATURE
            ),
            step=0.05
        )
    )


with parameter_col2:

    top_k = (
        st.slider(
            "Top-k",
            min_value=1,
            max_value=100,
            value=(
                DEFAULT_TOP_K
            ),
            step=1
        )
    )


with parameter_col3:

    top_p = (
        st.slider(
            "Top-p",
            min_value=0.1,
            max_value=1.0,
            value=(
                DEFAULT_TOP_P
            ),
            step=0.05
        )
    )


st.caption(
    "Temperature controls the sharpness of "
    "the probability distribution, Top-k "
    "restricts sampling to the highest ranked "
    "tokens and Top-p restricts sampling using "
    "cumulative probability."
)


# Run Inference
run_inference = (
    st.button(
        "Generate Continuation",
        type="primary",
        use_container_width=True
    )
)


if run_inference:

    if not Path(
        TOKENIZER_CHECKPOINT_PATH
    ).exists():

        st.error(
            "Tokenizer checkpoint could not "
            "be found: "
            f"{TOKENIZER_CHECKPOINT_PATH}"
        )

        st.stop()


    if not Path(
        TRANSFORMER_CHECKPOINT_PATH
    ).exists():

        st.error(
            "Transformer checkpoint could not "
            "be found: "
            f"{TRANSFORMER_CHECKPOINT_PATH}"
        )

        st.stop()


    try:

        inference_pipeline = (
            get_inference_pipeline()
        )

        # Update sampling hyperparameters
        inference_pipeline.temperature = (
            temperature
        )

        inference_pipeline.top_k = (
            top_k
        )

        inference_pipeline.top_p = (
            top_p
        )

        progress_bar = (
            st.progress(
                0,
                text=(
                    "Preparing inference..."
                )
            )
        )

        start_time = (
            time.perf_counter()
        )


        # Convert Context to STFT Patches
        progress_bar.progress(
            10,
            text=(
                "Converting context audio "
                "to STFT patches..."
            )
        )

        context_patches = (
            inference_pipeline
            .convert_context_to_stft_patches(
                context_audio
            )
        )


        # Tokenise Context
        progress_bar.progress(
            20,
            text=(
                "Tokenising context audio..."
            )
        )

        context_token_ids = (
            inference_pipeline
            .tokenize_context(
                context_patches
            )
        )


        # Prepare Generation Progress Callback
        progress_bar.progress(
            30,
            text=(
                "Generating drum continuation..."
            )
        )


        def update_generation_progress(
            generated_tokens,
            total_tokens
        ):

            generation_fraction = (
                generated_tokens
                / total_tokens
            )

            progress_value = int(
                30
                + (
                    generation_fraction
                    * 55
                )
            )

            progress_bar.progress(
                progress_value,
                text=(
                    "Generating drum continuation... "
                    f"{generated_tokens}/"
                    f"{total_tokens} patches"
                )
            )


        # Generate Target Tokens
        generated_token_ids = (
            inference_pipeline
            .generate_target_tokens(
                context_token_ids=(
                    context_token_ids
                ),
                progress_callback=(
                    update_generation_progress
                )
            )
        )


        # Decode Generated Tokens
        progress_bar.progress(
            90,
            text=(
                "Decoding generated tokens..."
            )
        )

        (
            generated_audio,
            generated_patches
        ) = (
            inference_pipeline
            .reconstruct_generated_audio(
                generated_token_ids
            )
        )


        # Create Sequence Comparisons
        progress_bar.progress(
            95,
            text=(
                "Preparing audio comparisons..."
            )
        )

        original_sequence = (
            np.concatenate(
                [
                    context_audio,
                    target_audio
                ]
            )
        )

        generated_sequence = (
            np.concatenate(
                [
                    context_audio,
                    generated_audio
                ]
            )
        )


        inference_time = (
            time.perf_counter()
            - start_time
        )


        # Store Inference Results
        st.session_state[
            "drum_inference_results"
        ] = {
            "context_audio":
                context_audio,

            "target_audio":
                target_audio,

            "generated_audio":
                generated_audio,

            "original_sequence":
                original_sequence,

            "generated_sequence":
                generated_sequence,

            "generated_patches":
                generated_patches,

            "context_token_ids":
                context_token_ids
                .detach()
                .cpu(),

            "generated_token_ids":
                generated_token_ids
                .detach()
                .cpu(),

            "sample_rate":
                sample_rate,

            "inference_time":
                inference_time,

            "temperature":
                temperature,

            "top_k":
                top_k,

            "top_p":
                top_p,

            "selected_pair_label":
                selected_pair_label
        }


        progress_bar.progress(
            100,
            text=(
                "Inference complete."
            )
        )


    except Exception as error:

        st.error(
            f"Inference failed: "
            f"{error}"
        )


# Inference Results
if (
    "drum_inference_results"
    in st.session_state
):

    results = (
        st.session_state[
            "drum_inference_results"
        ]
    )

    st.subheader(
        "Inference Results"
    )


    # Original Context Audio
    st.write(
        "### Original Context Audio"
    )

    st.audio(
        audio_to_bytes(
            results[
                "context_audio"
            ],
            results[
                "sample_rate"
            ]
        ),
        format="audio/wav"
    )


    # Original Target Audio
    st.write(
        "### Original Next 5 Seconds"
    )

    st.caption(
        "The actual next 5 seconds of the "
        "uploaded drum audio used as the "
        "ground-truth comparison."
    )

    st.audio(
        audio_to_bytes(
            results[
                "target_audio"
            ],
            results[
                "sample_rate"
            ]
        ),
        format="audio/wav"
    )


    # Generated Target Audio
    st.write(
        "### Generated Next 5 Seconds"
    )

    st.audio(
        audio_to_bytes(
            results[
                "generated_audio"
            ],
            results[
                "sample_rate"
            ]
        ),
        format="audio/wav"
    )


    # Sequence Comparison
    st.write(
        "### Sequence Comparison"
    )

    (
        comparison_col1,
        comparison_col2
    ) = (
        st.columns(
            2
        )
    )


    with comparison_col1:

        st.write(
            "**Context + Original Target**"
        )

        st.audio(
            audio_to_bytes(
                results[
                    "original_sequence"
                ],
                results[
                    "sample_rate"
                ]
            ),
            format="audio/wav"
        )


    with comparison_col2:

        st.write(
            "**Context + Generated Continuation**"
        )

        st.audio(
            audio_to_bytes(
                results[
                    "generated_sequence"
                ],
                results[
                    "sample_rate"
                ]
            ),
            format="audio/wav"
        )


    # STFT Comparison
    st.write(
        "### STFT Comparison"
    )

    (
        original_target_stft_col,
        generated_stft_col
    ) = (
        st.columns(
            2
        )
    )


    with original_target_stft_col:

        original_target_figure = (
            create_stft_figure(
                results[
                    "target_audio"
                ],
                results[
                    "sample_rate"
                ],
                "Original Target STFT"
            )
        )

        st.pyplot(
            original_target_figure,
            use_container_width=True
        )

        plt.close(
            original_target_figure
        )


    with generated_stft_col:

        generated_figure = (
            create_stft_figure(
                results[
                    "generated_audio"
                ],
                results[
                    "sample_rate"
                ],
                "Generated Continuation STFT"
            )
        )

        st.pyplot(
            generated_figure,
            use_container_width=True
        )

        plt.close(
            generated_figure
        )


    # Inference Information
    st.write(
        "### Inference Information"
    )

    (
        info_col1,
        info_col2,
        info_col3,
        info_col4
    ) = (
        st.columns(
            4
        )
    )


    with info_col1:

        st.metric(
            "Inference Time",
            (
                f"{results['inference_time']:.2f}s"
            )
        )


    with info_col2:

        st.metric(
            "Temperature",
            results[
                "temperature"
            ]
        )


    with info_col3:

        st.metric(
            "Top-k",
            results[
                "top_k"
            ]
        )


    with info_col4:

        st.metric(
            "Top-p",
            results[
                "top_p"
            ]
        )