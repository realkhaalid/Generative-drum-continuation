import torch
import numpy as np
import soundfile as sf

from pathlib import Path

from stft_tokeniser import (
    STFTTokenizer
)

from autoregressive_transformer import (
    AutoregressiveDrumTransformer
)

from data_processing_pipeline import (
    load_and_validate_audio,
    convert_to_stft_spectrogram,
    convert_stft_to_2d_patch_tokens,
    reconstruct_audio_from_stft_tokens,
    TARGET_SAMPLE_RATE,
    CLIP_DURATION_SECONDS
)


# Checkpoints
TOKENIZER_CHECKPOINT_PATH = (
    "tokenizer_checkpoints/"
    "stft_tokenizer_2_retrained_best.pt"
)

TRANSFORMER_CHECKPOINT_PATH = (
    "transformer_checkpoints_retrained/"
    "autoregressive_drum_transformer_discrete_retrained_best.pt"
)


# Inference configuration
TEMPERATURE = 0.8

TOP_K = 20

TOP_P = 0.9

OUTPUT_DIRECTORY = (
    "inference_outputs"
)


class DrumContinuationInferencePipeline:

    def __init__(
        self,
        tokenizer_checkpoint_path=(
            TOKENIZER_CHECKPOINT_PATH
        ),
        transformer_checkpoint_path=(
            TRANSFORMER_CHECKPOINT_PATH
        ),
        temperature=(
            TEMPERATURE
        ),
        top_k=(
            TOP_K
        ),
        top_p=(
            TOP_P
        ),
        output_directory=(
            OUTPUT_DIRECTORY
        ),
        device=None
    ):

        self.tokenizer_checkpoint_path = Path(
            tokenizer_checkpoint_path
        )

        self.transformer_checkpoint_path = Path(
            transformer_checkpoint_path
        )

        self.temperature = (
            temperature
        )

        self.top_k = (
            top_k
        )

        self.top_p = (
            top_p
        )

        self.output_directory = Path(
            output_directory
        )

        self.output_directory.mkdir(
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

        self.tokenizer = (
            self.load_tokenizer()
        )

        self.transformer = (
            self.load_transformer()
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

        return tokenizer


    def load_transformer(
        self
    ):

        checkpoint = torch.load(
            self.transformer_checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        model_configuration = (
            checkpoint[
                "model_configuration"
            ]
        )

        transformer = (
            AutoregressiveDrumTransformer(
                codebook_size=(
                    model_configuration[
                        "codebook_size"
                    ]
                ),
                number_of_quantizers=(
                    model_configuration[
                        "number_of_quantizers"
                    ]
                ),
                embedding_dimension=(
                    model_configuration[
                        "embedding_dimension"
                    ]
                ),
                number_of_heads=(
                    model_configuration[
                        "number_of_heads"
                    ]
                ),
                number_of_layers=(
                    model_configuration[
                        "number_of_layers"
                    ]
                ),
                feed_forward_dimension=(
                    model_configuration[
                        "feed_forward_dimension"
                    ]
                ),
                dropout=(
                    model_configuration[
                        "dropout"
                    ]
                ),
                n_frequency_patches=(
                    model_configuration[
                        "n_frequency_patches"
                    ]
                )
            )
            .to(
                self.device
            )
        )

        transformer.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        transformer.eval()

        for parameter in (
            transformer.parameters()
        ):

            parameter.requires_grad = False

        return transformer


    def load_context_audio(
        self,
        audio_file_path,
        start_time_seconds=0.0
    ):

        audio_file_path = Path(
            audio_file_path
        )

        (
            audio,
            sample_rate
        ) = (
            load_and_validate_audio(
                audio_file_path
            )
        )

        context_length = int(
            TARGET_SAMPLE_RATE
            * CLIP_DURATION_SECONDS
        )

        start_sample = int(
            start_time_seconds
            * TARGET_SAMPLE_RATE
        )

        end_sample = (
            start_sample
            + context_length
        )

        if (
            end_sample
            > len(
                audio
            )
        ):

            raise ValueError(
                "The selected audio does not "
                "contain a complete 5 second "
                "context segment."
            )

        context_audio = (
            audio[
                start_sample:
                end_sample
            ]
        )

        return (
            context_audio,
            sample_rate
        )


    def convert_context_to_stft_patches(
        self,
        context_audio
    ):

        (
            real_tensor,
            imaginary_tensor
        ) = (
            convert_to_stft_spectrogram(
                context_audio
            )
        )

        context_patches = (
            convert_stft_to_2d_patch_tokens(
                real_tensor,
                imaginary_tensor
            )
        )

        context_patches = (
            context_patches.to(
                device=self.device,
                dtype=torch.float32
            )
        )

        return context_patches


    def tokenize_context(
        self,
        context_patches
    ):

        self.tokenizer.eval()

        with torch.no_grad():

            encoded_patches = (
                self.tokenizer.encoder(
                    context_patches
                )
            )

            (
                quantized_latents,
                token_ids,
                commitment_loss
            ) = (
                self.tokenizer.quantizer(
                    encoded_patches
                )
            )

        token_ids = (
            token_ids.to(
                device=self.device,
                dtype=torch.long
            )
        )

        return token_ids


    def generate_target_tokens(
        self,
        context_token_ids,
        progress_callback=None
    ):

        number_of_target_tokens = (
            context_token_ids.shape[
                1
            ]
        )

        generated_token_ids = (
            self.transformer
            .generate_continuation_inference(
                context_token_ids=(
                    context_token_ids
                ),
                number_of_target_tokens=(
                    number_of_target_tokens
                ),
                temperature=(
                    self.temperature
                ),
                top_k=(
                    self.top_k
                ),
                top_p=(
                    self.top_p
                ),
                progress_callback=(
                    progress_callback
                )
            )
        )

        return generated_token_ids


    def decode_token_ids(
        self,
        token_ids
    ):

        if (
            token_ids.ndim
            == 2
        ):

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


    def reconstruct_generated_audio(
        self,
        generated_token_ids
    ):

        generated_patches = (
            self.decode_token_ids(
                generated_token_ids
            )
        )

        generated_audio = (
            reconstruct_audio_from_stft_tokens(
                generated_patches,
                sample_rate=(
                    TARGET_SAMPLE_RATE
                )
            )
        )

        return (
            generated_audio,
            generated_patches
        )


    def save_audio_outputs(
        self,
        context_audio,
        generated_audio,
        output_name
    ):

        context_path = (
            self.output_directory
            / (
                f"{output_name}_"
                f"context.wav"
            )
        )

        generated_path = (
            self.output_directory
            / (
                f"{output_name}_"
                f"generated.wav"
            )
        )

        combined_path = (
            self.output_directory
            / (
                f"{output_name}_"
                f"context_and_generated.wav"
            )
        )

        sf.write(
            context_path,
            context_audio,
            TARGET_SAMPLE_RATE,
            subtype="FLOAT"
        )

        sf.write(
            generated_path,
            generated_audio,
            TARGET_SAMPLE_RATE,
            subtype="FLOAT"
        )

        combined_audio = (
            np.concatenate(
                [
                    context_audio,
                    generated_audio
                ]
            )
        )

        sf.write(
            combined_path,
            combined_audio,
            TARGET_SAMPLE_RATE,
            subtype="FLOAT"
        )

        return (
            context_path,
            generated_path,
            combined_path
        )


    def run_inference(
        self,
        audio_file_path,
        start_time_seconds=0.0,
        output_name="drum_continuation"
    ):

        (
            context_audio,
            sample_rate
        ) = (
            self.load_context_audio(
                audio_file_path=(
                    audio_file_path
                ),
                start_time_seconds=(
                    start_time_seconds
                )
            )
        )

        context_patches = (
            self.convert_context_to_stft_patches(
                context_audio
            )
        )

        context_token_ids = (
            self.tokenize_context(
                context_patches
            )
        )

        generated_token_ids = (
            self.generate_target_tokens(
                context_token_ids
            )
        )

        (
            generated_audio,
            generated_patches
        ) = (
            self.reconstruct_generated_audio(
                generated_token_ids
            )
        )

        (
            context_path,
            generated_path,
            combined_path
        ) = (
            self.save_audio_outputs(
                context_audio=(
                    context_audio
                ),
                generated_audio=(
                    generated_audio
                ),
                output_name=(
                    output_name
                )
            )
        )

        return {
            "context_audio":
                context_audio,

            "generated_audio":
                generated_audio,

            "context_token_ids":
                context_token_ids
                .detach()
                .cpu(),

            "generated_token_ids":
                generated_token_ids
                .detach()
                .cpu(),

            "generated_patches":
                generated_patches,

            "context_path":
                context_path,

            "generated_path":
                generated_path,

            "combined_path":
                combined_path
        }


if __name__ == "__main__":

    AUDIO_FILE_PATH = (
        "C:/Uni/Advanced_AI/Tyer_KT_222078632_IT18X57_Prototype/src/audio_inputs_for_inference/Drums3.flac"
    )

    START_TIME_SECONDS = 0.0

    inference_pipeline = (
        DrumContinuationInferencePipeline(
            tokenizer_checkpoint_path=(
                TOKENIZER_CHECKPOINT_PATH
            ),
            transformer_checkpoint_path=(
                TRANSFORMER_CHECKPOINT_PATH
            ),
            temperature=(
                TEMPERATURE
            ),
            top_k=(
                TOP_K
            ),
            top_p=(
                TOP_P
            )
        )
    )

    inference_results = (
        inference_pipeline.run_inference(
            audio_file_path=(
                AUDIO_FILE_PATH
            ),
            start_time_seconds=(
                START_TIME_SECONDS
            ),
            output_name=(
                "inference_test_1"
            )
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DRUM CONTINUATION INFERENCE"
    )

    print(
        "=" * 70
    )

    print(
        "Device:",
        inference_pipeline.device
    )

    print(
        "Input audio:",
        AUDIO_FILE_PATH
    )

    print(
        "Start time:",
        START_TIME_SECONDS
    )

    print(
        "Temperature:",
        inference_pipeline.temperature
    )

    print(
        "Top-k:",
        inference_pipeline.top_k
    )

    print(
        "Top-p:",
        inference_pipeline.top_p
    )

    print(
        "\nContext token IDs"
    )

    print(
        "=" * 70
    )

    print(
        "Shape:",
        inference_results[
            "context_token_ids"
        ].shape
    )

    print(
        "Minimum:",
        inference_results[
            "context_token_ids"
        ].min().item()
    )

    print(
        "Maximum:",
        inference_results[
            "context_token_ids"
        ].max().item()
    )

    print(
        "\nGenerated token IDs"
    )

    print(
        "=" * 70
    )

    print(
        "Shape:",
        inference_results[
            "generated_token_ids"
        ].shape
    )

    print(
        "Minimum:",
        inference_results[
            "generated_token_ids"
        ].min().item()
    )

    print(
        "Maximum:",
        inference_results[
            "generated_token_ids"
        ].max().item()
    )

    print(
        "\nSaved audio"
    )

    print(
        "=" * 70
    )

    print(
        "Context:",
        inference_results[
            "context_path"
        ]
    )

    print(
        "Generated:",
        inference_results[
            "generated_path"
        ]
    )

    print(
        "Combined:",
        inference_results[
            "combined_path"
        ]
    )