# Copyright 2020 Hirofumi Inaguma
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer common arguments."""


from distutils.util import strtobool
import logging


def add_arguments_conformer_common(group):
    """Add Transformer common arguments."""
    group.add_argument(
        "--transformer-encoder-pos-enc-layer-type",
        type=str,
        default="abs_pos",
        choices=["abs_pos", "scaled_abs_pos", "rel_pos"],
        help="Transformer encoder positional encoding layer type",
    )
    group.add_argument(
        "--transformer-encoder-activation-type",
        type=str,
        default="swish",
        choices=["relu", "hardtanh", "selu", "swish"],
        help="Transformer encoder activation function type",
    )
    group.add_argument(
        "--macaron-style",
        default=False,
        type=strtobool,
        help="Whether to use macaron style for positionwise layer",
    )
    # Attention
    group.add_argument(
        "--zero-triu",
        default=False,
        type=strtobool,
        help="If true, zero the uppper triangular part of attention matrix.",
    )
    # Relative positional encoding
    group.add_argument(
        "--rel-pos-type",
        type=str,
        default="legacy",
        choices=["legacy", "latest"],
        help="Whether to use the latest relative positional encoding or the legacy one."
        "The legacy relative positional encoding will be deprecated in the future."
        "More Details can be found in https://github.com/espnet/espnet/pull/2816.",
    )
    # CNN module
    group.add_argument(
        "--use-cnn-module",
        default=False,
        type=strtobool,
        help="Use convolution module or not",
    )
    group.add_argument(
        "--cnn-module-kernel",
        default=31,
        type=int,
        help="Kernel size of convolution module.",
    )
    return group


def verify_rel_pos_type(args):
    """Verify the relative positional encoding type for compatibility.

    Args:
        args (Namespace): original arguments
    Returns:
        args (Namespace): modified arguments
    """
    rel_pos_type = getattr(args, "rel_pos_type", None)
    if rel_pos_type is None or rel_pos_type == "legacy":
        if args.transformer_encoder_pos_enc_layer_type == "rel_pos":
            args.transformer_encoder_pos_enc_layer_type = "legacy_rel_pos"
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        if args.transformer_encoder_selfattn_layer_type == "rel_selfattn":
            args.transformer_encoder_selfattn_layer_type = "legacy_rel_selfattn"
            logging.warning(
                "Using legacy_rel_selfattn and it will be deprecated in the future."
            )

    return args
