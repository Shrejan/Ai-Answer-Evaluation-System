# ============================================================
# TrOCR BASE HANDWRITTEN — MODEL AUDIT
# ============================================================


import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# ------------------------------------------------------------
# 1. MODEL
# ------------------------------------------------------------

MODEL_NAME = "microsoft/trocr-base-handwritten"

print("=" * 80)
print("LOADING MODEL")
print("=" * 80)

processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

model.eval()

print(f"Model: {MODEL_NAME}")
print(f"Architecture: {model.__class__.__name__}")


# ------------------------------------------------------------
# 2. PARAMETER COUNTS
# ------------------------------------------------------------

def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def count_trainable_parameters(module):
    return sum(
        p.numel()
        for p in module.parameters()
        if p.requires_grad
    )


total_params = count_parameters(model)
trainable_params = count_trainable_parameters(model)

vision_params = count_parameters(model.encoder)
decoder_params = count_parameters(model.decoder)

print("\n" + "=" * 80)
print("PARAMETER COUNTS")
print("=" * 80)

print(f"Total parameters       : {total_params:,}")
print(f"Trainable parameters   : {trainable_params:,}")
print(f"Frozen parameters      : {total_params - trainable_params:,}")

print(f"\nVision Encoder params  : {vision_params:,}")
print(f"Text Decoder params    : {decoder_params:,}")

print(f"\nTotal parameters (M)   : {total_params / 1e6:.2f} M")
print(f"Vision Encoder (M)     : {vision_params / 1e6:.2f} M")
print(f"Text Decoder (M)       : {decoder_params / 1e6:.2f} M")


# ============================================================
# MODEL CONFIGURATION
# ============================================================

config = model.config

print("\n" + "=" * 80)
print("MODEL CONFIGURATION")
print("=" * 80)

print(f"Encoder type            : {config.encoder.model_type}")
print(f"Decoder type            : {config.decoder.model_type}")

print(f"Encoder layers          : {config.encoder.num_hidden_layers}")
print(f"Decoder layers          : {config.decoder.num_hidden_layers}")

print(f"Encoder hidden size     : {config.encoder.hidden_size}")
print(f"Decoder hidden size     : {config.decoder.hidden_size}")

print(f"Encoder attention heads : {config.encoder.num_attention_heads}")
print(f"Decoder attention heads : {config.decoder.num_attention_heads}")

print(
    f"Encoder intermediate    : "
    f"{getattr(config.encoder, 'intermediate_size', 'N/A')}"
)

print(
    f"Decoder intermediate    : "
    f"{getattr(config.decoder, 'intermediate_size', 'N/A')}"
)

# ------------------------------------------------------------
# 4. VOCABULARY
# ------------------------------------------------------------

tokenizer = processor.tokenizer

print("\n" + "=" * 80)
print("TOKENIZER / VOCABULARY")
print("=" * 80)

print(f"Tokenizer class        : {tokenizer.__class__.__name__}")
print(f"Vocabulary size        : {len(tokenizer):,}")
print(f"Config vocab size      : {config.decoder.vocab_size:,}")


# ------------------------------------------------------------
# 5. SPECIAL TOKEN IDs
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SPECIAL TOKEN IDs")
print("=" * 80)

special_tokens = {
    "BOS": tokenizer.bos_token,
    "EOS": tokenizer.eos_token,
    "PAD": tokenizer.pad_token,
    "UNK": tokenizer.unk_token,
}

special_token_ids = {
    "BOS": tokenizer.bos_token_id,
    "EOS": tokenizer.eos_token_id,
    "PAD": tokenizer.pad_token_id,
    "UNK": tokenizer.unk_token_id,
}

for name in special_tokens:
    print(
        f"{name:4} : "
        f"token = {special_tokens[name]!r:15} "
        f"ID = {special_token_ids[name]}"
    )


# ------------------------------------------------------------
# 6. GENERATION / DECODER IDs
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("DECODER / GENERATION IDs")
print("=" * 80)

print(f"decoder_start_token_id : {config.decoder_start_token_id}")
print(f"eos_token_id           : {config.eos_token_id}")
print(f"pad_token_id           : {config.pad_token_id}")
print(f"bos_token_id           : {config.bos_token_id}")

print(f"max_length             : {config.max_length}")
print(f"max_position_embeddings: "
      f"{getattr(config.decoder, 'max_position_embeddings', 'N/A')}")


# ------------------------------------------------------------
# 7. IMAGE PROCESSOR
# ------------------------------------------------------------

image_processor = processor.image_processor

print("\n" + "=" * 80)
print("IMAGE PROCESSOR")
print("=" * 80)

print(f"Processor class        : {image_processor.__class__.__name__}")

print(f"Do resize              : "
      f"{getattr(image_processor, 'do_resize', 'N/A')}")

print(f"Size                   : "
      f"{getattr(image_processor, 'size', 'N/A')}")

print(f"Do rescale             : "
      f"{getattr(image_processor, 'do_rescale', 'N/A')}")

print(f"Rescale factor         : "
      f"{getattr(image_processor, 'rescale_factor', 'N/A')}")

print(f"Do normalize           : "
      f"{getattr(image_processor, 'do_normalize', 'N/A')}")

print(f"Image mean             : "
      f"{getattr(image_processor, 'image_mean', 'N/A')}")

print(f"Image std              : "
      f"{getattr(image_processor, 'image_std', 'N/A')}")


# ------------------------------------------------------------
# 8. ATTENTION / PROJECTION MODULES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("ATTENTION / PROJECTION MODULES")
print("=" * 80)

projection_keywords = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "query",
    "key",
    "value",
]

projection_modules = []

for name, module in model.named_modules():

    if any(keyword in name.lower() for keyword in projection_keywords):

        if isinstance(module, torch.nn.Linear):

            projection_modules.append(
                (name, module)
            )

            print(
                f"{name:80} "
                f"{module.in_features:5} -> "
                f"{module.out_features:5}"
            )


# ------------------------------------------------------------
# 9. FFN MODULES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FFN / FEED-FORWARD MODULES")
print("=" * 80)

ffn_keywords = [
    "fc1",
    "fc2",
    "intermediate",
    "dense",
]

ffn_modules = []

for name, module in model.named_modules():

    if any(keyword in name.lower() for keyword in ffn_keywords):

        if isinstance(module, torch.nn.Linear):

            ffn_modules.append(
                (name, module)
            )

            print(
                f"{name:80} "
                f"{module.in_features:5} -> "
                f"{module.out_features:5}"
            )


# # ------------------------------------------------------------
# # 10. ALL LINEAR MODULES
# # ------------------------------------------------------------

# print("\n" + "=" * 80)
# print("ALL LINEAR MODULES")
# print("=" * 80)

# linear_count = 0

# for name, module in model.named_modules():

#     if isinstance(module, torch.nn.Linear):

#         linear_count += 1

#         print(
#             f"{linear_count:3}. "
#             f"{name:75} "
#             f"{module.in_features:5} -> "
#             f"{module.out_features:5}"
#         )


# # ------------------------------------------------------------
# # 11. COMPLETE MODULE NAMES
# # ------------------------------------------------------------

# print("\n" + "=" * 80)
# print("COMPLETE MODEL MODULE TREE")
# print("=" * 80)

# for name, module in model.named_modules():
#     print(f"{name:100} {type(module).__name__}")


# ------------------------------------------------------------
# 12. PARAMETER COUNT BY MAJOR COMPONENT
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("PARAMETER BREAKDOWN")
print("=" * 80)

components = {
    "Vision Encoder": model.encoder,
    "Text Decoder": model.decoder,
}

for component_name, component in components.items():

    params = count_parameters(component)
    trainable = count_trainable_parameters(component)

    print(f"\n{component_name}")
    print(f"  Total      : {params:,}")
    print(f"  Trainable  : {trainable:,}")
    print(f"  Frozen     : {params - trainable:,}")


# ------------------------------------------------------------
# 13. TRAINABLE PARAMETERS BY MODULE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TRAINABLE PARAMETERS")
print("=" * 80)

for name, module in model.named_modules():

    trainable = count_trainable_parameters(module)

    if trainable > 0:
        print(
            f"{name:80} "
            f"{trainable:,}"
        )


# ------------------------------------------------------------
# 14. GENERATION CONFIG
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("GENERATION CONFIG")
print("=" * 80)

generation_config = model.generation_config

generation_attributes = [
    "max_length",
    "min_length",
    "num_beams",
    "early_stopping",
    "length_penalty",
    "no_repeat_ngram_size",
    "decoder_start_token_id",
    "eos_token_id",
    "pad_token_id",
    "bos_token_id",
]

for attr in generation_attributes:

    value = getattr(
        generation_config,
        attr,
        "N/A"
    )

    print(f"{attr:25}: {value}")


# ------------------------------------------------------------
# 15. TOKENIZER SPECIAL TOKENS MAP
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SPECIAL TOKENS MAP")
print("=" * 80)

print(tokenizer.special_tokens_map)


# ------------------------------------------------------------
# 16. TOKENIZER SANITY TEST
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TOKENIZER SANITY TEST")
print("=" * 80)

test_text = "The aquifer contains groundwater."

encoded = tokenizer(
    test_text,
    return_tensors="pt"
)

decoded = tokenizer.decode(
    encoded["input_ids"][0],
    skip_special_tokens=False
)

print(f"Original : {test_text}")
print(f"IDs      : {encoded['input_ids'][0].tolist()}")
print(f"Decoded  : {decoded}")


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print(f"""
Model                    : {MODEL_NAME}
Architecture             : {model.__class__.__name__}

------------------------------------------------------------
PARAMETERS
------------------------------------------------------------

Total parameters         : {total_params:,}
Trainable parameters     : {trainable_params:,}
Frozen parameters        : {total_params - trainable_params:,}

Vision Encoder params    : {vision_params:,}
Text Decoder params      : {decoder_params:,}

Total parameters (M)     : {total_params / 1e6:.2f} M
Vision Encoder (M)       : {vision_params / 1e6:.2f} M
Text Decoder (M)         : {decoder_params / 1e6:.2f} M

------------------------------------------------------------
ENCODER
------------------------------------------------------------

Encoder type             : {config.encoder.model_type}
Encoder layers           : {config.encoder.num_hidden_layers}
Encoder hidden size      : {config.encoder.hidden_size}
Encoder attention heads  : {config.encoder.num_attention_heads}
Encoder intermediate     : {config.encoder.intermediate_size}

------------------------------------------------------------
DECODER
------------------------------------------------------------

Decoder type             : {config.decoder.model_type}
Decoder layers           : {config.decoder.num_hidden_layers}
Decoder hidden size      : {config.decoder.hidden_size}
Decoder attention heads  : {config.decoder.num_attention_heads}



------------------------------------------------------------
TOKENIZER
------------------------------------------------------------

Tokenizer class          : {tokenizer.__class__.__name__}
Vocabulary size          : {len(tokenizer):,}

------------------------------------------------------------
SPECIAL TOKENS
------------------------------------------------------------

BOS token                : {tokenizer.bos_token!r}
BOS ID                   : {tokenizer.bos_token_id}

EOS token                : {tokenizer.eos_token!r}
EOS ID                   : {tokenizer.eos_token_id}

PAD token                : {tokenizer.pad_token!r}
PAD ID                   : {tokenizer.pad_token_id}

UNK token                : {tokenizer.unk_token!r}
UNK ID                   : {tokenizer.unk_token_id}

------------------------------------------------------------
DECODER / GENERATION
------------------------------------------------------------

Decoder start token ID   : {config.decoder_start_token_id}
EOS token ID             : {config.eos_token_id}
PAD token ID             : {config.pad_token_id}
BOS token ID             : {config.bos_token_id}

Maximum length           : {getattr(config, "max_length", "N/A")}

------------------------------------------------------------
IMAGE PROCESSOR
------------------------------------------------------------

Processor                : {image_processor.__class__.__name__}

Resize                   : {getattr(image_processor, "do_resize", "N/A")}
Image size               : {getattr(image_processor, "size", "N/A")}

Rescale                  : {getattr(image_processor, "do_rescale", "N/A")}
Rescale factor            : {getattr(image_processor, "rescale_factor", "N/A")}

Normalize                : {getattr(image_processor, "do_normalize", "N/A")}
Mean                     : {getattr(image_processor, "image_mean", "N/A")}
Std                      : {getattr(image_processor, "image_std", "N/A")}

============================================================
MODEL AUDIT COMPLETE
============================================================
""")