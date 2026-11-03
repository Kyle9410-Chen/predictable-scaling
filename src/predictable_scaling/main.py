from transformers import LlamaConfig, AutoModelForCausalLM

def build_model(attn_impl="kernels-community/flash-attn3"):
    config = LlamaConfig(
        vocab_size=128256,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=6,
        num_attention_heads=12,
        num_key_value_heads=3,
        head_dim=64,
        hidden_act="sliu",
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192
        },
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
        initializer_range=0.02,
        bos_token_id=128000,
        eos_token_id=128001,
        use_cache=False,
    )

    return AutoModelForCausalLM.from_config(config, attn_implementation=attn_impl)

def main():
    pass

if __name__ == "__main__":
    main()