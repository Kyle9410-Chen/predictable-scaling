from transformers import AutoTokenizer, LlamaConfig, AutoModelForCausalLM
import torch.distributed as dist

"""
Requested budget	Layers (L)  Hidden (d)	MLP (m)	Query heads	KV heads	Actual parameters
150M	6	768	    3,072	12	3	149,825,280
300M	9	1,024	5,184	16	4	298,273,792
500M	11	1,536	4,672	24	6	498,731,520
750M	13	1,792	5,888	28	7	745,749,760
1B	    15	1,792	8,000	28	7	995,432,704
"""

class ModelScaleConfig:
    VOCAB_SIZE = 128256
    HEAD_DIM = 64
    
    def __init__(self, layers=6, hidden_size=768, mlp=3072, heads=12, kv_heads=3):
        self.layers = layers
        self.hidden_size = hidden_size
        self.mlp = mlp
        self.heads = heads
        self.kv_heads = kv_heads

    def total_params(self):
        embedding = self.VOCAB_SIZE * self.hidden_size # tie_word_embedding share embedding and lm head

        rms_norm = self.hidden_size

        q_proj = self.hidden_size * self.HEAD_DIM * self.heads
        k_proj = self.hidden_size * self.HEAD_DIM * self.kv_heads
        v_proj = self.hidden_size * self.HEAD_DIM * self.kv_heads
        o_proj = self.hidden_size * self.HEAD_DIM * self.heads
        attn = q_proj + k_proj + v_proj + o_proj

        mlp = 3 * self.hidden_size * self.mlp # gate_proj, up_proj, down_proj (no bias)

        block = rms_norm + attn + mlp + rms_norm

        return embedding + self.layers * block + rms_norm

predefined_scales = {
    "150M": ModelScaleConfig(6, 768, 3072, 12, 3),
    "300M": ModelScaleConfig(9, 1024, 5184, 16, 4),
    "500M": ModelScaleConfig(11, 1536, 4672, 24, 6),
    "750M": ModelScaleConfig(13, 1792, 5888, 28, 7),
    "1B": ModelScaleConfig(15, 1792, 8000, 28, 7)
}

def build_model(attn_impl="kernels-community/flash-attn3", scale="150M", config=None) -> AutoModelForCausalLM:
    cfg = None

    # custom config override default
    if config is not None:
        cfg = config
    elif scale is not None and scale in predefined_scales:
        cfg = predefined_scales[scale]
    else:
        raise ValueError("Invalid scale: " + str(scale))

    config = LlamaConfig(
        vocab_size=128256,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.mlp,
        num_hidden_layers=cfg.layers,
        num_attention_heads=cfg.heads,
        num_key_value_heads=cfg.kv_heads,
        head_dim=cfg.HEAD_DIM,
        hidden_act="silu",
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        # rope_scaling={
        #     "rope_type": "llama3",
        #     "factor": 32.0,
        #     "low_freq_factor": 1.0,
        #     "high_freq_factor": 4.0,
        #     "original_max_position_embeddings": 8192
        # },
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
        initializer_range=0.02,
        bos_token_id=128000,
        eos_token_id=128001,
        use_cache=False,
    )

    return AutoModelForCausalLM.from_config(config, attn_implementation=attn_impl)

def save_model(model, path):
    model.module.save_pretrained(path)
    AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B").save_pretrained(path)

if __name__ == "__main__":
    print(ModelScaleConfig().total_params())