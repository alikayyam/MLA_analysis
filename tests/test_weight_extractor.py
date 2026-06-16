import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proxy.toy_mla import make_toy_model
from model_io.weight_extractor import extract_mla_weights_from_toy, toy_config_from_model


def make_cfg_and_model(**kwargs):
    model = make_toy_model(**kwargs)
    config = toy_config_from_model(model)
    return model, config


def test_shapes_default():
    model, config = make_cfg_and_model()
    layer = model.layers[0]
    weights = extract_mla_weights_from_toy(layer, config)
    r = config["kv_lora_rank"]
    d = config["hidden_size"]
    nh = config["num_attention_heads"]
    qk_nope = config["qk_nope_head_dim"]
    v = config["v_head_dim"]
    assert weights["W_c"].shape == (d, r), weights["W_c"].shape
    assert weights["W_K_up"].shape == (nh * qk_nope, r), weights["W_K_up"].shape
    assert weights["W_V_up"].shape == (nh * v, r), weights["W_V_up"].shape


def test_dtype_cast():
    model, config = make_cfg_and_model()
    layer = model.layers[0]
    weights = extract_mla_weights_from_toy(layer, config, dtype=torch.float64)
    for k, v in weights.items():
        assert v.dtype == torch.float64, f"{k} has wrong dtype {v.dtype}"


def test_effective_operator_shape():
    model, config = make_cfg_and_model()
    layer = model.layers[0]
    w = extract_mla_weights_from_toy(layer, config)
    W_tilde_K = w["W_c"] @ w["W_K_up"].T
    W_tilde_V = w["W_c"] @ w["W_V_up"].T
    d = config["hidden_size"]
    nh = config["num_attention_heads"]
    qk_nope = config["qk_nope_head_dim"]
    v = config["v_head_dim"]
    assert W_tilde_K.shape == (d, nh * qk_nope)
    assert W_tilde_V.shape == (d, nh * v)


def test_all_layers():
    model, config = make_cfg_and_model(num_layers=4)
    for i, layer in enumerate(model.layers):
        w = extract_mla_weights_from_toy(layer, config)
        assert "W_c" in w and "W_K_up" in w and "W_V_up" in w
