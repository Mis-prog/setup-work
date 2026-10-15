from __future__ import annotations

from setup_work.engines import LaunchConfig, LlamaCppEngine, VllmEngine, get_engine, is_loopback


def test_vllm_command():
    lc = LaunchConfig(
        name="d",
        engine="vllm",
        model_ref="qwen3-32b:awq",
        model_path="/cache/snap",
        served_model_name="qwen3-32b",
        port=8001,
        context=32768,
        tensor_parallel=2,
        gpu_memory_utilization=0.85,
        dtype="half",
        concurrency=8,
    )
    argv = VllmEngine().build_command(lc, "/opt/vllm/bin/vllm")
    assert argv[:3] == ["/opt/vllm/bin/vllm", "serve", "/cache/snap"]
    joined = " ".join(argv)
    for frag in (
        "--host 127.0.0.1",
        "--port 8001",
        "--served-model-name qwen3-32b",
        "--max-model-len 32768",
        "--tensor-parallel-size 2",
        "--gpu-memory-utilization 0.85",
        "--dtype half",
        "--max-num-seqs 8",
    ):
        assert frag in joined
    assert "--api-key" not in joined


def test_vllm_api_key_goes_to_env_not_argv(tmp_path):
    kf = tmp_path / "k"
    kf.write_text("s3cret\n")
    lc = LaunchConfig(
        name="d",
        engine="vllm",
        model_ref="x",
        model_path="m",
        served_model_name="m",
        api_key_file=str(kf),
    )
    eng = VllmEngine()
    assert "s3cret" not in " ".join(eng.build_command(lc, "vllm"))
    assert eng.build_env(lc) == {"VLLM_API_KEY": "s3cret"}


def test_llama_cpp_command_full_offload_and_split():
    lc = LaunchConfig(
        name="d",
        engine="llama.cpp",
        model_ref="m:q4",
        model_path="/m/x-00001-of-00003.gguf",
        served_model_name="m",
        port=8080,
        context=8192,
        concurrency=2,
        gpu_layers=-1,
        tensor_split=[0.7, 0.3],
        threads=8,
    )
    argv = LlamaCppEngine().build_command(lc, "llama-server")
    j = " ".join(argv)
    assert "-m /m/x-00001-of-00003.gguf" in j
    assert "-c 16384" in j and "-np 2" in j  # total cache = context x slots
    assert "-ngl all" in j
    assert "-sm layer -ts 0.7,0.3" in j
    assert "-t 8" in j


def test_llama_cpp_api_key_file_and_embedding(tmp_path):
    lc = LaunchConfig(
        name="d",
        engine="llama.cpp",
        model_ref="e",
        model_path="e.gguf",
        served_model_name="e",
        api_key_file="/state/d.key",
        embedding=True,
        gpu_layers=0,
    )
    argv = LlamaCppEngine().build_command(lc, "llama-server")
    assert "--api-key-file" in argv and "--api-key" not in argv
    assert "--embedding" in argv
    assert argv[argv.index("-ngl") + 1] == "0"


def test_engine_aliases_and_loopback():
    assert get_engine("llamacpp").name == "llama.cpp"
    assert get_engine("VLLM").name == "vllm"
    assert is_loopback("127.0.0.1") and is_loopback("::1") and is_loopback("localhost")
    assert not is_loopback("0.0.0.0") and not is_loopback("10.0.0.5")


def test_base_url_for_wildcard_bind():
    lc = LaunchConfig(
        name="d",
        engine="vllm",
        model_ref="x",
        model_path="x",
        served_model_name="x",
        host="0.0.0.0",
        port=9000,
    )
    assert lc.base_url == "http://127.0.0.1:9000"
