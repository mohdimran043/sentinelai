# Phase 1B spike result — AutoAWQ load (Task 1)

Ran: 2026-08-01
Machine: RTX 4060 Laptop, 8188 MiB, driver 580.173.02, compute capability 8.9,
20 CPU cores, 15 GB RAM (~7 GB free). Desktop session running throughout (nvidia-smi
readings include the compositor's baseline usage, not an idle/headless GPU).

## Decision

transformers + bitsandbytes 4-bit NF4 fallback. Task 13 loads the unquantised Qwen/Qwen2.5-VL-3B-Instruct checkpoint with BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16). pyproject.toml's gpu extra now carries bitsandbytes instead of autoawq. autoawq itself DID install and its model DID load into VRAM, but inference failed inside autoawq's own bundled Triton GEMM kernel (a genuine bug against the installed Triton 3.7.1 / torch 2.13 combination), after three separate cheap-fix attempts (transformers==4.49.0, transformers==4.51.3 — the maintainer's own last-tested version — and an explicit torch_dtype=torch.float16). This is exactly the maintainer-deprecation fragility this spike exists to retire.

## Measured VRAM (desktop session running)

AutoAWQ path (Branch A attempt, transformers==4.51.3 + explicit torch_dtype=torch.float16,
the furthest-progressing configuration):
- baseline (desktop idle): 694 MiB / 8188 MiB
- after model load: 4279 MiB / 8188 MiB
- after inference: N/A — generate() crashed inside autoawq's Triton kernel before completing

bitsandbytes NF4 fallback (Branch B, verified working):
- baseline (desktop idle): 694 MiB / 8188 MiB
- after model load: 3359 MiB / 8188 MiB
- after inference (peak): 3517 MiB / 8188 MiB
- torch.cuda.max_memory_allocated(): 2564 MiB
- inference wall time: 2.66 s for 64 new tokens

## Resolved versions

- torch: 2.13.0+cu130
- transformers: 5.14.1 (bitsandbytes-verified run); also exercised 4.49.0 and 4.51.3 while
  diagnosing the autoawq failure
- autoawq: 0.2.9 (installed and importable; fails at inference, see below)
- bitsandbytes: 0.50.0
- accelerate: 1.14.0
- qwen-vl-utils: 0.0.14

## Why Branch A was rejected

autoawq did install successfully (`pip install -e '.[dev,runtime,gpu]' pillow` completed
with `PIP_EXIT=0`), and this is not simply a stale-pin issue — it was retried through three
genuinely distinct cheap fixes, each of which surfaced a new and different failure:

1. **As-installed (transformers>=4.49 floating pin resolved to 5.14.1).** transformers has
   fully removed autoawq as a supported AWQ backend: `AwqQuantizer.validate_environment()`
   unconditionally requires `gptqmodel` now, regardless of whether autoawq is installed.
   `ImportError: Loading an AWQ quantized model requires gptqmodel.`
2. **Pinned transformers==4.49.0** (the pyproject.toml floor). Past the gptqmodel gate, but
   autoawq 0.2.9's own model registry unconditionally imports Qwen3 modeling code that
   does not exist in transformers 4.49.0: `ModuleNotFoundError: No module named
   'transformers.models.qwen3'`.
3. **Pinned transformers==4.51.3** — the exact version autoawq's own deprecation banner
   names as "the last tested configuration" (paired with torch 2.6.0; this machine has
   torch 2.13.0). The model now loads (4249 MiB). `model.generate()` fails: a Triton
   compilation error from a dtype mismatch between AWQ's fp16 scales and the
   auto-resolved bf16 activations.
4. **Same transformers==4.51.3, explicit `torch_dtype=torch.float16`** (as suggested by
   transformers' own runtime warning). Model loads (4279 MiB). Inference now gets past
   every decoder layer and fails only at the final lm_head projection, inside
   **autoawq's own bundled Triton GEMM kernel**, with a nonsensical type error between two
   float16 operands — a bug in that kernel against the installed Triton 3.7.1, not
   anything controllable from model config. autoawq has no compiled CUDA-extension wheel
   (`awq_ext`) for this torch/CUDA build, so it falls back to this broken Triton path.

Conclusion: this is a hard incompatibility, not an incidental one. autoawq was deprecated
by its maintainer in 2025 and is demonstrably unmaintained against current
torch/transformers/triton releases, exactly as the task brief anticipated. Further
debugging (e.g. patching autoawq's Triton kernel, or pinning torch/triton down to the
maintainer's exact 2.6.0/4.51.3 combination, which would fight the project's already-pinned
torch>=2.5 baseline) is out of scope for a spike whose job is to answer, not to fix,
this question.

## Branch B verification (bitsandbytes NF4, not just assumed)

Loaded the unquantised `Qwen/Qwen2.5-VL-3B-Instruct` checkpoint (not the AWQ checkpoint)
via plain `transformers` with `BitsAndBytesConfig(load_in_4bit=True,
bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)`, ran the same synthetic-image
probe end to end. Completed cleanly: `INFERENCE_OK elapsed_seconds=2.66`, produced a real
caption ("The image is a solid blue color with no other objects or details present."),
peak VRAM 3517 MiB — comfortably inside the 8192 MiB ceiling (2048 MiB reserved) and close
to the ~3.5 GB the spec estimated for this path.

## pyproject.toml changes (Branch B)

```diff
 gpu = [
     "torch>=2.5",
     "ultralytics>=8.3",
     "supervision>=0.24",
     "transformers>=4.49",
     "accelerate>=1.0",
-    "autoawq>=0.2.6",
+    "bitsandbytes>=0.43",
     "qwen-vl-utils>=0.0.8",
 ]
@@ [[tool.mypy.overrides]] module list @@
     "ultralytics.*",
     "supervision.*",
     "av.*",
-    "autoawq.*",
+    "bitsandbytes.*",
     "qwen_vl_utils.*",
     "minio.*",
     "jsonschema.*",
```

## Raw probe log

```
Obtaining file:///home/imran/Documents/github/SentinelAI/ai-engine
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Checking if build backend supports build_editable: started
  Checking if build backend supports build_editable: finished with status 'done'
  Getting requirements to build editable: started
  Getting requirements to build editable: finished with status 'done'
  Preparing editable metadata (pyproject.toml): started
  Preparing editable metadata (pyproject.toml): finished with status 'done'
Requirement already satisfied: pillow in ./.venv/lib/python3.12/site-packages (12.3.0)
Requirement already satisfied: pydantic>=2.9 in ./.venv/lib/python3.12/site-packages (from sentinel-ai==0.1.0) (2.13.4)
Requirement already satisfied: pydantic-settings>=2.5 in ./.venv/lib/python3.12/site-packages (from sentinel-ai==0.1.0) (2.14.2)
Requirement already satisfied: jsonschema>=4.23 in ./.venv/lib/python3.12/site-packages (from sentinel-ai==0.1.0) (4.26.0)
Requirement already satisfied: fastapi>=0.115 in ./.venv/lib/python3.12/site-packages (from sentinel-ai==0.1.0) (0.141.1)
Requirement already satisfied: uvicorn>=0.31 in ./.venv/lib/python3.12/site-packages (from uvicorn[standard]>=0.31; extra == "runtime"->sentinel-ai==0.1.0) (0.52.0)
Requirement already satisfied: av>=13.0 in ./.venv/lib/python3.12/site-packages (from sentinel-ai==0.1.0) (18.0.0)
Requirement already satisfied: numpy<2.2,>=1.26 in ./.venv/lib/python3.12/site-packages (from sentinel-ai==0.1.0) (2.1.3)
Requirement already satisfied: aio-pika>=9.4 in ./.venv/lib/python3.12/site-packages (from sentinel-ai==0.1.0) (10.0.1)
Requirement already satisfied: minio>=7.2 in ./.venv/lib/python3.12/site-packages (from sentinel-ai==0.1.0) (7.2.20)
Requirement already satisfied: torch>=2.5 in ./.venv/lib/python3.12/site-packages (from sentinel-ai==0.1.0) (2.13.0)
Requirement already satisfied: ultralytics>=8.3 in ./.venv/lib/python3.12/site-packages (from sentinel-ai==0.1.0) (8.4.114)
Requirement already satisfied: supervision>=0.24 in ./.venv/lib/python3.12/site-packages (from sentinel-ai==0.1.0) (0.29.1)
Requirement already satisfied: transformers>=4.49 in ./.venv/lib/python3.12/site-packages (from sentinel-ai==0.1.0) (5.14.1)
Requirement already satisfied: accelerate>=1.0 in ./.venv/lib/python3.12/site-packages (from sentinel-ai==0.1.0) (1.14.0)
Requirement already satisfied: autoawq>=0.2.6 in ./.venv/lib/python3.12/site-packages (from sentinel-ai==0.1.0) (0.2.9)
Requirement already satisfied: qwen-vl-utils>=0.0.8 in ./.venv/lib/python3.12/site-packages (from sentinel-ai==0.1.0) (0.0.14)
Requirement already satisfied: pytest>=8.3 in ./.venv/lib/python3.12/site-packages (from sentinel-ai==0.1.0) (9.1.1)
Requirement already satisfied: pytest-asyncio>=0.24 in ./.venv/lib/python3.12/site-packages (from sentinel-ai==0.1.0) (1.4.0)
Requirement already satisfied: pytest-cov>=5.0 in ./.venv/lib/python3.12/site-packages (from sentinel-ai==0.1.0) (7.1.0)
Requirement already satisfied: ruff>=0.7 in ./.venv/lib/python3.12/site-packages (from sentinel-ai==0.1.0) (0.16.1)
Requirement already satisfied: mypy>=1.13 in ./.venv/lib/python3.12/site-packages (from sentinel-ai==0.1.0) (2.3.0)
Requirement already satisfied: packaging>=20.0 in ./.venv/lib/python3.12/site-packages (from accelerate>=1.0->sentinel-ai==0.1.0) (26.2)
Requirement already satisfied: psutil in ./.venv/lib/python3.12/site-packages (from accelerate>=1.0->sentinel-ai==0.1.0) (7.2.2)
Requirement already satisfied: pyyaml in ./.venv/lib/python3.12/site-packages (from accelerate>=1.0->sentinel-ai==0.1.0) (6.0.3)
Requirement already satisfied: huggingface_hub>=0.21.0 in ./.venv/lib/python3.12/site-packages (from accelerate>=1.0->sentinel-ai==0.1.0) (1.26.0)
Requirement already satisfied: safetensors>=0.4.3 in ./.venv/lib/python3.12/site-packages (from accelerate>=1.0->sentinel-ai==0.1.0) (0.8.0)
Requirement already satisfied: aiormq<8,>=7 in ./.venv/lib/python3.12/site-packages (from aio-pika>=9.4->sentinel-ai==0.1.0) (7.0.0)
Requirement already satisfied: yarl in ./.venv/lib/python3.12/site-packages (from aio-pika>=9.4->sentinel-ai==0.1.0) (1.24.5)
Requirement already satisfied: pamqp<5,>=4 in ./.venv/lib/python3.12/site-packages (from aiormq<8,>=7->aio-pika>=9.4->sentinel-ai==0.1.0) (4.0.1)
Requirement already satisfied: triton in ./.venv/lib/python3.12/site-packages (from autoawq>=0.2.6->sentinel-ai==0.1.0) (3.7.1)
Requirement already satisfied: tokenizers>=0.12.1 in ./.venv/lib/python3.12/site-packages (from autoawq>=0.2.6->sentinel-ai==0.1.0) (0.22.2)
Requirement already satisfied: typing_extensions>=4.8.0 in ./.venv/lib/python3.12/site-packages (from autoawq>=0.2.6->sentinel-ai==0.1.0) (4.16.0)
Requirement already satisfied: datasets>=2.20 in ./.venv/lib/python3.12/site-packages (from autoawq>=0.2.6->sentinel-ai==0.1.0) (5.0.1)
Requirement already satisfied: zstandard in ./.venv/lib/python3.12/site-packages (from autoawq>=0.2.6->sentinel-ai==0.1.0) (0.25.0)
Requirement already satisfied: filelock in ./.venv/lib/python3.12/site-packages (from datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (3.32.2)
Requirement already satisfied: pyarrow>=21.0.0 in ./.venv/lib/python3.12/site-packages (from datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (25.0.0)
Requirement already satisfied: dill<0.4.2,>=0.3.0 in ./.venv/lib/python3.12/site-packages (from datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (0.4.1)
Requirement already satisfied: pandas in ./.venv/lib/python3.12/site-packages (from datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (3.0.5)
Requirement already satisfied: requests>=2.32.2 in ./.venv/lib/python3.12/site-packages (from datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (2.34.2)
Requirement already satisfied: httpx<1.0.0 in ./.venv/lib/python3.12/site-packages (from datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (0.28.1)
Requirement already satisfied: tqdm>=4.66.3 in ./.venv/lib/python3.12/site-packages (from datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (4.70.0)
Requirement already satisfied: xxhash in ./.venv/lib/python3.12/site-packages (from datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (3.8.1)
Requirement already satisfied: multiprocess<0.70.20 in ./.venv/lib/python3.12/site-packages (from datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (0.70.19)
Requirement already satisfied: fsspec<=2026.6.0,>=2023.1.0 in ./.venv/lib/python3.12/site-packages (from fsspec[http]<=2026.6.0,>=2023.1.0->datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (2026.6.0)
Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in ./.venv/lib/python3.12/site-packages (from fsspec[http]<=2026.6.0,>=2023.1.0->datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (3.14.3)
Requirement already satisfied: anyio in ./.venv/lib/python3.12/site-packages (from httpx<1.0.0->datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (4.14.2)
Requirement already satisfied: certifi in ./.venv/lib/python3.12/site-packages (from httpx<1.0.0->datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (2026.7.22)
Requirement already satisfied: httpcore==1.* in ./.venv/lib/python3.12/site-packages (from httpx<1.0.0->datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (1.0.9)
Requirement already satisfied: idna in ./.venv/lib/python3.12/site-packages (from httpx<1.0.0->datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (3.18)
Requirement already satisfied: h11>=0.16 in ./.venv/lib/python3.12/site-packages (from httpcore==1.*->httpx<1.0.0->datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (0.16.0)
Requirement already satisfied: click<9.0.0,>=8.4.2 in ./.venv/lib/python3.12/site-packages (from huggingface_hub>=0.21.0->accelerate>=1.0->sentinel-ai==0.1.0) (8.4.2)
Requirement already satisfied: hf-xet<2.0.0,>=1.5.1 in ./.venv/lib/python3.12/site-packages (from huggingface_hub>=0.21.0->accelerate>=1.0->sentinel-ai==0.1.0) (1.5.2)
Requirement already satisfied: aiohappyeyeballs>=2.5.0 in ./.venv/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.6.0,>=2023.1.0->datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (2.7.1)
Requirement already satisfied: aiosignal>=1.4.0 in ./.venv/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.6.0,>=2023.1.0->datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (1.4.0)
Requirement already satisfied: attrs>=17.3.0 in ./.venv/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.6.0,>=2023.1.0->datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (26.1.0)
Requirement already satisfied: frozenlist>=1.1.1 in ./.venv/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.6.0,>=2023.1.0->datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (1.8.0)
Requirement already satisfied: multidict<7.0,>=4.5 in ./.venv/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.6.0,>=2023.1.0->datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (6.7.1)
Requirement already satisfied: propcache>=0.2.0 in ./.venv/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.6.0,>=2023.1.0->datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (0.5.2)
Requirement already satisfied: starlette>=0.46.0 in ./.venv/lib/python3.12/site-packages (from fastapi>=0.115->sentinel-ai==0.1.0) (1.3.1)
Requirement already satisfied: typing-inspection>=0.4.2 in ./.venv/lib/python3.12/site-packages (from fastapi>=0.115->sentinel-ai==0.1.0) (0.4.2)
Requirement already satisfied: annotated-doc>=0.0.2 in ./.venv/lib/python3.12/site-packages (from fastapi>=0.115->sentinel-ai==0.1.0) (0.0.5)
Requirement already satisfied: jsonschema-specifications>=2023.03.6 in ./.venv/lib/python3.12/site-packages (from jsonschema>=4.23->sentinel-ai==0.1.0) (2025.9.1)
Requirement already satisfied: referencing>=0.28.4 in ./.venv/lib/python3.12/site-packages (from jsonschema>=4.23->sentinel-ai==0.1.0) (0.37.0)
Requirement already satisfied: rpds-py>=0.25.0 in ./.venv/lib/python3.12/site-packages (from jsonschema>=4.23->sentinel-ai==0.1.0) (2026.6.3)
Requirement already satisfied: argon2-cffi in ./.venv/lib/python3.12/site-packages (from minio>=7.2->sentinel-ai==0.1.0) (25.1.0)
Requirement already satisfied: pycryptodome in ./.venv/lib/python3.12/site-packages (from minio>=7.2->sentinel-ai==0.1.0) (3.23.0)
Requirement already satisfied: urllib3 in ./.venv/lib/python3.12/site-packages (from minio>=7.2->sentinel-ai==0.1.0) (2.7.0)
Requirement already satisfied: mypy_extensions>=1.0.0 in ./.venv/lib/python3.12/site-packages (from mypy>=1.13->sentinel-ai==0.1.0) (1.1.0)
Requirement already satisfied: pathspec>=1.0.0 in ./.venv/lib/python3.12/site-packages (from mypy>=1.13->sentinel-ai==0.1.0) (1.1.1)
Requirement already satisfied: librt>=0.13.0 in ./.venv/lib/python3.12/site-packages (from mypy>=1.13->sentinel-ai==0.1.0) (0.13.0)
Requirement already satisfied: ast-serialize<1.0.0,>=0.6.0 in ./.venv/lib/python3.12/site-packages (from mypy>=1.13->sentinel-ai==0.1.0) (0.6.0)
Requirement already satisfied: annotated-types>=0.6.0 in ./.venv/lib/python3.12/site-packages (from pydantic>=2.9->sentinel-ai==0.1.0) (0.8.0)
Requirement already satisfied: pydantic-core==2.46.4 in ./.venv/lib/python3.12/site-packages (from pydantic>=2.9->sentinel-ai==0.1.0) (2.46.4)
Requirement already satisfied: python-dotenv>=0.21.0 in ./.venv/lib/python3.12/site-packages (from pydantic-settings>=2.5->sentinel-ai==0.1.0) (1.2.2)
Requirement already satisfied: iniconfig>=1.0.1 in ./.venv/lib/python3.12/site-packages (from pytest>=8.3->sentinel-ai==0.1.0) (2.3.0)
Requirement already satisfied: pluggy<2,>=1.5 in ./.venv/lib/python3.12/site-packages (from pytest>=8.3->sentinel-ai==0.1.0) (1.6.0)
Requirement already satisfied: pygments>=2.7.2 in ./.venv/lib/python3.12/site-packages (from pytest>=8.3->sentinel-ai==0.1.0) (2.20.0)
Requirement already satisfied: coverage>=7.10.6 in ./.venv/lib/python3.12/site-packages (from coverage[toml]>=7.10.6->pytest-cov>=5.0->sentinel-ai==0.1.0) (7.15.2)
Requirement already satisfied: charset_normalizer<4,>=2 in ./.venv/lib/python3.12/site-packages (from requests>=2.32.2->datasets>=2.20->autoawq>=0.2.6->sentinel-ai==0.1.0) (3.4.9)
Requirement already satisfied: defusedxml>=0.7.1 in ./.venv/lib/python3.12/site-packages (from supervision>=0.24->sentinel-ai==0.1.0) (0.7.1)
Requirement already satisfied: matplotlib>=3.6 in ./.venv/lib/python3.12/site-packages (from supervision>=0.24->sentinel-ai==0.1.0) (3.11.1)
Requirement already satisfied: opencv-python>=4.5.5.64 in ./.venv/lib/python3.12/site-packages (from supervision>=0.24->sentinel-ai==0.1.0) (5.0.0.93)
Requirement already satisfied: pydeprecate<0.10,>=0.9 in ./.venv/lib/python3.12/site-packages (from supervision>=0.24->sentinel-ai==0.1.0) (0.9.0)
Requirement already satisfied: scipy>=1.10 in ./.venv/lib/python3.12/site-packages (from supervision>=0.24->sentinel-ai==0.1.0) (1.18.0)
Requirement already satisfied: contourpy>=1.0.1 in ./.venv/lib/python3.12/site-packages (from matplotlib>=3.6->supervision>=0.24->sentinel-ai==0.1.0) (1.3.3)
Requirement already satisfied: cycler>=0.10 in ./.venv/lib/python3.12/site-packages (from matplotlib>=3.6->supervision>=0.24->sentinel-ai==0.1.0) (0.12.1)
Requirement already satisfied: fonttools>=4.28.2 in ./.venv/lib/python3.12/site-packages (from matplotlib>=3.6->supervision>=0.24->sentinel-ai==0.1.0) (4.63.0)
Requirement already satisfied: kiwisolver>=1.3.1 in ./.venv/lib/python3.12/site-packages (from matplotlib>=3.6->supervision>=0.24->sentinel-ai==0.1.0) (1.5.0)
Requirement already satisfied: pyparsing>=3 in ./.venv/lib/python3.12/site-packages (from matplotlib>=3.6->supervision>=0.24->sentinel-ai==0.1.0) (3.3.2)
Requirement already satisfied: python-dateutil>=2.7 in ./.venv/lib/python3.12/site-packages (from matplotlib>=3.6->supervision>=0.24->sentinel-ai==0.1.0) (2.9.0.post0)
Requirement already satisfied: six>=1.5 in ./.venv/lib/python3.12/site-packages (from python-dateutil>=2.7->matplotlib>=3.6->supervision>=0.24->sentinel-ai==0.1.0) (1.17.0)
Requirement already satisfied: setuptools>=77.0.3 in ./.venv/lib/python3.12/site-packages (from torch>=2.5->sentinel-ai==0.1.0) (83.0.0)
Requirement already satisfied: sympy>=1.13.3 in ./.venv/lib/python3.12/site-packages (from torch>=2.5->sentinel-ai==0.1.0) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in ./.venv/lib/python3.12/site-packages (from torch>=2.5->sentinel-ai==0.1.0) (3.6.1)
Requirement already satisfied: jinja2 in ./.venv/lib/python3.12/site-packages (from torch>=2.5->sentinel-ai==0.1.0) (3.1.6)
Requirement already satisfied: cuda-toolkit==13.0.3 in ./.venv/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch>=2.5->sentinel-ai==0.1.0) (13.0.3.0)
Requirement already satisfied: cuda-bindings<14,>=13.0.3 in ./.venv/lib/python3.12/site-packages (from torch>=2.5->sentinel-ai==0.1.0) (13.3.1)
Requirement already satisfied: nvidia-cudnn-cu13==9.20.0.48 in ./.venv/lib/python3.12/site-packages (from torch>=2.5->sentinel-ai==0.1.0) (9.20.0.48)
Requirement already satisfied: nvidia-cusparselt-cu13==0.8.1 in ./.venv/lib/python3.12/site-packages (from torch>=2.5->sentinel-ai==0.1.0) (0.8.1)
Requirement already satisfied: nvidia-nccl-cu13==2.29.7 in ./.venv/lib/python3.12/site-packages (from torch>=2.5->sentinel-ai==0.1.0) (2.29.7)
Requirement already satisfied: nvidia-nvshmem-cu13==3.4.5 in ./.venv/lib/python3.12/site-packages (from torch>=2.5->sentinel-ai==0.1.0) (3.4.5)
Requirement already satisfied: nvidia-cublas==13.1.1.3.* in ./.venv/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch>=2.5->sentinel-ai==0.1.0) (13.1.1.3)
Requirement already satisfied: nvidia-cuda-nvrtc==13.0.88.* in ./.venv/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch>=2.5->sentinel-ai==0.1.0) (13.0.88)
Requirement already satisfied: nvidia-cuda-runtime==13.0.96.* in ./.venv/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch>=2.5->sentinel-ai==0.1.0) (13.0.96)
Requirement already satisfied: nvidia-cufft==12.0.0.61.* in ./.venv/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch>=2.5->sentinel-ai==0.1.0) (12.0.0.61)
Requirement already satisfied: nvidia-nvjitlink<14,>=13.0.88 in ./.venv/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch>=2.5->sentinel-ai==0.1.0) (13.3.33)
Requirement already satisfied: nvidia-cufile==1.15.1.6.* in ./.venv/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch>=2.5->sentinel-ai==0.1.0) (1.15.1.6)
Requirement already satisfied: nvidia-cuda-cupti==13.0.85.* in ./.venv/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch>=2.5->sentinel-ai==0.1.0) (13.0.85)
Requirement already satisfied: nvidia-curand==10.4.0.35.* in ./.venv/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch>=2.5->sentinel-ai==0.1.0) (10.4.0.35)
Requirement already satisfied: nvidia-cusolver==12.0.4.66.* in ./.venv/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch>=2.5->sentinel-ai==0.1.0) (12.0.4.66)
Requirement already satisfied: nvidia-cusparse==12.6.3.3.* in ./.venv/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch>=2.5->sentinel-ai==0.1.0) (12.6.3.3)
Requirement already satisfied: nvidia-nvtx==13.0.85.* in ./.venv/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch>=2.5->sentinel-ai==0.1.0) (13.0.85)
Requirement already satisfied: cuda-pathfinder>=1.4.2 in ./.venv/lib/python3.12/site-packages (from cuda-bindings<14,>=13.0.3->torch>=2.5->sentinel-ai==0.1.0) (1.6.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in ./.venv/lib/python3.12/site-packages (from sympy>=1.13.3->torch>=2.5->sentinel-ai==0.1.0) (1.3.0)
Requirement already satisfied: regex>=2025.10.22 in ./.venv/lib/python3.12/site-packages (from transformers>=4.49->sentinel-ai==0.1.0) (2026.7.19)
Requirement already satisfied: typer in ./.venv/lib/python3.12/site-packages (from transformers>=4.49->sentinel-ai==0.1.0) (0.27.0)
Requirement already satisfied: torchvision>=0.9.0 in ./.venv/lib/python3.12/site-packages (from ultralytics>=8.3->sentinel-ai==0.1.0) (0.28.0)
Requirement already satisfied: polars>=0.20.0 in ./.venv/lib/python3.12/site-packages (from ultralytics>=8.3->sentinel-ai==0.1.0) (1.43.1)
Requirement already satisfied: nvidia-ml-py>=12.0.0 in ./.venv/lib/python3.12/site-packages (from ultralytics>=8.3->sentinel-ai==0.1.0) (13.610.43)
Requirement already satisfied: ultralytics-thop>=2.1.6 in ./.venv/lib/python3.12/site-packages (from ultralytics>=8.3->sentinel-ai==0.1.0) (2.1.6)
Requirement already satisfied: polars-runtime-32==1.43.1 in ./.venv/lib/python3.12/site-packages (from polars>=0.20.0->ultralytics>=8.3->sentinel-ai==0.1.0) (1.43.1)
Requirement already satisfied: httptools>=0.8.0 in ./.venv/lib/python3.12/site-packages (from uvicorn[standard]>=0.31; extra == "runtime"->sentinel-ai==0.1.0) (0.8.0)
Requirement already satisfied: uvloop>=0.15.1 in ./.venv/lib/python3.12/site-packages (from uvicorn[standard]>=0.31; extra == "runtime"->sentinel-ai==0.1.0) (0.22.1)
Requirement already satisfied: watchfiles>=0.20 in ./.venv/lib/python3.12/site-packages (from uvicorn[standard]>=0.31; extra == "runtime"->sentinel-ai==0.1.0) (1.2.0)
Requirement already satisfied: websockets>=13.0 in ./.venv/lib/python3.12/site-packages (from uvicorn[standard]>=0.31; extra == "runtime"->sentinel-ai==0.1.0) (17.0.1)
Requirement already satisfied: argon2-cffi-bindings in ./.venv/lib/python3.12/site-packages (from argon2-cffi->minio>=7.2->sentinel-ai==0.1.0) (25.1.0)
Requirement already satisfied: cffi>=1.0.1 in ./.venv/lib/python3.12/site-packages (from argon2-cffi-bindings->argon2-cffi->minio>=7.2->sentinel-ai==0.1.0) (2.1.0)
Requirement already satisfied: pycparser in ./.venv/lib/python3.12/site-packages (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi->minio>=7.2->sentinel-ai==0.1.0) (3.0)
Requirement already satisfied: MarkupSafe>=2.0 in ./.venv/lib/python3.12/site-packages (from jinja2->torch>=2.5->sentinel-ai==0.1.0) (3.0.3)
Requirement already satisfied: shellingham>=1.3.0 in ./.venv/lib/python3.12/site-packages (from typer->transformers>=4.49->sentinel-ai==0.1.0) (1.5.4)
Requirement already satisfied: rich>=13.8.0 in ./.venv/lib/python3.12/site-packages (from typer->transformers>=4.49->sentinel-ai==0.1.0) (15.0.0)
Requirement already satisfied: markdown-it-py>=2.2.0 in ./.venv/lib/python3.12/site-packages (from rich>=13.8.0->typer->transformers>=4.49->sentinel-ai==0.1.0) (4.2.0)
Requirement already satisfied: mdurl~=0.1 in ./.venv/lib/python3.12/site-packages (from markdown-it-py>=2.2.0->rich>=13.8.0->typer->transformers>=4.49->sentinel-ai==0.1.0) (0.1.2)
Building wheels for collected packages: sentinel-ai
  Building editable for sentinel-ai (pyproject.toml): started
  Building editable for sentinel-ai (pyproject.toml): finished with status 'done'
  Created wheel for sentinel-ai: filename=sentinel_ai-0.1.0-0.editable-py3-none-any.whl size=3077 sha256=d8d4093d68283b5ef062a53e5c1a691644ee8b50124952a1ec00fc5e36d10870
  Stored in directory: /tmp/pip-ephem-wheel-cache-6x8qs86m/wheels/b9/39/94/6df3548991d9d90e78d26ea6dbc9ab17e45e3e95095e0b3261
Successfully built sentinel-ai
Installing collected packages: sentinel-ai
  Attempting uninstall: sentinel-ai
    Found existing installation: sentinel-ai 0.1.0
    Uninstalling sentinel-ai-0.1.0:
      Successfully uninstalled sentinel-ai-0.1.0
Successfully installed sentinel-ai-0.1.0
PIP_EXIT=0
name, memory.total [MiB], driver_version, compute_cap
NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 580.173.02, 8.9
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
NVIDIA_SMI[baseline_desktop_idle]: 694 MiB, 8188 MiB
Traceback (most recent call last):
  File "/tmp/spike_qwen_awq_probe.py", line 28, in <module>
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/imran/Documents/github/SentinelAI/ai-engine/.venv/lib/python3.12/site-packages/transformers/modeling_utils.py", line 4332, in from_pretrained
    hf_quantizer, config, device_map = get_hf_quantizer(
                                       ^^^^^^^^^^^^^^^^^
  File "/home/imran/Documents/github/SentinelAI/ai-engine/.venv/lib/python3.12/site-packages/transformers/quantizers/auto.py", line 354, in get_hf_quantizer
    hf_quantizer.validate_environment(
  File "/home/imran/Documents/github/SentinelAI/ai-engine/.venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_awq.py", line 50, in validate_environment
    raise ImportError(
ImportError: Loading an AWQ quantized model requires gptqmodel. Please install it with `pip install gptqmodel`
PROBE_EXIT=1

=== AutoAWQ cheap-fix attempts (within ~20min budget) ===

Attempt 1: as-installed (pyproject floating pin resolved transformers==5.14.1, torch==2.13.0+cu130, autoawq==0.2.9)
Result: ImportError at from_pretrained() — transformers' AwqQuantizer.validate_environment()
unconditionally requires `gptqmodel` now (transformers/quantizers/quantizer_awq.py), regardless
of autoawq being installed. transformers has fully removed autoawq detection
(no is_autoawq_available() in transformers/utils/import_utils.py at all).
  ImportError: Loading an AWQ quantized model requires gptqmodel. Please install it with `pip install gptqmodel`

Attempt 2: pinned transformers==4.49.0 (the pyproject.toml floor pin)
Result: past the gptqmodel gate, but ModuleNotFoundError — autoawq 0.2.9's own model registry
(awq/models/__init__.py) unconditionally imports Qwen3 modeling code that does not exist in
transformers 4.49.0 (Qwen3 support landed later in transformers' history than 4.49).
  ModuleNotFoundError: No module named 'transformers.models.qwen3'

Attempt 3: pinned transformers==4.51.3 — the exact version autoawq's own deprecation banner
names as "the last tested configuration" (paired with torch 2.6.0; we have torch 2.13.0).
Result: model loads (NVIDIA_SMI[after_model_load]: 4249 MiB, 8188 MiB) with torch_dtype="auto".
inference (model.generate) fails:
  triton.compiler.errors.CompilationError: ... Both operands must be same dtype. Got fp16 and bf16
  (mismatch between AWQ scales stored fp16 and model activations auto-resolved to bf16)

Attempt 4: same transformers==4.51.3, explicit torch_dtype=torch.float16 (as suggested by
transformers' own runtime warning: "We suggest you to set torch_dtype=torch.float16 for better
efficiency with AWQ").
Result: model loads (NVIDIA_SMI[after_model_load]: 4279 MiB, 8188 MiB). Inference gets further —
past all decoder attention/MLP layers — then fails inside autoawq's own Triton GEMM kernel at the
lm_head projection:
  triton.compiler.errors.CompilationError: at 102:13:
      b = (b >> shifts) & 0xF
           ^
  IncompatibleTypeErrorImpl('invalid operands of type triton.language.float16 and triton.language.float16')

This is a bug internal to autoawq's bundled Triton kernel (awq/modules/triton/gemm.py) — it is
incompatible with the installed Triton 3.7.1 (pulled in transitively by torch 2.13). It is not a
missing system package or a resolvable pin: autoawq has no CUDA-extension wheel for this
torch/CUDA/Triton combination (no awq_ext), falls back to its own Triton kernel, and that kernel
itself throws a type error unrelated to any model config we control. Consistent with the
project's premise: autoawq was deprecated by its maintainer in 2025 and is not maintained against
current torch/transformers/triton releases.

DECISION: Branch B — autoawq does not work on this machine. Falling back to
transformers + bitsandbytes 4-bit NF4 for the unquantised Qwen/Qwen2.5-VL-3B-Instruct checkpoint.
Verifying that fallback below.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
NVIDIA_SMI[baseline_desktop_idle]: 694 MiB, 8188 MiB
Fetching 2 files:   0%|          | 0/2 [00:00<?, ?it/s]Fetching 2 files:  50%|█████     | 1/2 [02:26<02:26, 146.80s/it]Fetching 2 files: 100%|██████████| 2/2 [02:36<00:00, 66.15s/it] Fetching 2 files: 100%|██████████| 2/2 [02:36<00:00, 78.25s/it]
Loading weights:   0%|          | 0/824 [00:00<?, ?it/s]Loading weights:   0%|          | 1/824 [00:00<02:51,  4.79it/s]Loading weights:   2%|▏         | 15/824 [00:00<00:13, 57.88it/s]Loading weights:   4%|▍         | 35/824 [00:00<00:07, 107.88it/s]Loading weights:   6%|▋         | 53/824 [00:00<00:06, 125.54it/s]Loading weights:   9%|▉         | 77/824 [00:00<00:04, 154.77it/s]Loading weights:  12%|█▏        | 100/824 [00:00<00:04, 176.52it/s]Loading weights:  15%|█▍        | 123/824 [00:00<00:03, 190.20it/s]Loading weights:  17%|█▋        | 143/824 [00:00<00:03, 188.40it/s]Loading weights:  20%|█▉        | 163/824 [00:01<00:03, 182.58it/s]Loading weights:  22%|██▏       | 184/824 [00:01<00:03, 188.05it/s]Loading weights:  25%|██▌       | 207/824 [00:01<00:03, 198.51it/s]Loading weights:  28%|██▊       | 232/824 [00:01<00:02, 208.84it/s]Loading weights:  31%|███       | 255/824 [00:01<00:02, 208.85it/s]Loading weights:  34%|███▎      | 277/824 [00:01<00:02, 194.45it/s]Loading weights:  36%|███▌      | 297/824 [00:01<00:02, 187.51it/s]Loading weights:  38%|███▊      | 316/824 [00:01<00:02, 178.79it/s]Loading weights:  41%|████      | 335/824 [00:02<00:02, 165.79it/s]Loading weights:  43%|████▎     | 352/824 [00:02<00:03, 125.93it/s]Loading weights:  45%|████▍     | 367/824 [00:02<00:03, 123.32it/s]Loading weights:  46%|████▌     | 381/824 [00:02<00:04, 109.31it/s]Loading weights:  48%|████▊     | 397/824 [00:02<00:03, 120.11it/s]Loading weights:  50%|█████     | 412/824 [00:02<00:03, 122.02it/s]Loading weights:  52%|█████▏    | 425/824 [00:02<00:03, 106.13it/s]Loading weights:  54%|█████▍    | 445/824 [00:03<00:02, 127.78it/s]Loading weights:  56%|█████▌    | 459/824 [00:03<00:02, 129.29it/s]Loading weights:  58%|█████▊    | 480/824 [00:03<00:02, 149.47it/s]Loading weights:  64%|██████▍   | 526/824 [00:03<00:01, 231.01it/s]Loading weights:  72%|███████▏  | 596/824 [00:03<00:00, 360.40it/s]Loading weights:  81%|████████  | 666/824 [00:03<00:00, 456.22it/s]Loading weights:  89%|████████▉ | 732/824 [00:03<00:00, 514.37it/s]Loading weights:  97%|█████████▋| 802/824 [00:03<00:00, 566.72it/s]Loading weights: 100%|██████████| 824/824 [00:03<00:00, 217.32it/s]
NVIDIA_SMI[after_model_load]: 3359 MiB, 8188 MiB
NVIDIA_SMI[after_inference_peak]: 3517 MiB, 8188 MiB
INFERENCE_OK elapsed_seconds=2.66
OUTPUT_TAIL='system\nYou are a helpful assistant.\nuser\nDescribe this scene in one sentence.\nassistant\nThe image is a solid blue color with no other objects or details present.'
TORCH_VERSION=2.13.0+cu130
TRANSFORMERS_VERSION=5.14.1
BITSANDBYTES_VERSION=0.50.0
TORCH_CUDA_MAX_ALLOCATED_MIB=2564
BNB_PROBE_EXIT=0
```
