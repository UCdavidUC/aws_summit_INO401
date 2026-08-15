# Technical Documentation: Small Language Models for CPU Inference

## 1. Introduction

This document presents the technical evaluation of frameworks and models suitable for fine tuning and deploying Small Language Models (SLMs) on CPU based infrastructure, targeting both x86_64 and ARM architectures. The findings support the hybrid multi region architecture where training occurs in GPU enabled AWS regions and inference runs on CPU only hardware in the Mexico (Queretaro) mx-central-1 region.

## 2. Architecture Overview

The deployment pipeline follows a linear flow:

1. Fine tune SLMs on GPU instances in us-west-2 (Oregon), the designated training region
2. Optimize and quantize models for CPU inference (GGUF Q4_K_M preferred, ONNX INT4 as fallback)
3. Replicate optimized artifacts via S3 Cross Region Replication to mx-central-1 (Queretaro), the designated deployment region
4. Serve on CPU only hardware via ECS, ECS Anywhere, or Lambda in mx-central-1

## 3. Candidate Models

### 3.1 Models Validated for CPU Inference

The following models have been evaluated based on published benchmarks, production evidence, and community validation as of 2025.

| Model | Parameters | Runs Unquantized on CPU | Quantized CPU Performance | Primary Use Case |
|-------|-----------|------------------------|--------------------------|-----------------|
| Qwen 3 0.6B | 600M | Yes, full precision on 8GB RAM | Not required | Tool use, reasoning, lightweight agents |
| Gemma 3 270M | 270M | Yes, full precision on 8GB RAM | Not required | Chat, fine tuning experiments |
| Gemma 3 1B | 1B | Yes, full precision on CPU | Not required | General purpose, middle ground |
| Qwen 2.5 1.5B/3B | 1.5B to 3B | Marginal at 3B | INT4/INT8 recommended | Multilingual, strong benchmarks |
| Llama 3.2 1B/3B | 1B to 3B | 1B yes, 3B marginal | Q4_K_M gives 6.5GB RAM usage | Best ecosystem and fine tuning support |
| Phi 4 Mini | 3.8B | No (requires more than 16GB) | INT4 ONNX runs fast on CPU | Reasoning, math, coding, 128K context |
| AFM 4.5B (Arcee) | 4.5B | bf16 works on 32 core CPU | Q4_0: 136 tok/s on Intel Xeon | Best documented CPU benchmarks |
| Mistral 7B | 7B | No | INT4 required | General purpose, slower on CPU |

### 3.2 Model Selection Criteria

When selecting a model for CPU deployment in a compliance context, the following factors apply:

1. **Memory footprint**: The quantized model must fit within the available RAM of the target hardware (typically 8 to 16 GB for edge nodes)
2. **Tokens per second**: Interactive applications require at minimum 10 tokens per second for acceptable user experience
3. **Perplexity degradation**: Quantization from bf16 to INT4 should not exceed 1 to 2 percent perplexity increase
4. **Multilingual support**: For Mexican Spanish regulatory content, models with strong multilingual training data are preferred
5. **License**: Apache 2.0 or similarly permissive licenses are required for on premises compliance workloads

### 3.3 Recommended Models

**Primary: Phi 4 Mini (3.8B)**

Microsoft provides official ONNX INT4 exports via the Hugging Face repository `microsoft/Phi-4-mini-instruct-onnx`. The model features a 128K context window and native tool calling capabilities. It offers the best ONNX Runtime support given that both the model and the runtime are Microsoft products. A notable advantage is that CPU based fine tuning is documented and validated (QLoRA on CPU via Azure Machine Learning). The 3.8B parameter count sits in the sweet spot for CPU inference with INT4 quantization.

**Alternative: Llama 3.2 3B**

Meta's Llama 3.2 3B demonstrates the best ARM/Graviton performance due to Arm Kleidi kernel optimizations in llama.cpp. It has an excellent fine tuning ecosystem with LoRA, QLoRA, Unsloth, and TRL support. The Q4_K_M quantization reduces memory to approximately 6.5GB with less than 1 percent perplexity loss. Strong community adoption means broad tooling and container image availability.

**Lightweight Option: Qwen 3 0.6B or Gemma 3 1B**

These models run unquantized on CPU without any optimization step, which simplifies the deployment pipeline. They are suitable for constrained edge hardware with 4 to 8 GB of RAM. Qwen 3 0.6B notably supports reasoning mode and tool use despite its small size.

## 4. Inference Frameworks

### 4.1 Framework Comparison

| Framework | Best Scenario | x86 Performance | ARM Performance | Key Advantage |
|-----------|--------------|----------------|-----------------|---------------|
| llama.cpp (GGUF) | Maximum CPU speed for LLMs | High (AVX-512/AMX) | Excellent (SVE/NEON, 3x over x86) | Single binary, zero dependencies, OpenAI compatible server |
| ONNX Runtime | Portability and heterogeneous hardware | High (AVX2/512) | High (NEON) | Cross platform, NPU delegation, Microsoft backed |
| Hugging Face Optimum | Easy export from training to ONNX | Via ONNX Runtime backend | Via ONNX Runtime backend | Seamless Hugging Face to ONNX pipeline |
| vLLM (CPU mode) | OpenAI compatible batch serving | Moderate | Experimental | Better suited for GPU; CPU mode still maturing |
| OpenVINO | Intel only optimization | Excellent (Intel specific) | Not supported | Best choice for Intel Xeon hardware specifically |

### 4.2 Framework Analysis

**llama.cpp**

The llama.cpp project is the de facto standard for CPU based LLM inference. It supports AVX2, AVX-512, ARM NEON, and SVE/SVE2 instruction sets. Models are served in GGUF format (Grokking GGML Unified Format) which packs quantized weights and metadata into a single file. The project includes a built in HTTP server compatible with the OpenAI API specification. Key optimizations include memory mapped model loading, block interleaved weight repacking for wider vector registers, and multi threaded tensor operations across CPU cores.

For ARM targets specifically, llama.cpp with Arm Kleidi kernels delivers up to 3x performance improvement over x86 for prompt processing. AWS Graviton3 meets the 100ms latency target for interactive LLM deployments. Graviton4 based instances (c8g) deliver 40 tokens per second at batch size 1 for a 4.5B model quantized to Q4_0.

**ONNX Runtime**

ONNX Runtime is a cross platform high performance inference engine that supports the Open Neural Network Exchange format. It provides optimized execution providers for CPU (x86 and ARM), GPU, and NPU hardware. The Hugging Face Optimum library integrates directly with ONNX Runtime for model export and quantization.

Microsoft publishes pre optimized ONNX models for Phi 4 Mini with INT4 quantization, which can be loaded directly without additional conversion steps. ONNX Runtime supports both dynamic quantization (post training, simpler) and static quantization (requires calibration data, more accurate). The runtime handles operator fusion, constant folding, and memory planning automatically.

For heterogeneous environments where the same model must run on different hardware (Intel x86, AMD x86, ARM Graviton, local workstations), ONNX Runtime provides the most portable solution.

**Hugging Face Optimum**

Optimum serves as the bridge between training (Hugging Face Transformers) and optimized inference (ONNX Runtime, OpenVINO). It handles model export to ONNX format, graph optimization, and quantization in a unified API. The typical workflow is:

1. Fine tune a model using Transformers and PEFT (LoRA/QLoRA)
2. Export to ONNX using `optimum-cli export onnx`
3. Quantize using ORTQuantizer (INT8 dynamic or static)
4. Deploy using ORTModelForCausalLM

### 4.3 Framework Recommendation

For the target architecture (CPU inference in mx-central-1):

If deploying on ARM hardware (Graviton instances or ARM based on premises servers): use **llama.cpp** with GGUF Q4_K_M or Q5_K_M quantization. The 3x to 4x cost performance advantage over x86 is well documented.

If deploying on x86 hardware or mixed environments: use **ONNX Runtime** with INT4 quantized ONNX models exported via Hugging Face Optimum. This provides maximum portability across different x86 vendors (Intel, AMD) and can also run on ARM.

If simplicity is the priority and the model is small enough (sub 1B): use **Hugging Face Transformers directly** with no quantization or conversion step.

## 5. CPU Architecture Performance

### 5.1 x86_64 (Intel, AMD)

Modern x86 processors support hardware acceleration for low precision inference:

| Feature | Intel Xeon (Sapphire Rapids+) | AMD EPYC (Zen 4+) |
|---------|------------------------------|-------------------|
| AVX-512 | Supported | Supported |
| VNNI (Vector Neural Network Instructions) | Supported | Supported |
| AMX (Advanced Matrix Extensions) | Supported | Not available |
| BF16 native | Supported | Supported (Zen 4+) |

Intel AMX provides dedicated matrix multiplication hardware with tile based registers, delivering significant throughput gains for quantized inference. AMD compensates with higher core counts and memory bandwidth.

Benchmark reference (Arcee AI, AFM 4.5B, Q4_0, Intel Xeon c7i 32 threads): 136.77 total tokens per second at batch size 4.

### 5.2 ARM (Graviton, Neoverse, Apple Silicon)

ARM processors offer superior performance per watt and per dollar for CPU inference:

| Feature | AWS Graviton3 (Neoverse V1) | AWS Graviton4 (Neoverse V2) |
|---------|----------------------------|----------------------------|
| NEON (128 bit SIMD) | Supported | Supported |
| SVE/SVE2 (Scalable Vector Extension) | SVE (256 bit) | SVE2 (variable width) |
| SDOT/MMLA instructions | Supported | Enhanced |
| Memory bandwidth | DDR5 | DDR5 (higher) |

Benchmark reference (Arcee AI, Virtuoso Lite 10B, Q4_0, Graviton4 c8g 32 vCPU): 40 tokens per second at batch size 1, which is 4x the throughput of comparable Intel instances at lower cost ($1.28/hr vs $1.43/hr).

ARM Kleidi provides optimized kernels that allow AI frameworks and libraries to unlock the performance of ARM CPUs without vendor specific add ons. The optimizations are integrated into llama.cpp by default.

### 5.3 Cost Performance Comparison

| Instance | Architecture | Cores/vCPUs | Cost (us-east-1 on-demand) | AFM 4.5B Q4_0 tok/s | Tokens per Dollar |
|----------|-------------|-------------|---------------------------|---------------------|-------------------|
| c8g.8xlarge | ARM Graviton4 | 32 vCPU | $1.276/hr | 40 (batch 1) | 112,853 |
| c7i.8xlarge | Intel Xeon | 32 vCPU | $1.428/hr | 10 (batch 1) | 25,210 |

Graviton4 provides 4.5x cost performance advantage for CPU inference workloads.

## 6. Quantization Methods

### 6.1 Quantization Comparison

| Method | Format | Size Reduction | Quality Loss | x86 Compatibility | ARM Compatibility |
|--------|--------|---------------|-------------|-------------------|-------------------|
| INT8 Dynamic (ONNX) | ONNX | 4x | Minimal | Fast (AVX2) | Fast (NEON) |
| INT4 AWQ | Safetensors | 8x | Low to moderate | Moderate | Moderate |
| GGUF Q4_K_M | GGUF | 6x | Low (less than 1% perplexity) | Fast | Fast |
| GGUF Q5_K_M | GGUF | 5x | Very low | Fast | Fast |
| GGUF Q8_0 | GGUF | 2x | Negligible | Very fast | Very fast |
| GPTQ INT4 | Safetensors | 8x | Low | Requires GPU | Not recommended |

### 6.2 Perplexity Impact

Quantization quality measurements from Arcee AI (AFM 4.5B, 2025):

| Quantization | Perplexity Change vs bf16 |
|-------------|--------------------------|
| bf16 (baseline) | 0% |
| Q8_0 (8 bit) | 0% (no measurable change) |
| Q4_K_M (4 bit K means) | +1% |
| Q4_0 (4 bit basic) | +1% to +2% |

These results confirm that 4 bit quantization via K Quant methods maintains model quality within acceptable margins for production use.

### 6.3 Recommended Quantization Strategy

For the deployment pipeline:

1. Fine tune on GPU in full precision (bf16)
2. Export to target format:
   a. For ONNX Runtime path: export via Hugging Face Optimum, quantize to INT4 with dynamic quantization
   b. For llama.cpp path: convert to GGUF using llama.cpp conversion scripts, quantize to Q4_K_M
3. Validate perplexity on held out evaluation set (accept if delta is less than 2%)
4. Upload optimized artifact to S3 for cross region replication

## 7. Fine Tuning Pipeline

### 7.1 Supported Approaches

| Model | Fine Tune Method | GPU Required | Export Path | CPU Inference Runtime |
|-------|-----------------|-------------|-------------|----------------------|
| Phi 4 Mini | QLoRA (4 bit) + LoRA adapters | Yes (T4/A10G on SageMaker) | ONNX INT4 via Optimum | ONNX Runtime |
| Phi 4 Mini | QLoRA on CPU (Azure ML method) | No (CPU only) | ONNX INT4 via Optimum | ONNX Runtime |
| Llama 3.2 3B | QLoRA + Unsloth/TRL | Yes (T4 minimum) | GGUF via llama.cpp convert | llama.cpp server |
| Qwen 2.5 3B | LoRA + SageMaker | Yes (g5.xlarge) | GGUF or ONNX | Either framework |
| Gemma 3 1B | Full fine tune on Colab GPU | Yes (T4) | Run directly (no quantization needed) | PyTorch Transformers |

### 7.2 SageMaker Integration

Fine tuning on SageMaker uses the Hugging Face Deep Learning Containers in us-west-2 (Oregon):

1. Launch a training job in us-west-2 on ml.g6.xlarge (or ml.p3.2xlarge for larger models)
2. Use the HuggingFace estimator with PEFT/LoRA configuration
3. Hyperparameters include learning rate, batch size, number of epochs, LoRA rank, and target modules
4. Training output is stored in S3 as merged model weights
5. The optimization Lambda function triggers on upload to convert and quantize for CPU

### 7.3 LoRA and QLoRA Configuration

Parameter Efficient Fine Tuning (PEFT) via LoRA reduces trainable parameters to less than 1% of the full model:

| Configuration | Llama 3.2 3B | Phi 4 Mini | Qwen 2.5 3B |
|--------------|-------------|-----------|-------------|
| LoRA rank | 16 to 64 | 16 to 32 | 16 to 64 |
| Target modules | q_proj, v_proj, k_proj, o_proj | q_proj, v_proj | q_proj, v_proj, k_proj, o_proj |
| Trainable parameters | ~12M (0.37% of 3.2B) | ~8M | ~12M |
| Quantization during training | 4 bit (NF4) | 4 bit (NF4) | 4 bit (NF4) |
| Training memory | ~8GB VRAM (T4 compatible) | ~6GB VRAM | ~8GB VRAM |

## 8. Deployment Considerations for mx-central-1

### 8.1 Region Constraints

The Mexico (Queretaro) mx-central-1 region is the designated Deployment Region. It has the following constraints relevant to this architecture:

1. No GPU instances available (no ml.g*, ml.p*, inf*, or trn* instance families)
2. No SageMaker service (training occurs in us-west-2)
3. EC2 CPU instances are available: Graviton (c6g, c7g, m6g, m7g, r6g, r7g, r8g, t4g) and Intel (c6i, c7i, m6i, m7i, r6i, r7i, t3)
4. ECS, ECS Anywhere, and Lambda (ARM64 and x86_64) are supported for container workloads
5. S3 is available as a cross-region replication destination from us-west-2

### 8.2 Container Strategy

The inference container should include:

For the ONNX Runtime path:
1. Base image: Python 3.12 slim or ARM64 equivalent
2. Dependencies: onnxruntime, tokenizers, numpy
3. Model loading: Download INT4 ONNX model from local S3 bucket at container start
4. Serving: FastAPI or Flask endpoint exposing an OpenAI compatible API

For the llama.cpp path:
1. Base image: Ubuntu or Alpine with compiled llama.cpp server binary
2. No Python dependencies required (pure C++ binary)
3. Model loading: Download GGUF file from local S3 bucket at container start
4. Serving: Built in llama.cpp server (OpenAI compatible API on port 8080)

### 8.3 Hardware Sizing

Recommended ECS task definitions for CPU inference:

| Model | Quantization | CPU Units | Memory (MiB) | Expected Performance |
|-------|-------------|-----------|-------------|---------------------|
| Phi 4 Mini INT4 ONNX | INT4 | 4096 (4 vCPU) | 8192 | 15 to 25 tok/s |
| Llama 3.2 3B Q4_K_M | GGUF Q4_K_M | 4096 (4 vCPU) | 8192 | 20 to 35 tok/s (ARM) |
| Qwen 3 0.6B (unquantized) | None | 2048 (2 vCPU) | 4096 | 30 to 50 tok/s |
| Gemma 3 1B (unquantized) | None | 2048 (2 vCPU) | 4096 | 20 to 40 tok/s |

## 9. References

1. Arcee AI. "Is Running Language Models on CPU Really Viable?" July 2025. https://www.arcee.ai/blog/is-running-language-models-on-cpu-really-viable
2. Julien Simon. "The Case for Small Language Model Inference on Arm CPUs." April 2025. https://www.julien.org/blog/arcee-posts/2025-04-17_the-case-for-small-language-model-inference-on-arm-cpus/
3. Arm Newsroom. "Small Language Models: Efficient Arm Computing Enables a Custom AI Future." May 2024. https://newsroom.arm.com/blog/small-language-models-on-arm
4. WhileOne. "Benchmarking Meta Llama 4 Scout on CPU Only Systems." April 2025. https://www.whileone.in/post/benchmarking-meta-llama-4-scout-on-cpu-only-systems-performance-quantization-and-architecture-tuning
5. Neurl Creators. "Small Language Models: LLMs You Can Run on Your CPU Without Quantization." November 2025. https://neurlcreators.substack.com/p/llms-you-can-run-on-your-cpu-without
6. Microsoft. "Phi-4-Mini-Instruct ONNX." Hugging Face. https://huggingface.co/microsoft/Phi-4-mini-instruct-onnx
7. Hugging Face. "Optimum ONNX Runtime Quantization." https://huggingface.co/docs/optimum-onnx/onnxruntime/usage_guides/quantization
8. ClearML. "Benchmarking llama.cpp on Arm Neoverse based AWS Graviton instances." https://clear.ml/blog/benchmarking-llama-cpp-on-arm-neoverse-based-aws-graviton-instances-with-clearml
9. AWS Documentation. "Migrate inference workload from x86 to AWS Graviton." https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-graviton.html
10. Jingkun04. "CPU Based Fine Tuning of Small Language Models with Azure Machine Learning (Phi 4)." GitHub. https://github.com/Jingkun04/CPU-Based-Fine-Tuning-of-Small-Language-Models-SLMs-with-Azure-Machine-Learning-Phi-4
11. AWS Documentation. "SageMaker Supported Models for Optimization." https://docs.aws.amazon.com/sagemaker/latest/dg/optimization-supported-models.html
12. GitHub. "Fine Tuning Qwen 2.5 3B on AWS SageMaker." https://github.com/bhaiyahnsingh45/Fine-Tuning-Qwen-2.5-3B-on-AWS-SageMaker

## 10. Export Formats for ARM Processors

### 10.1 Format Evaluation (2025/2026)

The choice of export format directly impacts inference performance on ARM processors. Based on current benchmarks and ecosystem maturity:

| Format | ARM Performance | Ecosystem | Quantization Options | Deployment Simplicity |
|--------|----------------|-----------|---------------------|----------------------|
| GGUF (llama.cpp) | Best (Arm Kleidi kernels, SVE/NEON) | llama.cpp, Ollama, LM Studio | Q2 through Q8, K-quants, IQ variants | Single file, self contained |
| ONNX (ONNX Runtime) | Good (NEON execution provider) | ONNX Runtime, Hugging Face Optimum | INT8, INT4 (via ORT-GenAI) | Multi file, requires runtime config |
| ExecuTorch (Meta) | Emerging (mobile focused) | PyTorch ecosystem | INT8, INT4 | Mobile optimized, less server focus |
| MLX (Apple) | Apple Silicon only | MLX framework | FP16, INT4, INT8 | Apple hardware exclusive |

### 10.2 GGUF: Recommended Format for ARM Deployment

GGUF (Grokking GGML Unified Format) is the recommended export format for ARM based inference in this architecture. The rationale:

1. **Performance**: llama.cpp with Arm Kleidi kernels provides optimized GEMM operations using NEON, SVE, and MMLA instructions. These are hardware specific optimizations that deliver 3x to 4x better performance than generic ONNX Runtime on the same ARM hardware.

2. **Single file distribution**: A GGUF file contains model weights, tokenizer configuration, chat template, and all metadata in one binary file. This simplifies S3 storage, replication, and container loading compared to ONNX which requires multiple files.

3. **Quantization quality**: K-quant methods (Q4_K_M, Q5_K_M) preserve more information in high importance weight layers, delivering less than 1% perplexity increase at 4 bit precision. This approach is specific to GGUF and not available in standard ONNX quantization.

4. **Zero dependency serving**: The llama.cpp server binary is a single compiled executable with no Python runtime, no framework dependencies, and no version conflicts. This reduces container image size and eliminates dependency management issues.

5. **Architecture support in 2025/2026**: llama.cpp supports new model architectures (Llama, Phi, Qwen, Gemma, Mistral) within days of release. ONNX Runtime GenAI support typically lags by weeks to months.

### 10.3 When to Use ONNX Instead

ONNX remains the better choice in specific scenarios:

1. When the same model must run on NPU hardware (Apple Neural Engine, Qualcomm Hexagon)
2. When browser based inference is required (ONNX Runtime Web on WASM/WebGPU)
3. When Microsoft provides official pre-optimized ONNX exports (Phi 4 Mini)
4. When the workload is not an LLM (vision models, embeddings, rerankers)
5. When the deployment target includes Windows DirectML

For the mx-central-1 architecture where inference runs on Linux ARM64 (Graviton) or Linux x86_64 (Intel Xeon), GGUF is the superior choice for LLM workloads.

### 10.4 GGUF Quantization Variants for ARM

The following quantization levels are recommended for ARM deployment:

| Variant | Bits | Size (3B model) | Perplexity Delta | ARM Performance | Use Case |
|---------|------|-----------------|-----------------|-----------------|----------|
| Q4_K_M | 4 bit | ~2.0 GB | +1% | Excellent | Production default, best quality/speed tradeoff |
| Q5_K_M | 5 bit | ~2.4 GB | +0.5% | Very good | When quality is critical |
| Q4_K_S | 4 bit | ~1.9 GB | +1.2% | Excellent | Memory constrained environments |
| Q8_0 | 8 bit | ~3.4 GB | 0% | Good | Maximum quality, sufficient RAM |
| Q4_0 | 4 bit | ~1.8 GB | +1.5% | Best speed | Latency critical, quality acceptable |

For Graviton instances with 8GB or more RAM, Q4_K_M is the recommended default. For Lambda (10GB max), Q4_K_M or Q5_K_M both fit comfortably with room for KV cache.

## 11. Target Deployment Infrastructure in mx-central-1

### 11.1 Available EC2 Instance Types

The following instance families are confirmed available in mx-central-1 as of 2025:

**ARM64 (Graviton) Instances:**

| Family | Processor | Use Case | Sizes Available | Base Clock |
|--------|-----------|----------|-----------------|------------|
| c6g | Graviton2 | Compute optimized | medium through 16xlarge | 2.5 GHz |
| c6gn | Graviton2 | Compute + networking | medium through 16xlarge | 2.5 GHz |
| c7g | Graviton3 | Compute optimized (latest) | medium through 16xlarge | 2.6 GHz |
| m6g | Graviton2 | General purpose | medium through 16xlarge | 2.5 GHz |
| m7g | Graviton3 | General purpose (latest) | medium through 16xlarge | 2.6 GHz |
| r6g | Graviton2 | Memory optimized | medium through 16xlarge | 2.5 GHz |
| r7g | Graviton3 | Memory optimized (latest) | medium through 16xlarge | 2.6 GHz |
| r8g | Graviton4 | Memory optimized (newest) | medium through 48xlarge | 2.8 GHz |
| t4g | Graviton2 | Burstable | nano through 2xlarge | 2.5 GHz |

**x86_64 (Intel) Instances:**

| Family | Processor | Use Case | Sizes Available | Base Clock |
|--------|-----------|----------|-----------------|------------|
| c6i | Intel Xeon 8375C (Ice Lake) | Compute optimized | large through 32xlarge | 3.5 GHz |
| c7i | Intel Xeon (Sapphire Rapids) | Compute optimized (latest) | large through 48xlarge | 3.2 GHz |
| m6i | Intel Xeon 8375C (Ice Lake) | General purpose | large through 32xlarge | 3.5 GHz |
| m7i | Intel Xeon (Sapphire Rapids) | General purpose (latest) | large through 48xlarge | 3.2 GHz |
| r6i | Intel Xeon 8375C (Ice Lake) | Memory optimized | large through 32xlarge | 3.5 GHz |
| r7i | Intel Xeon (Sapphire Rapids) | Memory optimized (latest) | large through 48xlarge | 3.2 GHz |
| t3 | Intel Xeon | Burstable | nano through 2xlarge | 2.5 GHz |

### 11.2 Recommended Instance Selection for SLM Inference

| Workload Profile | Recommended Instance | Architecture | vCPU | RAM | Estimated tok/s (3B Q4_K_M) | Cost/hr |
|-----------------|---------------------|-------------|------|-----|------------------------------|---------|
| Low latency, single user | c7g.xlarge | ARM Graviton3 | 4 | 8 GB | 20 to 30 | ~$0.152 |
| Production, moderate load | c7g.4xlarge | ARM Graviton3 | 16 | 32 GB | 35 to 50 | ~$0.609 |
| Production, high throughput | r8g.4xlarge | ARM Graviton4 | 16 | 128 GB | 40 to 60 | ~$0.990 |
| Budget, single user | t4g.xlarge | ARM Graviton2 | 4 | 16 GB | 10 to 15 | ~$0.141 |
| Intel baseline | c7i.4xlarge | Intel Sapphire Rapids | 16 | 32 GB | 10 to 20 | ~$0.749 |
| Intel high memory | m7i.4xlarge | Intel Sapphire Rapids | 16 | 64 GB | 10 to 20 | ~$0.846 |

The Graviton instances deliver 2x to 4x better cost performance for inference workloads. The c7g and r8g families are the primary recommendation for production SLM inference.

### 11.3 AWS Lambda for SLM Inference

Lambda is a viable deployment target for Small Language Models under specific conditions:

**Capabilities (2025/2026):**

| Parameter | Value |
|-----------|-------|
| Maximum memory | 10,240 MB (10 GB) |
| CPU allocation at max memory | 6 vCPUs |
| Maximum container image size | 10 GB |
| Maximum execution time | 15 minutes |
| Architecture support | x86_64 and arm64 (Graviton) |
| SnapStart support | Yes (reduces cold starts to sub 500ms) |
| Ephemeral storage | Up to 10 GB (/tmp) |

**Feasible Models on Lambda:**

| Model | Format | Size | Fits in Lambda | Expected Performance |
|-------|--------|------|---------------|---------------------|
| Llama 3.2 1B Q4_K_M | GGUF | ~0.7 GB | Yes | 15 to 25 tok/s |
| Qwen 3 0.6B (unquantized) | PyTorch/GGUF | ~1.2 GB | Yes | 20 to 35 tok/s |
| Gemma 3 1B Q4_K_M | GGUF | ~0.7 GB | Yes | 15 to 25 tok/s |
| Phi 4 Mini Q4_K_M | GGUF | ~2.3 GB | Yes (tight) | 10 to 15 tok/s |
| Llama 3.2 3B Q4_K_M | GGUF | ~2.0 GB | Yes | 8 to 15 tok/s |

**Lambda Architecture Pattern:**

1. Package the GGUF model file inside the container image (avoid runtime downloads)
2. Use ARM64 (Graviton) runtime for 20 to 30 percent faster inference over x86
3. Set memory to maximum (10,240 MB) to unlock all 6 vCPUs
4. Use SnapStart with memfd_create for sub 500ms cold starts (stream model from S3 into RAM)
5. Use AWS Lambda Web Adapter for streaming token by token responses
6. Configure llama.cpp with n_threads equal to 6 (matching Lambda vCPU allocation)

**When Lambda is Appropriate:**

Lambda is cost effective when utilization is below 15 percent (sporadic, bursty workloads). Examples include:

1. Internal tools used intermittently by compliance teams
2. Document processing pipelines with variable load
3. Development and testing environments
4. Batch processing of regulatory documents during off peak hours

When utilization exceeds 15 percent, dedicated EC2 instances (Graviton) become more cost effective.

**Lambda Limitations for LLM Inference:**

1. Maximum context window is constrained by available RAM after model loading
2. Models larger than 5 GB are impractical due to memory pressure from KV cache
3. 15 minute timeout limits long generation tasks
4. Cold starts (even with SnapStart) add latency for the first request in a window
5. No persistent state between invocations (stateless inference only)

### 11.4 Deployment Decision Matrix

| Criteria | EC2 Graviton | EC2 Intel | Lambda ARM64 |
|----------|-------------|-----------|--------------|
| Best for | Sustained production load | Legacy x86 requirements | Sporadic/bursty workloads |
| Model size limit | Unlimited (scale vertically) | Unlimited (scale vertically) | Practical limit ~5 GB |
| Cost model | Per hour (reserved/on demand) | Per hour (reserved/on demand) | Per millisecond + per request |
| Cold start | None (always running) | None (always running) | Sub 500ms with SnapStart |
| Max throughput | 40 to 60+ tok/s | 10 to 25 tok/s | 15 to 25 tok/s |
| Scaling | Auto Scaling Groups or ECS | Auto Scaling Groups or ECS | Automatic (concurrent executions) |
| Maintenance | Patching, OS updates required | Patching, OS updates required | Fully managed |
| Format recommendation | GGUF (Q4_K_M) | GGUF or ONNX | GGUF (Q4_K_M) |
| Break even vs Lambda | Above 15% utilization | Above 15% utilization | Below 15% utilization |

## 12. Updated Deployment Architecture

Based on the infrastructure available in mx-central-1 and the export format evaluation, the recommended deployment architecture is:

### 12.1 Primary Path (Production)

1. Train on SageMaker in us-west-2 (Oregon) using Hugging Face DLCs on ml.g6.xlarge
2. Convert to GGUF Q4_K_M using llama.cpp conversion scripts
3. Replicate GGUF file to mx-central-1 (Queretaro) via S3 Cross Region Replication
4. Serve on c7g.4xlarge or r8g.4xlarge (Graviton) running llama.cpp server in a container
5. Expose OpenAI compatible API endpoint

### 12.2 Secondary Path (Serverless)

1. Same training and conversion as primary path (us-west-2)
2. Package GGUF model inside Lambda container image (ARM64)
3. Deploy Lambda function in mx-central-1 with 10,240 MB memory and SnapStart enabled
4. Use for bursty internal compliance workloads below 15% utilization
5. Scale to zero when idle (zero cost at rest)

### 12.3 Fallback Path (x86 Compatibility)

1. Same training step (us-west-2)
2. Export to ONNX INT4 via Hugging Face Optimum (for maximum portability)
3. Replicate to mx-central-1
4. Serve on c7i.4xlarge (Intel Sapphire Rapids) in mx-central-1 running ONNX Runtime
5. Use when ARM containers are not available or x86 hardware is mandated

## 13. MLflow Implementation for Model Deployment

### 13.1 MLflow in the Deployment Pipeline

MLflow provides experiment tracking, model registry, and deployment tooling that integrates into the training and deployment pipeline. The implementation covers the full lifecycle from fine tuning in us-west-2 through deployment to EC2 and ECS in mx-central-1.

**Pipeline Integration Points:**

| Stage | MLflow Component | Description |
|-------|-----------------|-------------|
| Fine tuning (us-west-2) | MLflow Tracking | Log hyperparameters, metrics (loss, perplexity), and artifacts per training run |
| Model evaluation | MLflow Tracking | Log evaluation metrics (accuracy, latency benchmarks) from SageMaker Pipeline |
| Model registration | MLflow Model Registry | Register validated models with versioning and stage transitions (staging, production) |
| Optimization | MLflow Tracking | Log quantization parameters, output format (GGUF/ONNX), and quality metrics |
| Deployment to EC2/ECS | MLflow Models + Docker | Build inference container images from registered models |

### 13.2 MLflow Tracking Server Deployment

The MLflow tracking server runs in us-west-2 alongside the training infrastructure:

1. Deploy MLflow on ECS Fargate in us-west-2 (serverless, no EC2 management)
2. Use Amazon RDS (PostgreSQL) as the backend metadata store
3. Use S3 in us-west-2 as the artifact store for model binaries and experiment data
4. Expose the MLflow UI via Application Load Balancer with authentication

**Infrastructure Components:**

| Component | Service | Region | Purpose |
|-----------|---------|--------|---------|
| MLflow Server | ECS Fargate | us-west-2 | Tracking UI and API |
| Metadata Store | RDS PostgreSQL | us-west-2 | Experiment and run metadata |
| Artifact Store | S3 | us-west-2 | Model files, metrics, parameters |
| Load Balancer | ALB | us-west-2 | HTTPS access to MLflow UI |

### 13.3 Model Registry Workflow

The MLflow Model Registry manages model lifecycle stages:

1. **Logging**: After each SageMaker training job completes, a post-training script logs the model to MLflow using `mlflow.log_model()` with the `transformers` flavor
2. **Registration**: Models that pass evaluation thresholds (perplexity delta less than 2%) are registered in the Model Registry with `mlflow.register_model()`
3. **Staging**: Registered models enter the "Staging" stage for GGUF conversion and quantization validation
4. **Production**: After successful quantization and perplexity validation, the model alias is set to "Production" via `client.set_registered_model_alias()`
5. **Deployment trigger**: Transitioning to Production triggers the deployment pipeline (Lambda or CI/CD) that pushes the container to mx-central-1

### 13.4 Deployment to EC2

MLflow models can be deployed to EC2 instances in mx-central-1 using containerized serving:

**Approach: MLflow Docker Container on EC2**

1. Build a Docker image using `mlflow models build-docker` with the registered model
2. For GGUF models, use a custom MLflow `pyfunc` flavor that wraps llama.cpp Python bindings
3. Push the container image to ECR in mx-central-1
4. Launch EC2 instances (c7g.4xlarge Graviton recommended) with the container
5. The container exposes the MLflow serving endpoint (REST API on port 5000)

**Custom MLflow Model Wrapper for llama.cpp:**

```python
import mlflow.pyfunc

class LlamaCppModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from llama_cpp import Llama
        model_path = context.artifacts["model_path"]
        self.llm = Llama(model_path=model_path, n_threads=16, n_ctx=4096)

    def predict(self, context, model_input):
        prompt = model_input["prompt"].iloc[0]
        output = self.llm(prompt, max_tokens=512)
        return output["choices"][0]["text"]
```

**EC2 Deployment Steps:**

1. Register the GGUF model as an MLflow artifact with the custom pyfunc wrapper
2. Build the Docker image: `mlflow models build-docker --model-uri models:/slm-compliance/Production --name slm-inference`
3. Push to ECR: `docker push {account}.dkr.ecr.mx-central-1.amazonaws.com/slm-inference:latest`
4. Launch EC2 with user-data script that pulls and runs the container
5. Configure Auto Scaling Group with target tracking on CPU utilization

### 13.5 Deployment to ECS

ECS deployment provides managed container orchestration with auto scaling:

**Approach: MLflow Container on ECS (Fargate or EC2 launch type)**

1. Build the MLflow Docker image as described in Section 13.4
2. Push to ECR in mx-central-1
3. Create an ECS Task Definition referencing the ECR image
4. For ARM64 (Graviton): use Fargate with ARM64 platform or EC2 launch type with c7g instances
5. For ECS Anywhere (hybrid): register on-premises nodes and deploy the same task definition

**ECS Task Definition Configuration:**

| Parameter | Value (Graviton) | Value (Intel) |
|-----------|------------------|---------------|
| CPU | 4096 (4 vCPU) | 4096 (4 vCPU) |
| Memory | 8192 MiB | 8192 MiB |
| Runtime platform | LINUX/ARM64 | LINUX/X86_64 |
| Container port | 5000 (MLflow serving) or 8080 (llama.cpp) | 5000 or 8080 |
| Health check | /health or /ping | /health or /ping |
| Environment variable: MODEL_PATH | /models/slm.gguf | /models/slm.gguf |

**ECS Service Configuration:**

| Parameter | Value |
|-----------|-------|
| Launch type | FARGATE (ARM64) or EXTERNAL (ECS Anywhere) |
| Desired count | 2 (production) or 1 (dev) |
| Deployment circuit breaker | Enabled with rollback |
| Auto scaling | Target tracking on ECSServiceAverageCPUUtilization at 70% |
| Load balancer | Application Load Balancer with /v1/completions path routing |

### 13.6 MLflow vs Direct Deployment

| Aspect | With MLflow | Without MLflow |
|--------|------------|----------------|
| Model versioning | Automatic, with registry stages | Manual S3 versioning |
| Rollback | Change alias to previous version | Re-deploy previous container tag |
| Experiment comparison | Built in UI and API | Custom dashboards required |
| Container building | `mlflow models build-docker` | Manual Dockerfile |
| Multi-model serving | Switch model URI per deployment | Separate containers per model |
| Audit trail | Full lineage from training to deployment | Manual tracking |

MLflow adds operational overhead (tracking server, RDS, ALB) but provides significant value for compliance workloads where model lineage and audit trails are required.

## 14. AWS Lambda Validation for SLM Inference

### 14.1 Feasibility Assessment

AWS Lambda supports running Small Language Models in container images. The following constraints and validated configurations apply:

**Hard Limits (as of 2026):**

| Constraint | Limit | Impact on SLM Inference |
|-----------|-------|------------------------|
| Container image size | 10 GB | Models up to ~5 GB fit with runtime overhead |
| Memory allocation | 10,240 MB (10 GB) | At max memory, 6 vCPUs are allocated |
| Execution timeout | 15 minutes | Limits total generation length per invocation |
| Ephemeral storage (/tmp) | 10 GB | Can store model locally if not baked into image |
| Payload size (sync) | 6 MB | Response must stream or fit in 6 MB |
| Payload size (async) | 256 KB | Async invocations have smaller input limits |
| Concurrency | 1000 default (adjustable) | Each concurrent execution loads its own model copy |

### 14.2 Validated Model Configurations

| Model | Format | File Size | Total Image Size | RAM Usage | Performance | Verdict |
|-------|--------|-----------|-----------------|-----------|-------------|---------|
| Llama 3.2 1B Q4_K_M | GGUF | ~0.7 GB | ~2.5 GB | ~2 GB | 15 to 25 tok/s | Fully supported |
| Qwen 3 0.6B | GGUF/PyTorch | ~1.2 GB | ~3 GB | ~2.5 GB | 20 to 35 tok/s | Fully supported |
| Gemma 3 1B Q4_K_M | GGUF | ~0.7 GB | ~2.5 GB | ~2 GB | 15 to 25 tok/s | Fully supported |
| Phi 4 Mini Q4_K_M | GGUF | ~2.3 GB | ~4 GB | ~4 GB | 10 to 15 tok/s | Supported (tight) |
| Llama 3.2 3B Q4_K_M | GGUF | ~2.0 GB | ~4 GB | ~5 GB | 8 to 15 tok/s | Supported (marginal KV cache room) |
| Llama 3.2 3B Q5_K_M | GGUF | ~2.4 GB | ~4.5 GB | ~6 GB | 6 to 12 tok/s | At limit (short context only) |
| Any model larger than 5GB | Any | >5 GB | >7 GB | >7 GB | Impractical | NOT recommended |

### 14.3 Lambda Architecture for SLM Inference

**Recommended Pattern (SnapStart + memfd):**

1. Build a container image with the GGUF model baked in (~2 to 4 GB total)
2. Use ARM64 architecture (Graviton Lambda) for 20 to 30% better performance
3. Set memory to 10,240 MB to unlock 6 vCPUs
4. Implement memfd_create pattern: stream model from S3 into anonymous RAM file during init
5. Enable SnapStart to snapshot the warm state (model loaded in RAM)
6. Subsequent invocations restore from snapshot in sub 500ms
7. Use Lambda Web Adapter for OpenAI compatible streaming API
8. Configure llama.cpp with n_threads=6

**When Lambda Works:**

1. Utilization below 15% (sporadic, bursty workloads)
2. Models 3B parameters or smaller with Q4 quantization
3. Context windows under 4096 tokens (RAM constrained)
4. Latency tolerance of 500ms to 2s for first token (with SnapStart)
5. Internal tools, document processing, batch compliance checks

**When Lambda Does NOT Work:**

1. Sustained high throughput (above 15% utilization, use EC2/ECS instead)
2. Models larger than 5 GB (memory pressure kills performance)
3. Long context windows (8K+ tokens exhaust available RAM for KV cache)
4. Real-time chat with strict latency SLA (cold starts still apply)
5. Stateful conversations (Lambda is stateless between invocations)

### 14.4 Lambda in mx-central-1

Lambda is available in mx-central-1 with both x86_64 and arm64 (Graviton) architectures. Container image deployment is supported. SnapStart availability should be confirmed at deployment time as feature rollout to newer regions may lag.

## 15. Amazon Bedrock AgentCore Assessment

### 15.1 Service Overview

Amazon Bedrock AgentCore is a managed platform for deploying and operating AI agents. It provides serverless container runtime (AgentCore Runtime), session isolation via microVMs, memory management, tool gateway, identity, and observability. It supports deploying agents built with any framework (LangGraph, Strands, CrewAI, custom) and any model.

AgentCore Runtime is NOT a model hosting service. It hosts agent code that calls models. The model itself can be hosted on Bedrock, SageMaker, or a self-managed endpoint. This distinction is important: you would deploy your SLM on EC2/ECS/Lambda and have AgentCore Runtime call it as a tool or model endpoint.

### 15.2 Region Availability

As of July 2026, Amazon Bedrock AgentCore is available in the following regions:

| Region | AgentCore Runtime | Memory | Gateway | Identity |
|--------|------------------|--------|---------|----------|
| us-east-1 (N. Virginia) | Yes | Yes | Yes | Yes |
| us-east-2 (Ohio) | Yes | Yes | Yes | Yes |
| us-west-2 (Oregon) | Yes | Yes | Yes | Yes |
| eu-central-1 (Frankfurt) | Yes | Yes | Yes | Yes |
| eu-west-1 (Ireland) | Yes | Yes | Yes | Yes |
| eu-west-2 (London) | Yes | Yes | Yes | Yes |
| eu-south-1 (Milan) | Yes | Yes | Yes | Yes |
| eu-west-3 (Paris) | Yes | Yes | Yes | Yes |
| eu-south-2 (Spain) | Yes | Yes | Yes | Yes |
| eu-north-1 (Stockholm) | Yes | Yes | Yes | Yes |
| ap-southeast-5 (Malaysia) | Yes | Yes | Yes | Yes |
| ap-south-1 (Mumbai) | Yes | Yes | Yes | Yes |
| ap-southeast-1 (Singapore) | Yes | Yes | Yes | Yes |
| ap-southeast-2 (Sydney) | Yes | Yes | Yes | Yes |
| ap-southeast-7 (Thailand) | Yes | Yes | Yes | Yes |
| ap-northeast-1 (Tokyo) | Yes | Yes | Yes | Yes |
| ap-northeast-2 (Seoul) | Yes | Yes | Yes | Yes |
| ca-central-1 (Canada) | Yes | Yes | Yes | Yes |
| sa-east-1 (Sao Paulo) | Yes | Yes | Yes | Yes |
| us-gov-west-1 (GovCloud) | Yes | Yes | Yes | Yes |

**mx-central-1 (Queretaro) is NOT in the supported regions list.**

### 15.3 Can the SLM Be Deployed to AgentCore in mx-central-1?

**Answer: No.** Bedrock AgentCore is not available in mx-central-1 as of July 2026.

**Alternatives for achieving AgentCore-like capabilities in mx-central-1:**

| Need | AgentCore Feature | Alternative in mx-central-1 |
|------|------------------|----------------------------|
| Agent hosting | AgentCore Runtime | ECS on Graviton with custom agent container |
| Session isolation | MicroVM per session | ECS task per session or in-process isolation |
| Memory | AgentCore Memory | DynamoDB or ElastiCache (Redis) for conversation state |
| Tool integration | AgentCore Gateway | API Gateway + Lambda for tool routing |
| Identity | AgentCore Identity | Cognito or custom OAuth implementation |
| Observability | AgentCore Observability | CloudWatch + X-Ray + OpenTelemetry |

### 15.4 Hybrid Architecture with AgentCore

A viable pattern uses AgentCore in us-west-2 as the agent orchestration layer while the SLM inference runs in mx-central-1:

1. Deploy the SLM (llama.cpp + GGUF) on EC2/ECS in mx-central-1 with an OpenAI compatible API
2. Deploy the agent logic on AgentCore Runtime in us-west-2
3. AgentCore calls the SLM endpoint in mx-central-1 as its model backend
4. Cross-region latency (~30 to 50ms between us-west-2 and mx-central-1) is acceptable for agent orchestration

**Trade-offs of this approach:**

| Advantage | Disadvantage |
|-----------|-------------|
| Full AgentCore managed services (memory, identity, observability) | Cross-region latency on every model call |
| Agent code benefits from microVM isolation | Data leaves mx-central-1 temporarily (prompts traverse to us-west-2) |
| No infrastructure management for agent layer | May violate data residency requirements if prompts contain sensitive data |
| Built-in session management and scaling | Additional networking cost for cross-region traffic |

**Recommendation:** If data residency requirements mandate that all inference and agent logic remain in mx-central-1, deploy the full agent stack (custom code + SLM) on ECS in mx-central-1 without AgentCore. If data residency allows prompts to traverse to us-west-2 for orchestration while model weights and outputs remain in mx-central-1, the hybrid pattern is viable.

### 15.5 AgentCore Runtime for Custom Models (General)

For regions where AgentCore IS available, it can host agents that use custom self-hosted SLMs:

1. AgentCore Runtime runs your agent container (any framework)
2. Your agent code calls a self-hosted SLM endpoint (EC2, ECS, Lambda) via HTTP
3. AgentCore handles session isolation, memory, identity, and observability
4. The SLM does not need to be a Bedrock-hosted model; any OpenAI-compatible endpoint works
5. Container images for AgentCore must be ARM64 (Graviton-based runtime)

This means your fine-tuned SLM can integrate with AgentCore in supported regions (like us-west-2) while the inference itself runs on your own infrastructure.

## 16. Post Deployment Model Evaluation in mx-central-1

### 16.1 Evaluation Strategy

Model evaluation does not end at the training phase. After deployment to the target region (mx-central-1), the SLM must be evaluated in its production environment to confirm that quantization, format conversion, and the target hardware produce acceptable results. This section covers the evaluation framework that runs against the deployed model endpoints.

**Evaluation Stages:**

| Stage | When | What is Measured | Tool |
|-------|------|-----------------|------|
| Deployment validation | Immediately after deployment | Model loads correctly, responds to prompts, no crashes | Automated health check |
| Quality evaluation | After deployment validation passes | Accuracy, correctness, adherence to guidelines | MLflow GenAI Evaluate |
| Performance benchmarking | After quality passes | Throughput (tok/s), latency (TTFT, ITL), concurrency limits | GuideLLM |
| Regression testing | On every model update | Compare new version against baseline metrics | MLflow + custom scorers |
| Production monitoring | Continuous | Drift detection, latency SLOs, error rates | CloudWatch + MLflow Tracing |

### 16.2 Quality Evaluation with MLflow

MLflow provides `mlflow.genai.evaluate()` for systematic quality assessment of deployed models. The evaluation runs against the live endpoint in mx-central-1.

**Evaluation Components:**

| Component | Implementation |
|-----------|---------------|
| Dataset | Curated test cases with inputs and expected outputs, stored in S3 (mx-central-1) |
| Predict function | HTTP client that calls the deployed model endpoint (OpenAI compatible API) |
| Scorers | Built-in (Correctness, Guidelines) + custom domain-specific scorers |
| Results | Logged to MLflow tracking server in us-west-2 for comparison across versions |

**Evaluation Dataset Structure:**

The evaluation dataset contains domain-specific test cases relevant to compliance workloads:

| Category | Example Count | Purpose |
|----------|--------------|---------|
| Regulatory question answering | 50 to 100 | Verify factual accuracy on Mexican regulatory content |
| Document summarization | 30 to 50 | Verify summary quality and completeness |
| Classification tasks | 50 to 100 | Verify correct categorization of compliance documents |
| Spanish language quality | 30 to 50 | Verify fluent and accurate Mexican Spanish generation |
| Tool use / function calling | 20 to 30 | Verify correct tool invocation format (if applicable) |
| Edge cases and safety | 20 to 30 | Verify appropriate handling of out-of-scope queries |

**Scorer Configuration:**

```python
import mlflow
from mlflow.genai.scorers import Correctness, Guidelines

# Custom compliance-specific scorer
@mlflow.genai.scorer
def regulatory_accuracy(expectations, outputs):
    """Check if the model output contains the expected regulatory reference."""
    expected_refs = expectations.get("regulatory_references", [])
    return all(ref in outputs for ref in expected_refs)

# Custom language quality scorer
@mlflow.genai.scorer
def spanish_quality(outputs):
    """Verify output is in proper Mexican Spanish."""
    # Uses an LLM judge to assess language quality
    return mlflow.genai.judges.is_correct(
        outputs,
        guidelines="The response must be in proper Mexican Spanish, using vocabulary and phrasing appropriate for Mexico."
    )

# Run evaluation against deployed endpoint
results = mlflow.genai.evaluate(
    data=evaluation_dataset,
    predict_fn=call_deployed_endpoint,
    scorers=[
        Correctness(),
        Guidelines(name="completeness", guidelines="The answer must address all parts of the question"),
        Guidelines(name="conciseness", guidelines="The answer should not exceed 500 words"),
        regulatory_accuracy,
        spanish_quality,
    ],
)
```

### 16.3 Performance Benchmarking

After quality evaluation passes, performance benchmarking validates that the deployed model meets latency and throughput SLOs on the target hardware.

**Tool: GuideLLM**

GuideLLM is an open source toolkit (from the vLLM project) for benchmarking LLM deployment performance by simulating real-world traffic patterns. It measures:

| Metric | Description | Target SLO |
|--------|-------------|-----------|
| TTFT (Time to First Token) | Latency before first token is generated | Less than 500ms |
| ITL (Inter-Token Latency) | Time between consecutive tokens | Less than 50ms |
| Throughput (tok/s) | Total tokens generated per second | Greater than 15 tok/s per instance |
| P50 latency | Median end-to-end response time | Less than 3 seconds |
| P99 latency | 99th percentile response time | Less than 10 seconds |
| Max concurrent requests | Maximum parallel requests before degradation | Greater than 4 |

**Benchmark Configuration:**

| Parameter | Value |
|-----------|-------|
| Target endpoint | http://model-endpoint:8080/v1/completions |
| Request distribution | Poisson (simulates real-world bursty traffic) |
| Prompt lengths | Mix of 128, 256, 512, and 1024 tokens |
| Output length | 256 tokens average |
| Concurrency levels | 1, 2, 4, 8 |
| Duration | 5 minutes per concurrency level |

**Expected Results by Hardware:**

| Instance | Model | Quantization | TTFT (P50) | Throughput | Max Concurrency |
|----------|-------|-------------|-----------|-----------|----------------|
| c7g.4xlarge (Graviton3) | Llama 3.2 3B | Q4_K_M | ~200ms | 35 to 50 tok/s | 4 to 6 |
| r8g.4xlarge (Graviton4) | Llama 3.2 3B | Q4_K_M | ~150ms | 40 to 60 tok/s | 6 to 8 |
| c7i.4xlarge (Intel) | Llama 3.2 3B | Q4_K_M | ~400ms | 10 to 20 tok/s | 2 to 4 |
| Lambda ARM64 (10GB) | Llama 3.2 1B | Q4_K_M | ~300ms (warm) | 15 to 25 tok/s | 1 |

### 16.4 Regression Testing Pipeline

Every model update (new fine-tune, re-quantization, version change) triggers an automated regression test:

**Pipeline Steps:**

1. Deploy new model version to a staging ECS service in mx-central-1 (separate from production)
2. Run the full evaluation dataset against the staging endpoint using `mlflow.genai.evaluate()`
3. Run GuideLLM performance benchmark against the staging endpoint
4. Compare results against the baseline (previous production version) stored in MLflow
5. Gate: Quality scores must be within 2% of baseline AND performance within 10% of baseline
6. If gate passes: promote to production (update ECS service, swap ALB target group)
7. If gate fails: alert the team, retain previous production version, log failure in MLflow

**Acceptance Criteria for Promotion:**

| Metric | Condition for Pass |
|--------|-------------------|
| Correctness score | Greater than or equal to baseline minus 2% |
| Regulatory accuracy | Greater than or equal to baseline (no regression allowed) |
| Spanish quality | Greater than or equal to baseline minus 1% |
| TTFT P50 | Less than or equal to baseline plus 10% |
| Throughput | Greater than or equal to baseline minus 10% |
| Error rate | Less than 1% |

### 16.5 Production Monitoring

After deployment to production, continuous monitoring tracks model behavior and detects degradation:

**Metrics Pipeline:**

| Source | Metric | Destination | Alert Threshold |
|--------|--------|-------------|-----------------|
| llama.cpp server | Tokens per second | CloudWatch custom metric | Below 10 tok/s sustained |
| llama.cpp server | Request latency (P99) | CloudWatch custom metric | Above 15 seconds |
| Application logs | Error rate | CloudWatch Logs metric filter | Above 2% |
| MLflow Tracing | Response quality (sampled) | MLflow tracking server | Below 80% correctness |
| Custom scorer (cron) | Drift detection (weekly) | MLflow + SNS alert | Score drops below 5% of baseline |

**MLflow Tracing Integration:**

MLflow Tracing captures every inference request in production (or a configurable sample):

1. Each request is traced with input prompt, output text, latency, and token counts
2. A background scorer evaluates a sample (1% to 10%) of traces for quality
3. Results are logged to the MLflow tracking server in us-west-2
4. Dashboards show quality trends over time
5. Alerts fire when quality metrics drop below configured thresholds

**CloudWatch Dashboard Panels:**

| Panel | Metric Source | Visualization |
|-------|-------------|---------------|
| Inference throughput | Custom metric (tok/s) | Time series line graph |
| Request latency distribution | Custom metric (P50, P95, P99) | Time series with bands |
| Error rate | CloudWatch Logs metric filter | Single value with threshold |
| Model version | ECS task definition tag | Text widget |
| Quality score trend | MLflow API (polled hourly) | Time series line graph |
| Memory and CPU utilization | ECS container insights | Stacked area chart |

### 16.6 Evaluation Automation Architecture

The complete evaluation automation runs as follows:

1. **Trigger**: New GGUF artifact lands in S3 (mx-central-1) via cross-region replication
2. **Step 1**: Lambda function detects new artifact, deploys to staging ECS service
3. **Step 2**: Lambda triggers evaluation Step Functions workflow
4. **Step 3**: Step Functions runs quality evaluation (MLflow evaluate against staging endpoint)
5. **Step 4**: Step Functions runs performance benchmark (GuideLLM against staging endpoint)
6. **Step 5**: Step Functions compares results against baseline in MLflow Model Registry
7. **Step 6**: If pass, update production ECS service with new task definition
8. **Step 7**: If fail, send SNS notification, retain production version
9. **Step 8**: Log all results to MLflow, update model registry with deployment metadata

**Infrastructure for Evaluation Pipeline:**

| Component | Service | Region | Purpose |
|-----------|---------|--------|---------|
| Trigger | S3 Event + Lambda | mx-central-1 | Detect new model artifacts |
| Orchestration | Step Functions | mx-central-1 | Coordinate evaluation steps |
| Quality eval runner | ECS Task (ephemeral) | mx-central-1 | Run mlflow.genai.evaluate() |
| Performance benchmark | ECS Task (ephemeral) | mx-central-1 | Run GuideLLM |
| Results storage | MLflow tracking server | us-west-2 | Store and compare metrics |
| Staging endpoint | ECS Service (staging) | mx-central-1 | Temporary model serving for eval |
| Production endpoint | ECS Service (production) | mx-central-1 | Live production serving |
| Notifications | SNS | mx-central-1 | Alert on failures |

## 17. Instance Selection: Training (us-west-2) and Inference (mx-central-1)

### 17.1 Training Instances in us-west-2

The following SageMaker instance types are available in us-west-2 and recommended for fine tuning SLMs (1B to 4B parameters) using QLoRA/LoRA:

**GPU Instances for Training:**

| Instance | GPU | GPU Memory | vCPU | RAM | Storage | Hourly Cost (On-Demand) | Best For |
|----------|-----|-----------|------|-----|---------|------------------------|----------|
| ml.g6.xlarge | 1x NVIDIA L4 | 24 GB | 4 | 16 GB | 250 GB NVMe | ~$1.13 | **Default**: single GPU LoRA/QLoRA for 1B to 3B models, best cost/hour |
| ml.g6.2xlarge | 1x NVIDIA L4 | 24 GB | 8 | 32 GB | 450 GB NVMe | ~$1.22 | More CPU/RAM headroom for larger datasets or longer context models |
| ml.g6.4xlarge | 1x NVIDIA L4 | 24 GB | 16 | 64 GB | 600 GB NVMe | ~$1.65 | Larger datasets, preprocessing in parallel |
| ml.g6.12xlarge | 4x NVIDIA L4 | 96 GB total | 48 | 192 GB | 3.8 TB NVMe | ~$5.75 | Multi-GPU training for 7B+ models |
| ml.g5.xlarge | 1x NVIDIA A10G | 24 GB | 4 | 16 GB | 250 GB NVMe | ~$1.41 | Previous generation, kept for compatibility |
| ml.g5.2xlarge | 1x NVIDIA A10G | 24 GB | 8 | 32 GB | 450 GB NVMe | ~$1.52 | Previous generation, kept for compatibility |
| ml.p3.2xlarge | 1x NVIDIA V100 | 16 GB | 8 | 61 GB | EBS only | ~$3.83 | Legacy, bf16 not supported |
| ml.p4d.24xlarge | 8x NVIDIA A100 | 320 GB total | 96 | 1152 GB | 8 TB NVMe | ~$37.69 | Full fine-tune of 7B+ models |
| ml.p5.48xlarge | 8x NVIDIA H100 | 640 GB total | 192 | 2048 GB | 30 TB NVMe | ~$98.32 | Pre-training or full fine-tune of 30B+ |

**Recommended Training Configuration for Target Models:**

The default recommendation is the G6 family (NVIDIA L4, 24 GB), not G5 (A10G, 24 GB): same GPU
memory, but 10 to 20 percent lower on demand cost per hour (verified via the AWS Price List
API for SageMaker Training in us-west-2, see pricing row above), and the G5-equivalent
instance sizes are already quota approved for training job usage in this account.

| Model | Method | Recommended Instance | Training Time (est.) | Cost (est.) | Notes |
|-------|--------|---------------------|---------------------|-------------|-------|
| Llama 3.2 1B | QLoRA (4-bit) | ml.g6.xlarge | 1 to 2 hours | $1.13 to $2.26 | Fits easily in 24GB L4 |
| Llama 3.2 3B | QLoRA (4-bit) | ml.g6.xlarge | 2 to 4 hours | $2.26 to $4.52 | 4-bit quantized base fits in 24GB |
| Phi 4 Mini (3.8B) | QLoRA (4-bit) | ml.g6.2xlarge | 2 to 4 hours | $2.44 to $4.88 | Extra RAM for 128K context handling |
| Qwen 2.5 3B | QLoRA (4-bit) | ml.g6.xlarge | 2 to 4 hours | $2.26 to $4.52 | Standard single GPU job |
| Llama 3.2 3B | Full fine-tune | ml.g6.12xlarge | 4 to 8 hours | $23 to $46 | Requires multi-GPU for full precision |

**Cost Optimization Strategies:**

1. **Managed Spot Training**: SageMaker supports spot instances for training, reducing cost by 60 to 90%. For a 4 hour ml.g6.xlarge job, spot pricing drops cost from ~$4.52 to ~$1.13 to $1.80.
2. **ml.g6 instances (default)**: The G6 family with NVIDIA L4 GPUs is the default recommendation over G5: similar or better performance, 10 to 20 percent lower cost per hour, and newer generation with longer expected availability.
3. **Right-sizing**: For QLoRA with 1B to 4B models, ml.g6.xlarge is sufficient. Over-provisioning (e.g., using ml.g6.12xlarge for a 3B QLoRA job) wastes money.
4. **Warm pools**: SageMaker warm pools keep instances ready for iterative training, eliminating 5 to 10 minute startup times on repeated jobs.

### 17.2 Inference Instances in mx-central-1 (Graviton ARM64)

Based on benchmarks from ClearML (January 2025) and Arcee AI (July 2025) running llama.cpp with GGUF quantized models:

**Graviton Benchmark Data (real measurements):**

| Instance | Processor | vCPU | RAM | Model | Quantization | Tokens/s | $/Hour | Tokens per Dollar |
|----------|-----------|------|-----|-------|-------------|----------|--------|-------------------|
| c8g.2xlarge | Graviton4 | 8 | 16 GB | Llama 3 8B | Q4_0 | 14.41 | $0.318 | 163,173 |
| c8g.4xlarge | Graviton4 | 16 | 32 GB | Llama 3 8B | Q4_0 | ~25 (est.) | $0.636 | ~141,500 |
| c8g.8xlarge | Graviton4 | 32 | 64 GB | Llama 3 8B | Q4_0 | ~40 (est.) | $1.272 | ~113,200 |
| r8g.4xlarge | Graviton4 | 16 | 128 GB | AFM 4.5B | Q4_0 | 44.5 | $0.943 | 169,883 |
| r8g.4xlarge | Graviton4 | 16 | 128 GB | AFM 4.5B | Q8_0 | 28.5 | $0.943 | 108,802 |
| r8g.4xlarge | Graviton4 | 16 | 128 GB | QWEN 32B | Q4_0 | 6.09 | $0.943 | 23,260 |
| c7g.2xlarge | Graviton3 | 8 | 16 GB | Llama 3 8B | Q4_0 | ~8 (est.) | $0.152 | ~189,474 |
| c7g.4xlarge | Graviton3 | 16 | 32 GB | Llama 3 8B | Q4_0 | ~14 (est.) | $0.305 | ~165,246 |

**Inference Instances Comparison (Intel x86 in mx-central-1):**

| Instance | Processor | vCPU | RAM | Model | Quantization | Tokens/s | $/Hour | Tokens per Dollar |
|----------|-----------|------|-----|-------|-------------|----------|--------|-------------------|
| c7i.2xlarge | Intel Sapphire Rapids | 8 | 16 GB | Llama 3 8B | Q4_0 | 4.48 | $0.357 | 45,176 |
| c7i.4xlarge | Intel Sapphire Rapids | 16 | 32 GB | Llama 3 8B | Q4_0 | ~8 (est.) | $0.714 | ~40,336 |
| c6i.4xlarge | Intel Ice Lake | 16 | 32 GB | Llama 3 8B | Q4_0 | ~7 (est.) | $0.714 | ~35,294 |

### 17.3 Cost-Performance Analysis for 3B SLM Inference

For the target workload (Llama 3.2 3B or Phi 4 Mini at Q4_K_M), the 3B model requires approximately 2GB RAM for weights and 2 to 4GB for KV cache, totaling 4 to 6 GB under load. The following instances are evaluated for best cost-performance:

**Recommended Instances (ranked by tokens per dollar):**

| Rank | Instance | Architecture | vCPU | RAM | Est. tok/s (3B Q4_K_M) | $/Hour (On-Demand) | Tokens per Dollar | Verdict |
|------|----------|-------------|------|-----|------------------------|-------------------|-------------------|---------|
| 1 | c8g.2xlarge | Graviton4 | 8 | 16 GB | 25 to 35 | $0.318 | ~340,000 | Best cost-performance for single user |
| 2 | c7g.2xlarge | Graviton3 | 8 | 16 GB | 18 to 25 | $0.152 | ~470,000 | Cheapest option, good for dev/test |
| 3 | c8g.4xlarge | Graviton4 | 16 | 32 GB | 40 to 55 | $0.636 | ~250,000 | Production with batching |
| 4 | r8g.2xlarge | Graviton4 | 8 | 64 GB | 25 to 35 | $0.495 | ~220,000 | When large KV cache is needed (long context) |
| 5 | c7g.4xlarge | Graviton3 | 16 | 32 GB | 28 to 38 | $0.305 | ~370,000 | Best value for multi-user production |
| 6 | c7i.4xlarge | Intel Sapphire Rapids | 16 | 32 GB | 10 to 18 | $0.749 | ~68,000 | Only if x86 is mandated |

**Key Finding**: Graviton instances deliver 3x to 5x better tokens per dollar compared to Intel for GGUF LLM inference. The c7g family offers the absolute best cost-performance ratio, while c8g/r8g deliver higher absolute throughput.

### 17.4 Instance Selection Decision Guide

**For Single User / Low Latency (interactive compliance tool):**

Choose: **c8g.2xlarge** (Graviton4, 8 vCPU, 16 GB)
Rationale: Delivers 25 to 35 tok/s for a 3B Q4_K_M model, which exceeds the 10 tok/s minimum for interactive use. At $0.318/hr, cost is minimal. The 16 GB RAM accommodates the model (2GB) plus KV cache (4 to 6 GB) with headroom.

**For Production Multi-User (2 to 4 concurrent requests):**

Choose: **c7g.4xlarge** (Graviton3, 16 vCPU, 32 GB) or **c8g.4xlarge** (Graviton4, 16 vCPU, 32 GB)
Rationale: 16 vCPUs enable parallel batch processing. The c7g.4xlarge at $0.305/hr is the most cost effective for sustained production. The c8g.4xlarge at $0.636/hr provides higher peak throughput. Deploy 2+ instances behind an ALB for high availability.

**For Long Context Windows (8K+ tokens):**

Choose: **r8g.2xlarge** (Graviton4, 8 vCPU, 64 GB)
Rationale: The memory optimized instance provides 64 GB RAM, allowing large KV caches for extended context. Essential when processing long regulatory documents that require full context in a single pass.

**For Budget / Dev / Test:**

Choose: **c7g.2xlarge** (Graviton3, 8 vCPU, 16 GB) or **t4g.xlarge** (Graviton2, 4 vCPU, 16 GB)
Rationale: At $0.152/hr (c7g) or $0.141/hr (t4g), these provide acceptable performance (15 to 25 tok/s for 3B Q4_K_M) at the lowest cost. The t4g has burstable CPU and lower baseline but works for low traffic testing.

**For ECS Fargate (ARM64):**

Choose: **4 vCPU / 8 GB** or **8 vCPU / 16 GB** Fargate task on ARM64 platform
Rationale: Fargate removes EC2 management overhead. ARM64 Fargate tasks run on Graviton2 hardware. Performance is comparable to t4g/c6g instances. Use when operational simplicity is more important than maximizing tokens per dollar.

### 17.5 Spot Instance Strategy for Inference

Graviton Spot instances reduce costs by 60 to 75% for fault-tolerant inference workloads:

| Instance | On-Demand $/hr | Spot $/hr (typical) | Savings | Use Case |
|----------|---------------|--------------------|---------|---------| 
| c8g.2xlarge | $0.318 | ~$0.095 | 70% | Batch processing, async inference |
| c7g.4xlarge | $0.305 | ~$0.091 | 70% | Development, non-critical workloads |
| r8g.4xlarge | $0.943 | ~$0.283 | 70% | Large model or long context batch jobs |

Spot instances are viable for:
1. Batch document processing where interruption is tolerable
2. Development and testing environments
3. Auto scaling groups with mixed on-demand and spot (e.g., 2 on-demand base + spot scaling)
4. Asynchronous inference queues (SQS + spot instances)

Spot instances are NOT recommended for:
1. Real-time interactive inference with strict SLAs
2. Single-instance deployments with no redundancy

### 17.6 Summary: Recommended Configurations

**Training (us-west-2):**

| Scenario | Instance | Cost/Job (est.) | Notes |
|----------|----------|-----------------|-------|
| QLoRA 3B model (default) | ml.g6.xlarge | $2.26 to $4.52 | Best value, NVIDIA L4 24GB sufficient |
| QLoRA 3B with spot | ml.g6.xlarge (spot) | $0.57 to $1.13 | 60-90% savings, add checkpointing |
| QLoRA 4B model (Phi 4 Mini) | ml.g6.2xlarge | $2.44 to $4.88 | Extra RAM for larger context |
| Legacy GPU (compatibility) | ml.g5.xlarge | $2.80 to $5.60 | A10G, 10-20% more expensive than G6 |
| Full fine-tune 3B | ml.g6.12xlarge | $23 to $46 | Multi-GPU, only if QLoRA quality insufficient |

**Inference (mx-central-1):**

| Scenario | Instance | Cost/hr | Expected tok/s | Monthly Cost (24/7) |
|----------|----------|---------|---------------|-------------------|
| Production (recommended) | c8g.2xlarge | $0.318 | 25 to 35 | ~$232 |
| Production (high throughput) | c7g.4xlarge | $0.305 | 28 to 38 | ~$223 |
| Production (max performance) | c8g.4xlarge | $0.636 | 40 to 55 | ~$464 |
| Budget / dev | c7g.2xlarge | $0.152 | 18 to 25 | ~$111 |
| Long context | r8g.2xlarge | $0.495 | 25 to 35 | ~$361 |
| Intel fallback | c7i.4xlarge | $0.749 | 10 to 18 | ~$547 |

## 18. LLM and Agent Performance Monitoring Dashboard

### 18.1 Dashboard Architecture

The monitoring solution combines CloudWatch for real-time operational metrics with QuickSight for historical cost analysis and executive reporting. Both dashboards consume data from the same metric pipeline but serve different audiences and time horizons.

**Architecture Overview:**

| Layer | Service | Purpose | Audience | Refresh Rate |
|-------|---------|---------|----------|-------------|
| Metric collection | CloudWatch Custom Metrics + Embedded Metric Format | Real-time ingestion of token counts, latency, costs | System | Per request |
| Operational dashboard | CloudWatch Dashboards | Live monitoring, alerting, incident response | Engineers, SRE | Real-time (1 min) |
| Analytical dashboard | Amazon QuickSight | Cost trends, model comparison, capacity planning | Management, FinOps | Hourly/Daily |
| Data lake | S3 + Athena | Raw metric storage for ad-hoc queries | Data analysts | Batch (hourly) |

### 18.2 Metric Collection Pipeline

Every inference request produces structured metrics that flow into both dashboards:

**Metrics Emitted Per Request:**

| Metric | Type | Unit | Dimensions | Description |
|--------|------|------|-----------|-------------|
| InputTokens | Count | Tokens | ModelName, AgentName, Environment | Number of tokens in the prompt |
| OutputTokens | Count | Tokens | ModelName, AgentName, Environment | Number of tokens generated |
| TotalTokens | Count | Tokens | ModelName, AgentName, Environment | Sum of input and output tokens |
| InferenceCost | Value | USD | ModelName, AgentName, Environment | Computed cost for this request |
| TimeToFirstToken | Timer | Milliseconds | ModelName, AgentName, InstanceType | Latency before first token |
| TokenGenerationRate | Gauge | Tokens/Second | ModelName, AgentName, InstanceType | Generation throughput |
| RequestLatency | Timer | Milliseconds | ModelName, AgentName, Environment | End-to-end request duration |
| RequestCount | Count | Count | ModelName, AgentName, Environment, StatusCode | Total requests (success/error) |
| ContextWindowUsage | Gauge | Percent | ModelName, AgentName | KV cache utilization |

**Dimensions for Cost Attribution:**

| Dimension | Values (examples) | Purpose |
|-----------|-------------------|---------|
| ModelName | llama-3.2-3b-q4, phi-4-mini-q4, qwen-0.6b | Cost per model |
| AgentName | compliance-qa, doc-summarizer, classifier | Cost per agent/use case |
| Environment | production, staging, development | Cost per environment |
| InstanceType | c8g.2xlarge, c7g.4xlarge, lambda-arm64 | Cost per infrastructure tier |
| Team | compliance-team, legal-team, ops-team | Cost per consuming team |

### 18.3 Cost Calculation for Self-Hosted Models

Unlike managed API services (Bedrock, OpenAI) which charge per token, self-hosted models require computing cost per token from infrastructure costs:

**Cost Per Token Formula:**

```
Cost per token = (Instance hourly cost / Tokens generated per hour)
```

**Example Calculations:**

| Instance | $/Hour | Tokens/Second | Tokens/Hour | Cost per 1K Tokens | Cost per 1M Tokens |
|----------|--------|--------------|-------------|--------------------|--------------------|
| c8g.2xlarge | $0.318 | 30 | 108,000 | $0.00294 | $2.94 |
| c7g.4xlarge | $0.305 | 35 | 126,000 | $0.00242 | $2.42 |
| r8g.4xlarge | $0.943 | 45 | 162,000 | $0.00582 | $5.82 |
| Lambda ARM64 (10GB) | ~$0.60/hr equiv | 20 | 72,000 | $0.00833 | $8.33 |
| c7i.4xlarge (Intel) | $0.749 | 14 | 50,400 | $0.01486 | $14.86 |

**Note:** These costs represent pure compute. Total cost of ownership includes S3 storage, data transfer, ALB, CloudWatch, and operational overhead.

**Cost Attribution Logic (implemented in the inference wrapper):**

```python
import boto3
import time

cloudwatch = boto3.client("cloudwatch", region_name="mx-central-1")

INSTANCE_COST_PER_SECOND = {
    "c8g.2xlarge": 0.318 / 3600,
    "c7g.4xlarge": 0.305 / 3600,
    "r8g.4xlarge": 0.943 / 3600,
}

def emit_inference_metrics(
    model_name: str,
    agent_name: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    instance_type: str,
    environment: str = "production",
):
    total_tokens = input_tokens + output_tokens
    duration_seconds = latency_ms / 1000.0
    cost_per_second = INSTANCE_COST_PER_SECOND.get(instance_type, 0)
    inference_cost = cost_per_second * duration_seconds

    dimensions = [
        {"Name": "ModelName", "Value": model_name},
        {"Name": "AgentName", "Value": agent_name},
        {"Name": "Environment", "Value": environment},
        {"Name": "InstanceType", "Value": instance_type},
    ]

    cloudwatch.put_metric_data(
        Namespace="SLM/Inference",
        MetricData=[
            {"MetricName": "TotalTokens", "Value": total_tokens, "Unit": "Count", "Dimensions": dimensions},
            {"MetricName": "InferenceCost", "Value": inference_cost, "Unit": "None", "Dimensions": dimensions},
            {"MetricName": "RequestLatency", "Value": latency_ms, "Unit": "Milliseconds", "Dimensions": dimensions},
            {"MetricName": "OutputTokens", "Value": output_tokens, "Unit": "Count", "Dimensions": dimensions},
            {"MetricName": "TokenGenerationRate", "Value": output_tokens / duration_seconds, "Unit": "Count/Second", "Dimensions": dimensions},
        ],
    )
```

### 18.4 CloudWatch Dashboard (Operational)

The CloudWatch dashboard provides real-time visibility for engineering and SRE teams. It is deployed via CloudFormation as part of the monitoring stack.

**Dashboard Layout (6 rows):**

| Row | Panel 1 | Panel 2 | Panel 3 |
|-----|---------|---------|---------|
| 1 | Total Requests (line, by agent) | Error Rate (line, threshold at 2%) | Active Instances (number) |
| 2 | Token Generation Rate (line, by model) | Time to First Token P50/P95/P99 (line) | Request Latency P50/P95/P99 (line) |
| 3 | Cost per Hour (line, by model) | Cost per Hour (line, by agent) | Cumulative Daily Cost (number) |
| 4 | Tokens per Second per Instance (line) | CPU Utilization (line, by instance) | Memory Utilization (line, by instance) |
| 5 | Cost per 1K Tokens (line, trend) | Tokens Processed (bar, by agent) | Context Window Usage % (gauge) |
| 6 | Model Version (text) | Deployment Status (text) | Alerts Summary (alarm status) |

**CloudWatch Alarms:**

| Alarm | Metric | Threshold | Action |
|-------|--------|-----------|--------|
| High Latency | RequestLatency P99 | Above 10,000 ms for 3 periods | SNS notification |
| High Error Rate | RequestCount (5xx) / RequestCount (total) | Above 2% for 2 periods | SNS notification + PagerDuty |
| Low Throughput | TokenGenerationRate | Below 10 tok/s for 5 periods | SNS notification |
| Cost Spike | InferenceCost (sum, 1 hour) | Above 150% of daily average | SNS notification to FinOps |
| Instance Unhealthy | ECS HealthCheckStatus | Any unhealthy for 2 periods | Auto-replace task |

### 18.5 QuickSight Dashboard (Analytical / FinOps)

The QuickSight dashboard provides historical cost analysis, trend visualization, and executive reporting. Data flows from CloudWatch to S3 (via Metric Stream or scheduled export), then into QuickSight via Athena.

**Data Pipeline:**

1. CloudWatch Metric Stream exports metrics to S3 in Parquet format (hourly partitions)
2. AWS Glue Crawler catalogs the metric data into an Athena table
3. QuickSight connects to Athena as a data source
4. SPICE (in-memory engine) refreshes hourly for fast dashboard loading

**QuickSight Dashboard Sheets:**

**Sheet 1: Cost Overview**

| Visual | Type | Data | Filters |
|--------|------|------|---------|
| Monthly cost trend | Line chart | InferenceCost aggregated daily | Date range, Environment |
| Cost by model | Pie chart | InferenceCost grouped by ModelName | Date range |
| Cost by agent | Bar chart (horizontal) | InferenceCost grouped by AgentName | Date range, Model |
| Cost by team | Stacked bar | InferenceCost grouped by Team | Date range |
| Cost per 1K tokens trend | Line chart | InferenceCost / (TotalTokens / 1000) daily | Date range, Model |
| Projected monthly cost | KPI | Linear projection from current month spend | Current month |

**Sheet 2: Model Performance**

| Visual | Type | Data | Filters |
|--------|------|------|---------|
| Tokens per second by model | Line chart (multi-series) | TokenGenerationRate averaged hourly | Date range, Instance |
| Latency distribution | Histogram | RequestLatency | Model, Date range |
| TTFT trend by model | Line chart | TimeToFirstToken P50 daily | Date range |
| Quality score trend | Line chart | MLflow quality metric (imported) | Date range, Model |
| Model version timeline | Gantt/timeline | Deployment events | Date range |

**Sheet 3: Agent Analytics**

| Visual | Type | Data | Filters |
|--------|------|------|---------|
| Requests by agent | Stacked area | RequestCount by AgentName hourly | Date range |
| Cost per agent per day | Heat map | InferenceCost by AgentName by Day | Date range |
| Average tokens per request by agent | Bar chart | TotalTokens / RequestCount by Agent | Date range |
| Error rate by agent | Line chart | Error requests / Total requests | Date range |
| Top 10 most expensive requests | Table | Individual requests sorted by cost | Date range, Agent |

**Sheet 4: Infrastructure Efficiency**

| Visual | Type | Data | Filters |
|--------|------|------|---------|
| Tokens per dollar by instance | Bar chart | TotalTokens / InferenceCost by InstanceType | Date range |
| Utilization heat map | Heat map | CPU utilization by instance by hour | Date range |
| Spot vs on-demand cost | Stacked bar | InferenceCost grouped by pricing model | Date range |
| Idle time (below 5% CPU) | KPI | Hours where CPU below 5% | Date range, Instance |
| Right-sizing recommendations | Table | Instances where avg CPU below 30% | Last 7 days |

### 18.6 Cost Attribution Model

The cost attribution model assigns infrastructure costs to specific models, agents, and teams based on actual token consumption:

**Attribution Levels:**

| Level | Granularity | Method | Example |
|-------|------------|--------|---------|
| Per request | Individual inference call | Direct measurement (duration x instance cost/second) | Request #12345 cost $0.0003 |
| Per agent | Aggregate per agent/use case | Sum of all request costs for that agent | compliance-qa agent: $45.20/day |
| Per model | Aggregate per model variant | Sum of all request costs for that model | llama-3.2-3b-q4: $38.00/day |
| Per team | Aggregate per consuming team | Sum of all request costs tagged to that team | legal-team: $120.50/day |
| Per environment | Production vs staging vs dev | Sum by environment dimension | production: $180/day, staging: $22/day |

**Tagging Strategy:**

Each inference request must carry the following tags (passed as HTTP headers or query parameters to the model endpoint):

| Tag | Source | Required |
|-----|--------|----------|
| X-Agent-Name | Application code | Yes |
| X-Team-Name | Authentication/IAM | Yes |
| X-Environment | Infrastructure config | Yes |
| X-Request-Id | Auto-generated (UUID) | Yes |
| X-Session-Id | Client session | Optional |

The inference wrapper (Section 18.3 code) reads these tags and includes them as CloudWatch metric dimensions.

### 18.7 CloudWatch vs QuickSight: When to Use Each

| Criteria | CloudWatch Dashboard | QuickSight Dashboard |
|----------|---------------------|---------------------|
| Refresh rate | Real-time (1 minute) | Hourly (SPICE refresh) |
| Primary audience | Engineers, SRE, on-call | Management, FinOps, Product |
| Use case | Incident response, live monitoring | Cost analysis, trend reporting, capacity planning |
| Alerting | Native CloudWatch Alarms | Not applicable (reporting only) |
| Data retention | 15 months (standard metrics) | Unlimited (S3 backed) |
| Cost | Included with CloudWatch ($3/dashboard/month) | $3 to $40/user/month depending on role |
| Customization | Widget based, JSON definition | Drag-and-drop visual builder, Q natural language |
| Sharing | AWS Console access required | Embedded dashboards, email reports, public URLs |
| AI features | Anomaly detection | Amazon Q for natural language queries |

**Recommendation:** Deploy both dashboards. CloudWatch is the primary operational tool (always-on, alerting, real-time). QuickSight is the analytical layer for cost governance, executive reporting, and capacity planning.

### 18.8 Implementation Plan

**Phase 1: CloudWatch Metrics + Dashboard (Week 1)**

1. Implement the metric emission code in the inference wrapper (Section 18.3)
2. Create the CloudWatch namespace `SLM/Inference` with all metric definitions
3. Deploy the CloudWatch dashboard via CloudFormation (monitoring stack)
4. Configure alarms for latency, error rate, throughput, and cost spikes
5. Validate metrics flow with test requests

**Phase 2: S3 Export + Athena (Week 2)**

1. Create a CloudWatch Metric Stream targeting S3 (Parquet, hourly partitions)
2. Configure S3 bucket with lifecycle policy (transition to IA after 90 days, Glacier after 1 year)
3. Create AWS Glue Crawler to catalog the metric data
4. Validate Athena queries against the partitioned data
5. Create named Athena queries for common cost attribution reports

**Phase 3: QuickSight Dashboard (Week 3)**

1. Create QuickSight data source connected to Athena
2. Import data into SPICE with hourly refresh schedule
3. Build the four dashboard sheets (Cost Overview, Model Performance, Agent Analytics, Infrastructure)
4. Configure row-level security so teams only see their own cost data
5. Set up scheduled email reports for weekly cost summaries to FinOps team
6. Enable Amazon Q in QuickSight for natural language cost queries

**Phase 4: Continuous Refinement (Ongoing)**

1. Add custom metrics as new agents or models are deployed
2. Refine cost-per-token calculations based on actual utilization patterns
3. Create QuickSight alerts for budget thresholds
4. Build embedded dashboard views for internal compliance portal

## 19. Security and Data Residency

### 19.1 Data Residency Requirements

The architecture must comply with Mexican data privacy regulations (Ley Federal de Protección de Datos Personales en Posesión de los Particulares, LFPDPPP) and any applicable sector-specific regulations for compliance workloads. The following principles govern data residency:

| Data Type | Storage Location | Processing Location | May Leave Mexico? |
|-----------|-----------------|---------------------|------------------|
| Inference prompts (user input) | mx-central-1 only | mx-central-1 (local inference) | No (default) |
| Model weights (GGUF files) | us-west-2 (training) + mx-central-1 (inference) | Both regions | Yes (model artifacts are not PII) |
| Inference outputs | mx-central-1 | mx-central-1 | No (default) |
| Training datasets | us-west-2 only | us-west-2 only | N/A (not PII if anonymized) |
| Monitoring metrics (tokens, latency, cost) | mx-central-1 + us-west-2 (aggregate only) | Both regions | Yes (aggregate metrics, no PII) |
| MLflow traces (prompt metadata) | us-west-2 (sampled) | us-west-2 | Only if MLflow sampling is enabled |

**Critical constraint:** The AgentCore hybrid pattern (Section 15.4) where prompts traverse to us-west-2 for orchestration is NOT recommended if prompts contain personally identifiable information or regulatory document content. In that scenario, all inference and agent logic must remain within mx-central-1.

### 19.2 Encryption at Rest

All data stores use server-side encryption:

| Service | Encryption Method | Key Management |
|---------|------------------|---------------|
| S3 (model artifacts, us-west-2) | SSE-KMS with customer-managed key | AWS KMS in us-west-2 |
| S3 (deployment bucket, mx-central-1) | SSE-KMS with customer-managed key | AWS KMS in mx-central-1 |
| S3 (metrics data, mx-central-1) | SSE-S3 (AES-256) | AWS managed key |
| RDS PostgreSQL (MLflow metadata) | Encryption at rest enabled | AWS KMS in us-west-2 |
| EBS volumes (EC2 inference instances) | EBS encryption enabled | AWS KMS in mx-central-1 |
| ECR container images | Encrypted at rest | AWS managed key |

KMS key rotation must be enabled for all customer-managed keys. Keys in us-west-2 and mx-central-1 are independent and not cross-region shared.

### 19.3 Encryption in Transit

All communication uses TLS 1.2 or higher:

| Communication Path | Protocol | Certificate |
|-------------------|----------|-------------|
| Client to ALB (inference endpoint) | HTTPS (TLS 1.2+) | ACM certificate (mx-central-1) |
| ALB to ECS containers | HTTP (internal VPC) | VPC security groups restrict access |
| Lambda Function URL | HTTPS (TLS 1.2+) | Managed by AWS |
| ECS to S3 (model loading) | HTTPS via S3 endpoint | VPC endpoint (no public internet) |
| us-west-2 to mx-central-1 (replication) | HTTPS (AWS internal) | AWS managed |
| MLflow UI access | HTTPS (TLS 1.2+) | ACM certificate (us-west-2) |
| SageMaker training to S3 | HTTPS via VPC endpoint | VPC endpoint (no public internet) |

### 19.4 Network Security

**Training Region (us-west-2):**

1. SageMaker training jobs run in private subnets with no internet access; all AWS service access is via VPC Endpoints
2. MLflow server is accessible only from within the VPC and via the ALB (no direct port exposure)
3. RDS PostgreSQL is in private subnets accessible only to the MLflow ECS security group
4. S3 bucket policies restrict access to specific IAM role ARNs; public access is blocked

**Deployment Region (mx-central-1):**

1. ECS inference containers run in private subnets; the ALB in public subnets accepts inbound HTTPS
2. Security groups restrict ECS containers to inbound port 8080 from the ALB security group only
3. Lambda functions execute within the VPC in private subnets; outbound traffic to S3 via VPC endpoint
4. No EC2 instances are accessible via public IP; all management access requires SSM Session Manager or bastion host

### 19.5 Identity and Access Management

**Principle of least privilege:** All IAM roles use resource-level ARN restrictions, not wildcard resources.

**Key IAM boundaries:**

| Principal | Allowed Actions | Resource Scope |
|-----------|----------------|---------------|
| SageMaker training job | S3 GetObject/PutObject, CloudWatch PutMetricData, ECR GetAuthorizationToken | Specific model bucket prefix |
| ECS inference task | S3 GetObject | Specific deployment bucket prefix |
| Lambda inference function | S3 GetObject | Specific deployment bucket prefix |
| Lambda optimization function | S3 GetObject/PutObject, CloudWatch | Source bucket trained model prefix and destination optimized prefix |
| S3 replication service | S3 GetObject, ReplicateObject | Source bucket to destination bucket |

**Secrets management:** Database credentials (RDS PostgreSQL for MLflow) are stored in AWS Secrets Manager and injected into ECS containers as environment variables at runtime. No credentials are hardcoded in CloudFormation templates or container images.

### 19.6 Container Security

1. All container images are scanned on push to ECR (scan on push enabled in Requirement 6)
2. Container images must not run as root; non-root user is specified in the Dockerfile
3. ECR lifecycle policies remove untagged images to reduce attack surface
4. Lambda container images are immutable once deployed; updates require new image versions
5. All base images should use the latest Amazon Linux 2023 or Ubuntu 22.04 LTS to minimize CVE exposure

### 19.7 Audit and Compliance Trail

| Audit Event | Source | Destination | Retention |
|-------------|--------|-------------|-----------|
| API calls to AWS services | AWS CloudTrail | S3 (both regions) | 1 year |
| Model deployments | MLflow Model Registry | MLflow RDS | Indefinite |
| Evaluation pass/fail | Step Functions execution history | CloudWatch Logs + MLflow | 90 days |
| Inference requests (sampled) | MLflow Tracing | MLflow RDS | 90 days |
| IAM role assumption | AWS CloudTrail | S3 (both regions) | 1 year |
| S3 object access | S3 Server Access Logging | S3 audit bucket | 1 year |

CloudTrail must be enabled in both us-west-2 and mx-central-1 with a dedicated S3 bucket for audit logs. Log integrity validation (SHA-256 digest files) must be enabled.

---

## 20. Cost Estimate Summary

### 20.1 Architecture Cost Components

The total monthly cost of the architecture covers three categories: training (episodic, us-west-2), inference infrastructure (continuous, mx-central-1), and supporting services (continuous, both regions).

### 20.2 Training Cost (us-west-2, per fine-tuning run)

Training is an episodic cost triggered when a new model version is needed, not a continuous monthly expense.

| Component | Configuration | Cost per Run |
|-----------|--------------|-------------|
| SageMaker training job | ml.g6.xlarge, QLoRA 3B, 2 to 4 hours | $2.26 to $4.52 |
| SageMaker training job (with spot) | ml.g6.xlarge spot, 60 to 90% savings | $0.57 to $1.13 |
| SageMaker evaluation pipeline | ml.t3.medium (processing), ~30 minutes | $0.03 |
| Optimization Lambda | 512MB, ~10 minutes per run | $0.002 |
| S3 storage (model artifacts) | 10 GB model files per version | $0.23/month |
| Training data transfer | S3 to SageMaker (within region) | $0.00 (no charge) |
| **Total per training cycle** | Using spot instances | **$0.60 to $1.16** |
| **Total per training cycle** | On-demand instances | **$2.29 to $4.55** |

Assuming 2 fine-tuning cycles per month, training costs $1.50 to $11.30/month using a mix of spot and on-demand.

### 20.3 Inference Infrastructure Cost (mx-central-1, monthly 24/7)

**Primary path: EC2 Graviton + ECS (recommended for sustained load)**

| Component | Configuration | Monthly Cost |
|-----------|--------------|-------------|
| EC2 c8g.2xlarge (Graviton4) | 2 instances, on-demand, 24/7 | $463 |
| EC2 c7g.4xlarge (Graviton3) alternative | 2 instances, on-demand, 24/7 | $446 |
| ALB | 1 ALB, ~1000 LCU/month | $22 |
| EBS storage | 2x 50 GB gp3 volumes | $8 |
| S3 (deployment bucket) | 10 GB, GET requests for model loading | $0.23 |
| ECR (container images) | 5 GB storage | $0.50 |
| **Total (ECS on c8g.2xlarge, 2 instances)** | Production HA configuration | **~$493/month** |

**Secondary path: Lambda ARM64 (for sporadic workloads)**

| Component | Configuration | Monthly Cost |
|-----------|--------------|-------------|
| Lambda (Llama 3.2 1B Q4_K_M) | 10GB RAM, 1000 req/day, 30s avg | ~$45 |
| Lambda (Llama 3.2 3B Q4_K_M) | 10GB RAM, 500 req/day, 60s avg | ~$43 |
| Function URL / API Gateway | 1M requests | $1 |
| **Total (Lambda, moderate usage)** | Below 15% utilization | **~$46 to $46/month** |

Lambda becomes more expensive than 2x EC2 at approximately 800+ requests per day (15% utilization). Below that threshold, Lambda is significantly cheaper due to zero idle cost.

### 20.4 Supporting Services Cost (both regions, monthly)

| Component | Region | Configuration | Monthly Cost |
|-----------|--------|--------------|-------------|
| MLflow ECS Fargate | us-west-2 | 0.5 vCPU, 1 GB, 24/7 | $14 |
| RDS PostgreSQL | us-west-2 | db.t3.micro, 20 GB | $14 |
| CloudWatch custom metrics | mx-central-1 | SLM/Inference namespace, 9 metrics | $27 |
| CloudWatch dashboards | both | 2 dashboards | $6 |
| CloudWatch Metric Stream | mx-central-1 | ~100K metric updates/day | $8 |
| AWS Glue Crawler | mx-central-1 | 1 DPU, hourly schedule | $10 |
| Athena | mx-central-1 | ~10 GB scanned/month | $0.50 |
| S3 metrics storage | mx-central-1 | 30 GB Parquet/month | $0.69 |
| SNS | both | ~1000 notifications/month | $0.001 |
| Step Functions | mx-central-1 | ~60 executions/month | $0.006 |
| KMS key usage | both | ~10K API calls/month | $1 |
| **Total supporting services** | | | **~$81/month** |

### 20.5 Total Monthly Cost Estimates

| Scenario | Training (2 cycles) | Inference | Supporting | **Total/Month** |
|----------|--------------------|-----------|-----------|-----------------| 
| Development (Lambda only) | $1.50 | $46 | $40 | **~$88** |
| Production (ECS Graviton, 2 instances) | $5 | $493 | $81 | **~$579** |
| Production with spot scaling (base 1 on-demand + spot scaling) | $5 | $285 | $81 | **~$371** |
| High availability (ECS, 3 instances c8g.2xlarge) | $5 | $700 | $81 | **~$786** |

### 20.6 Cost Optimization Recommendations

1. **Use Graviton over Intel for inference**: c8g.2xlarge ($0.318/hr) delivers 3 to 5x more tokens per dollar than c7i.4xlarge ($0.749/hr) for GGUF workloads.
2. **Mix on-demand and spot for scaling**: Keep 2 on-demand instances for baseline availability; scale with spot instances for peak load. Spot saves 60 to 70% for transient capacity.
3. **Use spot for training**: Enable Managed Spot Training in SageMaker with checkpointing; saves 60 to 90% on training cost with minimal development overhead.
4. **Right-size the model**: Llama 3.2 1B Q4_K_M (700 MB) vs Llama 3.2 3B Q4_K_M (2 GB) uses less RAM and can run on a smaller instance, reducing inference cost by 40%.
5. **Use Lambda for off-hours and bursty workloads**: Scale ECS to 1 instance during low-traffic periods; overflow to Lambda for spikes.
6. **Reserve compute for predictable baseline**: A 1-year reserved instance for the baseline EC2 reduces on-demand cost by 35 to 40%.
7. **Compress CloudWatch metrics**: Emit metrics at 1-minute resolution only during active inference; switch to 5-minute resolution during idle periods.
8. **Use S3 Intelligent-Tiering for model artifacts**: Model files that are not accessed for 30 days are automatically moved to lower-cost storage tiers.

### 20.7 Cost Per Token Benchmarks (mx-central-1)

| Instance | Model | Quant | Tok/s | $/hr | $/1M tokens |
|----------|-------|-------|-------|------|-------------|
| c8g.2xlarge (Graviton4) | Llama 3.2 3B | Q4_K_M | 30 | $0.318 | $2.94 |
| c7g.4xlarge (Graviton3) | Llama 3.2 3B | Q4_K_M | 35 | $0.305 | $2.42 |
| r8g.2xlarge (Graviton4) | Llama 3.2 3B | Q4_K_M | 30 | $0.495 | $4.58 |
| Lambda ARM64 (10GB) | Llama 3.2 1B | Q4_K_M | 20 | ~$0.60 equiv | $8.33 |
| c7i.4xlarge (Intel) | Llama 3.2 3B | Q4_K_M | 14 | $0.749 | $14.86 |

The c7g.4xlarge (Graviton3) delivers the lowest cost per token at $2.42/1M tokens for a 3B model at Q4_K_M quantization, combining good absolute throughput with the lowest hourly rate among the evaluated configurations.
