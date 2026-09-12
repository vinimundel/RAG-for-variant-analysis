# Local runtime safety

The stack is designed to run on developer hardware, including WSL, without loading all model
services at once. CPU execution is the default. GPU execution is an explicit opt-in.

## Incident diagnosis

During the September 2026 BRAF smoke test, two distinct failures were observed:

- A GROBID container with a 1 GiB limit exited with code 137 and reported `OOMKilled=true`.
- A later explicit CUDA probe coincided with a full WSL restart. The previous boot journal was
  unclean and repeatedly reported `dxgkrnl` adapter initialization errors. It did not contain a
  Linux kernel OOM record for the full restart.

The evidence supports a container memory exhaustion event for GROBID and a separate WSL GPU
passthrough or host-level failure for the restart. It does not establish that Linux system memory
exhaustion caused the full WSL restart. Deleting `.vscode-server` restored the editor connection
after the unclean shutdown, but that repairs the remote editor installation rather than the
underlying runtime problem.

An Ollama runner also returned an internal error when given a regular-expression constraint in a
generated JSON schema. The schema now uses simple string fields and validates citation formats in
Python. That failure terminated the runner request; it did not restart WSL.

## Protections in this repository

- MedCPT retrieval and Ollama generation default to CPU.
- CUDA discovery is not performed unless `MEDCPT_DEVICE=cuda` is explicitly set.
- Indexing, retrieval, generation, and evaluation run as sequential stages.
- Runtime guards stop model loading below conservative available-memory thresholds.
- The demo uses batch size four by default; `MEDCPT_BATCH_SIZE=1` is suitable for an 8 GiB WSL VM.
- Ollama defaults to the quantized 1.5B model, `num_gpu=0`, and `keep_alive=0`.
- GROBID has a 2.5 GiB hard limit, a 1.5 GiB Java heap, two CPUs, and no automatic restart.
- Qdrant has a 1 GiB limit. The default demo uses embedded Qdrant and needs no Qdrant container.
- PDFs, TEI, indexes, model caches, and generated reports remain outside version control.

Do not run GROBID, MedCPT indexing, Ollama generation, and RAGAS simultaneously on an 8 GiB WSL
VM. Complete extraction and stop GROBID before indexing. Let retrieval release its models before
generation, as the application does.

## WSL host configuration

For a Windows host with at least 16 GiB RAM, this `%UserProfile%\.wslconfig` is a conservative
starting point:

```ini
[wsl2]
memory=8GB
swap=8GB
processors=8
```

Leave enough RAM for Windows and adjust processor count to the host. Apply changes from PowerShell
with `wsl --shutdown`. Extra swap reduces abrupt allocation failures but cannot make concurrent
model loading efficient.

If `dxgkrnl` errors recur, keep `MEDCPT_DEVICE=cpu` and `OLLAMA_NUM_GPU=0`, update the Windows GPU
driver and WSL, and validate GPU passthrough separately before opting into CUDA. Useful checks are:

```bash
free -h
journalctl -b -1 -k | grep -Ei 'oom|killed process|dxg|gpu'
docker inspect <container> --format '{{.State.OOMKilled}} {{.State.ExitCode}}'
```

The memory guard can be bypassed with `HEALTHRAG_SKIP_MEMORY_GUARD=1` only when an external
scheduler already enforces resource limits.
