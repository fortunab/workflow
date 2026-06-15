# ggml

## Manifesto

**Tensor library for machine learning**

> Note that this project is under active development. Some of the development is currently happening in the `llama.cpp` and `whisper.cpp` repositories.

---

## Features

- Low-level cross-platform implementation
- Integer quantization support
- Broad hardware support
- Automatic differentiation
- ADAM and L-BFGS optimizers
- No third-party dependencies
- Zero memory allocations during runtime

---

## Build

```bash
git clone https://github.com/ggml-org/ggml
cd ggml
```

Create a Python virtual environment and install dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Build the examples:

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release -j 8
```

---

## GPT Inference Example

Download the GPT-2 Small (117M) model:

```bash
../examples/gpt-2/download-ggml-model.sh 117M
```

Run inference:

```bash
./bin/gpt-2-backend -m models/gpt-2-117M/ggml-model.bin -p "This is an example"
```

---

## Resources

- Introduction to ggml
- The GGUF file format

---

## Related Projects

- llama.cpp
- whisper.cpp

  Example: winget install llama.cpp.
  
