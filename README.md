## Fine-tuning LLMs in Docker

Fine-tune large language models (LLMs) using the Hugging Face Transformers library. The Dockerfile sets up an environment with the necessary dependencies, including PyTorch, CUDA, and various Python packages, to enable GPU-accelerated fine-tuning.

### Usage

Build the Docker image using the following command:

```bash
docker build -t llm-fine-tuning .
```

After building the image, you can run the container to start the fine-tuning process:

```bash
docker run --gpus all llm-fine-tuning
```

Please adjust the `fine_tune_llm.py` script and provide the necessary dataset files in the `dataset/` directory according to your specific fine-tuning requirements.

> Note: This `Dockerfile` assumes you have a compatible GPU and the necessary NVIDIA drivers installed on your host system.

To run the inference script, you can execute it directly using Python:

```bash
python3 inference.py
```

### License

This project is licensed under the [MIT License](./LICENSE).

### Copyright

(c) 2024 [Finbarrs Oketunji](https://finbarrs.eu).