# Hugging Face Space (Docker SDK) for the Article Relationship Model app.
# Assembled and uploaded by scripts/deploy_hf_space.py.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# CPU-only PyTorch keeps the image a fraction of the size of the CUDA build
COPY --chown=user requirements.txt .
RUN pip install --user torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --user -r requirements.txt

# Bake the embedding model into the image so a cold start does not download it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

COPY --chown=user . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
