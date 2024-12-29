FROM registry.deez.re/research/python-gpu-11-2:3.9

WORKDIR /workspace

COPY requirements.txt ./    
RUN pip install -r requirements.txt

# -------------------------------
# Jupyter setup
EXPOSE 8888
WORKDIR /workspace

