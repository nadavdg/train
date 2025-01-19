conda create -n detectron_env python=3.8

 conda activate detectron_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install lableme
