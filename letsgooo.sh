touch ~/.no_auto_tmux
sudo apt-get install python3-dev
curl https://bootstrap.pypa.io/get-pip.py | python3
pip install --upgrade pip
pip install nvitop
pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/
sudo apt-get -y install python3-dev
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1 --trust-remote-code --tp 2

