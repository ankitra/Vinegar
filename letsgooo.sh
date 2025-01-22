touch ~/.no_auto_tmux
pip install --upgrade pip
pip install nvitop
pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/
sudo apt-get install python3-dev
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --trust-remote-code --tp 2

