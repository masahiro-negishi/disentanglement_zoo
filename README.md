# disentanglement_zoo
[![licence: MIT](https://black.readthedocs.io/en/stable/_static/license.svg)](https://github.com/masahiro-negishi/disentanglement_zoo/blob/main/LICENSE)
[![Codestyle: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

PyTorch reproduction of models for disentangled representation learning

## Environment setup
All the libraries you need for executing codes are specified in requirements.txt. Note that cuda version should be modified based on your hardware. In addition, libraries for formatters and linters are included in the file as comments. Thus, if you want to write codes additionaly using the same formatters and linters as me, you should comment out the lines and install them as well. If you use venv, you can set up your environments as follows.
```
python -m venv your_env
source your_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Test lib
You can run pytest for all test files in lib as follows.
```
./script/test_lib.sh
```

## Example
For training
```
python script/interface.py train --dataset=shapes3d --train_size=10000 --eval_size=1000 --batch_size=64 --seed=0 --z_dim=10 --device=cuda --lr=1e-4 --epochs=30 --train_log=1 --save --save_dir="result/shape3d"
```

For evaluation
```
python script/interface.py eval --save_dir="result/shape3d" --device=cpu --recons --num=5 --latent_dist
```