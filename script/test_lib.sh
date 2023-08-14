cd $(dirname ${0})
cd ../
pwd
pytest lib/data/test_data.py
pytest lib/method/test_method.py
pytest lib/train/test_train.py