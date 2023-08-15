cd $(dirname ${0})
cd ../
pytest lib/data/test_data.py
pytest lib/method/test_method.py
pytest lib/train/test_train.py
pytest lib/visualize/test_visualize.py