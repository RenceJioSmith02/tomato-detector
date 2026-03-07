Requirements beh:
=======================================================================================
Python: 3.13.1

pip install opencv-python

Go to cmd locate the project and inside the web/: pip install -r requirements.txt

IF there are new requirements: pip freeze > requirements.txt

IF prefer a venv:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

=======================================================================================

train → Images used for training the model. The model updates its weights based on these.

val → Images used for validation during training. These help the model tune hyperparameters and decide when to stop, but the model does not learn from them.

test → Images used for final evaluation. The model never sees these during training or validation. They measure how well your model generalizes to unseen data.