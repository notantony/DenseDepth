from flask import Flask

app = Flask("depth-estimation-server")

from server.app import routes

# Initialized in run.py
estimator = None
