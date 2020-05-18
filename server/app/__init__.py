from flask import Flask

app = Flask("depth-estimation-server")

from server.app import routes
