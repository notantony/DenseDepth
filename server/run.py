import argparse
import server.app

from server.app import app
from server.api.estimator import Estimator


parser = argparse.ArgumentParser(
    description='Run depth estimation server'
)
# Server settings
parser.add_argument(
    '--host',
    type=str,
    default='0.0.0.0',
    help='Host address'
)
parser.add_argument(
    '--port',
    type=int,
    default=5052,
    help='Listening port'
)
parser.add_argument(
    '--model',
    default='kitti.h5',
    type=str,
    help='Path to trained Keras model file.'
)

def main():
    args = parser.parse_args()

    server.app.estimator = Estimator(args.model)
    with server.app.estimator:
        app.run(host=args.host, port=args.port)
