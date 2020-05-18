import argparse
import server.app

from server.app import app


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

def main():
    args = parser.parse_args()
 
    app.run(host=args.host, port=args.port)
