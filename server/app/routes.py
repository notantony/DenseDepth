import base64
import os
import json
import traceback
import server.app
import io

from flask import request, jsonify
from server.app import app
from server.app.utils import bad_request, server_error


@app.route('/depthmap', methods=['POST'])
def depthmap():
    if request.mimetype == "image/jpeg" or request.mimetype == "image/png":
        image_data = request.get_data()
    elif request.mimetype == "application/json":
        json_data = json.loads(request.get_data())
        image_data = base64.decodebytes(json_data["data"].encode())
    else:
        return bad_request("Unsupported MIME type: `{}`".format(request.mimetype))
    
    try:
        depthmap = server.app.estimator.estimate(image_data)
    except Exception as e:
        traceback.print_exc()
        return server_error(repr(e))

    response = {}
    response["depthmap"] = base64.b64encode(depthmap.tobytes()).decode("utf-8").replace("\n", "")
    response["shape"] = str(depthmap.shape)
    response["dtype"] = str(depthmap.dtype)
    
    return jsonify(response)
