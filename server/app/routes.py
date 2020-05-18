from server.app import app
from server.app.utils import bad_request


@app.route('/depth', methods=['POST'])
def depth():
    if request.mimetype == "image/jpeg" or request.mimetype == "image/png":
        image_data = request.get_data()
    elif request.mimetype == "application/json":
        json_data = json.loads(request.get_data())
        image_data = base64.decodebytes(json_data["data"].encode())
        _extension = json_data["type"]
    else:
        return bad_request("Unsupported MIME type: `{}`".format(request.mimetype))

    pass