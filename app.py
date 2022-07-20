from flask import Flask, app, request, jsonify

from core import get_prediction

app = Flask(__name__)


@app.route('/predict/', methods=['POST'])
def send_image():
    file = request.files['image']
    file.save('./image_from_api.jpg')

    prediction, confidence = get_prediction('./image_from_api.jpg')

    payload = {
        'Status': 'Success',
        'Message': 'Image sent',
        'Prediction': prediction,
        'Confidence': confidence,
    }

    return jsonify(payload), 201

# @app.route('/test/', methods=['GET', 'POST'])
# def test():
#     payload = {
#         'Status': 'Success',
#         'Message': 'haha brrrr'
#     }

#     return jsonify(payload)


if __name__ == '__main__':
    # url = ngrok.connect(5000).public_url
    # print('>>> Henzy Tunnel URL:', url)

    # app.run(host='192.168.1.105', port=5000, debug=True, threaded=False)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
