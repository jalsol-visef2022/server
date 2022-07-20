from flask import Flask, app, request, jsonify
import time

import atexit
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from core import get_prediction
import retrain

app = Flask(__name__)


@app.route('/predict/', methods=['POST'])
def send_image():
    file = request.files['image']
    path = f'storage/cache/{time.time_ns()}.jpg'

    file.save(path)
    prediction, confidence = get_prediction(path)

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
    scheduler = BackgroundScheduler()
    scheduler.start()

    trigger = CronTrigger(
        year="*", month="*", day="*",
        hour="0", minute="0", second="0"
    )

    scheduler.add_job(retrain.retrain, trigger=trigger)

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)

    atexit.register(lambda: scheduler.shutdown())
