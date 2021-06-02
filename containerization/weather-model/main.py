from decenter.ai.baseclass import BaseClass
from decenter.ai.appconfig import AppConfig
from decenter.ai.requesthandler import AIReqHandler
from decenter.ai.flask import init_handler
import decenter.ai.utils.model_utils as model_utils

import logging
import sys
import os
import json
from flask import request

from MyModel import MyModel


def main():
    # set logger config
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    # local_testing = True
    # local_testing_json = '{"input": {"url": "http://193.2.72.90:30156/construction.webm" },"output": {"url": {"mqtt":"mqtt://194.249.2.112:30533/ljubljana_yolo_demo"}},"autostart": {"value": "True"}}'

    # Init BaseClass
    #if local_testing:
    #    json_env = json.loads(local_testing_json)
    #    json_env["ai_model"] = {"url": "", "model_name": "weather_model", "model_version": "0.1"}
    #    app = BaseClass(json_env)
    if os.getenv('MY_APP_CONFIG') is None:
        app = BaseClass()
    else:
        json_env = json.loads(os.getenv('MY_APP_CONFIG'))
        json_env["ai_model"] = {"url": "", "model_name": "weather_model", "model_version": "0.1"}
        app = BaseClass(json_env)

    my_model = MyModel(app)

    app.start(my_model)

    if my_model.app.appconfig.get_input_source().scheme == "https" and my_model.app.appconfig.get_autostart() == "True":
        logging.info("starting compute_ai, HTTPS and autostart")

        result = my_model.compute_ai(my_model.app.appconfig.get_input_source().geturl())
        my_model.app.fire_notification(result)

    # start Flask message handler here
    msg_handler = init_handler(app)

    flaskapp = msg_handler.get_flask_app()

    @flaskapp.route('/getCurrentFPS', methods=['GET'])
    def getFPS():
        return str(my_model.get_current_fps())

    @flaskapp.route('/getCurrentDelay', methods=['GET'])
    def getDelay():
        return str(my_model.get_current_video_delay())

    @flaskapp.route('/setMinimumFPS', methods=['GET'])
    def setMinimumFPS():
        minimum_fps = request.args.get('minimum_fps', default=30, type=int)
        if my_model.set_minimum_fps(minimum_fps):
            return "setting minimum fps to " + str(request.args.get('minimum_fps', default=30, type=int))
        else:
            return "did not change fps. They have to between 60 and 1"

    @flaskapp.route('/startAI', methods=['GET'])
    def startAI():
        minimum_fps = request.args.get('minimum_fps', default=30, type=int)
        return my_model.start_thread(minimum_fps)

    @flaskapp.route('/stopAI', methods=['GET'])
    def stopAI():
        return my_model.stop_thread()

    flaskapp.run(host="0.0.0.0", threaded=True)


main()
