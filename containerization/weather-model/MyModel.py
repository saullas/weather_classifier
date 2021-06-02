from decenter.ai.baseclass import BaseClass
from decenter.ai.appconfig import AppConfig
from decenter.ai.requesthandler import AIReqHandler
from decenter.ai.flask import init_handler
from keras import backend as K

import base64
import numpy as np

import tensorflow as tf

import logging, sys
import time
import asyncio
from threading import Thread
import os

import zipfile

import cv2
import json
from uuid import uuid4

from urllib.request import urlopen, urlretrieve
from flask import jsonify

import paho.mqtt.client as paho

class MyModel:
    def __init__(self, app):
        """
        """
        self.app = app
        self.sess = tf.compat.v1.Session()

        self.fps = -1
        self.current_frame_time = -1

        self.source = None
        self.cap = None
        self.thread = None

        self.skip_frames = False
        self.continue_running = False
        self.id = str(uuid4())

        self.categories = ['cloudy', 'foggy', 'rain', 'snow', 'sunny']

        logging.info('creating session')

    def load_ai_model(self, filename):
        print("FILENAME: ", filename)
        logging.info('File downloaded, extacting...' )
        zip_ref = zipfile.ZipFile( filename, 'r')
        zip_ref.extractall("/")
        self.model = tf.keras.models.load_model('ai_model.h5')

        logging.info('Loading AI model onto memory...' )

        return 0

    def ai_thread(self, *args):
        def on_connect(client, userdata, flags, rc):
            print("CONNECT RECEIVED with code %d." % (rc))

        def on_publish(client, userdata, mid):
            print("PUBLISHED")

        def on_message(client, userdata, message):
            print("message received ", str(message.payload.decode("utf-8")))
            print("message topic=", message.topic)
            print("message qos=", message.qos)
            print("message retain flag=", message.retain)

        client = paho.Client(transport="websockets")
        client.on_connect = on_connect
        client.on_publish = on_publish
        # client.on_message = on_message
        print(self.app.appconfig.get_destination())
        result = client.connect(self.app.appconfig.get_destination()['mqtt'].hostname, self.app.appconfig.get_destination()['mqtt'].port)
        client.loop_start()

        # if you need to check the mqtt data send to prometheus you need to uncomment this an on_message line
        # client.subscribe("prometheus/job/AI_metrics/instance/yolov3/monitoring_fps")
        # client.subscribe("prometheus/job/AI_metrics/instance/yolov3/monitoring_video_delay")
        logging.info('opening source at: ' + self.source)

        # AXIS ip camera MJPEG
        self.cap = cv2.VideoCapture(self.source)
        self.set_minimum_fps(args[0])
        self.current_frame_time = time.time()
        prometheus_time = time.time()

        while self.continue_running:
            # skip frames until the delay is smaller than the value specified
            if self.skip_frames:
                self.current_frame_time += 1 / self.cap.get(cv2.CAP_PROP_FPS)

                if self.get_current_video_delay() < 0.2:
                    self.skip_frames = False
                    self.current_frame_time = time.time()
            else:
                try:
                    ret, frame_ori = self.cap.read()
                    start_time = time.time()  # start time of the loop
                    print()

                    if ret:
                        height_ori, width_ori = frame_ori.shape[:2]
                        frame = cv2.resize(frame_ori, tuple([64, 64]))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = np.asarray(frame) / 255.0
                        frame = frame.reshape(1, 64, 64, -1)

                        prediction = self.model.predict(frame)[0]
                        detections = []

                        for value, category in zip(prediction, self.categories):
                            detections.append({
                                'class': str(category),
                                'location': '',
                                'score': str(np.round(value, 3))
                            })

                        _, buffer = cv2.imencode('.jpg', frame_ori)
                        jpg_as_text = base64.b64encode(buffer).decode('ascii')

                        message = {'ai_id': str(self.id),
                                   'fps': str(self.fps),
                                   'delay': str(self.get_current_video_delay()),
                                   'timestamp': str(self.current_frame_time),
                                   'detections': detections,
                                   'encoded_image': jpg_as_text}
                        
                        message = json.dumps(message)
                        # print(message)

                        topic = str(self.app.appconfig.get_destination()["mqtt"].path)
                        
                        if topic[0] == "/":
                            topic = topic[1:]

                        infot = client.publish(topic, message, qos=0)
                        infot.wait_for_publish()

                        # send data to prometheus every 2 seconds
                        #if time.time() - prometheus_time > 1:
                        #    print("publishing to prometeus current FPS")
                        #    infot = client.publish("prometheus/job/AI_metrics/instance/yolov3/monitoring_fps", self.get_current_fps())
                        #    infot.wait_for_publish()
                        #    infot = client.publish("prometheus/job/AI_metrics/instance/yolov3/monitoring_video_delay", self.get_current_video_delay())
                        #    infot.wait_for_publish()
                        #    prometheus_time = time.time()

                        # yield(b'--frame\r\n'
                        #   b'Content-Type: image/jpeg\r\n\r\n' + frame_out + b'\r\n')

                        self.fps = 1.0 / (time.time() - start_time)
                        self.reset_video_feed()
                        if self.cap.isOpened():
                            self.current_frame_time += 1 / self.cap.get(cv2.CAP_PROP_FPS)
                    else:
                        logging.info("NO DATA!")
                        break
                except Exception as ex:
                    print(f"An exception occured while processing a frame: {type(ex)}: {ex}")

        client.disconnect()
        self.cap.release()
        return "DONE"

    def get_current_fps(self):
        if not self.continue_running:
            return -1
        return self.fps

    def get_current_video_delay(self):
        if not self.continue_running:
            return -1
        return time.time() - self.current_frame_time

    def set_minimum_fps(self, minimum_fps):
        if self.continue_running and 60 >= minimum_fps != self.cap.get(cv2.CAP_PROP_FPS) and minimum_fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, minimum_fps)
            return True
        return False

    def reset_video_feed(self):
        if self.get_current_video_delay() > 1:
            print("Delay was grater then the limit, so video feed was reset to align with the current stream")
            self.skip_frames = True

    def start_thread(self, fps):
        if not self.continue_running:
            print("starting AI computations")
            self.thread = Thread(target=self.ai_thread, args=[fps])
            self.continue_running = True
            self.thread.start()
            return "STARTED"
        return "ALREADY RUNNING"

    def stop_thread(self):
        print("Stopping AI computations")
        if self.continue_running:
            self.continue_running = False
            self.fps = -1
            self.current_frame_time = - 1
            self.cap.release()
            return "AI STOPPED"
        return "AI WAS NOT RUNNING"

    def compute_ai(self, *args):
        self.source = ''.join(args[0])
        # if we would need auto start of the AI model
        self.start_thread(30)
        return {}
