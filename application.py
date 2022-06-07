import face_recognition
import numpy as np
import urllib.request
import json
from flask import Flask, request, abort
import logging

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)


@app.route("/FaceRecognitionTraining", methods=['POST'])
def Training_Faces():
    if not request.json or 'urls' not in request.json:
        abort(400)

    all_urls = json.loads(request.json['urls'])

    known_names = []
    known_faces = []

    for key, value in all_urls.items():
        response = urllib.request.urlopen(value)
        image = face_recognition.load_image_file(response)

        encoding = face_recognition.face_encodings(image)[0]

        known_faces.append(encoding)
        known_names.append(key)

    data = {}
    for key in known_names:
        for value in known_faces:
            data[key] = value.tolist()
            known_faces.remove(value)
            break

    new_data = json.dumps(data)
    result_dict = {"output": new_data}
    return result_dict

# app.run(debug=True)
