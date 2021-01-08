import os
import json
import shlex
import subprocess
from datetime import datetime

from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS

process = None
model = None

def server():
    app = Flask(__name__)
    CORS(app)
    app.debug = True

    def convert_date(timestamp):
        d = datetime.utcfromtimestamp(timestamp)
        formated_date = d.strftime('%d %b %Y')
        return formated_date

    def event_stream(system_command, **kwargs):
        popen = subprocess.Popen(
            shlex.split(system_command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            universal_newlines=True,
            **kwargs)
        global process
        process = popen
        for stdout_line in iter(popen.stdout.readline, ""):
            stream_obj = {
                'execution': True,
                'response': stdout_line.strip()
            }
            yield "data: {}\n\n".format(json.dumps(stream_obj))
        popen.stdout.close()
        yield "data: {}\n\n".format(json.dumps({'execution': False}))
        return_code = popen.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, system_command)

    @app.route('/train', methods=['GET'])
    def train():
        return Response(event_stream('py main.py --train train.csv'), mimetype="text/event-stream")

    @app.route('/stream', methods=['GET'])
    def stream():
        try:
            if request.args['keyword']:
                # print(request.args['keyword'])
                return Response(
                    event_stream(
                        'py main.py --stream "' + str(request.args['keyword']) + '"'
                    ),
                    mimetype="text/event-stream"
                )
        except KeyError:
            return Response("data: {}\n\n".format(json.dumps({
                    'info': 'No keyword specified. Please Specify a stream keyword'
            })), mimetype='text/event-stream')

    @app.route('/prevStream', methods=['GET'])
    def prev_stream():
        files = []

        with os.scandir('stream_data') as contents:
            for entry in contents:
                info = entry.stat()
                files.append({
                    'name': entry.name,
                    'size': info.st_size,
                    'created_at': convert_date(info.st_ctime),
                    'modified_at': convert_date(info.st_mtime)

                })
        return jsonify(files)

    @app.route('/test')
    def test():
        return Response(event_stream('py test.py'), mimetype="text/event-stream")

    @app.route('/terminate')
    def terminate():
        global process
        if process is not None:
            process.terminate()
            return jsonify({
                    'info': 'Subprocess Terminated Successfully'
                })
        else:
            return jsonify({
                'info': "No subprocess is active at the moment"
                })

    @app.route('/', methods=['GET'])
    def index():
        return render_template('index.html')
        
    @app.route('/classify', methods=['GET'])
    def classify():
        pass

    app.run(threaded=True)


server()
