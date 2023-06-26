import time
from flask import Flask
from interface import Interface

app = Flask(__name__)

@app.route('/time')
def video_to_text():
    return {'time' : time.time()}

if __name__ == '__main__':
    app.run(debug=True)