import pathlib
from PIL import Image
import io

from flask import Flask, request, jsonify

from window_inference_model import WindowsInferenceModel
import json

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello World'


@app.route('/image-window-inference', methods=['GET', 'POST'])
def recommendation_query():
    detected_windows_with_corners = {"ImageWindowsCoordinates": []}
    try:
        request_data = request.get_json()
        image_data = bytes(request_data['ImageBytes'])
        image = Image.open(io.BytesIO(image_data))
        image.save(dir / "input/test_facade.png")
        windows_list_with_score = windowsInferenceModel.infer('input/')
        if len(windows_list_with_score) == 0:
            print("No windows detected!")
            return detected_windows_with_corners
        detected_windows_with_corners["ImageWindowsCoordinates"] = str(windows_list_with_score)
        write_window_coordinates_to_json(detected_windows_with_corners)
        return jsonify(detected_windows_with_corners)

    except Exception as e:
        print(e)
        return detected_windows_with_corners


def write_window_coordinates_to_json(window_coordinates):
    with open('output/window_coordinates.json','w') as outfile:
        json.dump(window_coordinates, outfile)


dir = pathlib.Path(__file__).parent.absolute()

windowsInferenceModel = WindowsInferenceModel('experiments/resnet/lr1e-3_x120-90-110_center_b2.yaml',
                                              'models/resnet18_model_latest.pth.tar', 'input/')

if __name__ == '__main__':
    app.run(port=5005, host='0.0.0.0')
