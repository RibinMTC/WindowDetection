import json
import pathlib

from window_detection_api.window_inference_model import WindowsInferenceModel


def infer_windows_list():
    windows_list_with_score = windowsInferenceModel.infer('input_local/')
    if len(windows_list_with_score) == 0:
        print("No windows detected!")
        return None
    return windows_list_with_score


def write_window_coordinates_to_json(window_coordinates):
    with open('../output/zhdk_stussihof_window_coordinates.json', 'w') as outfile:
        json.dump(window_coordinates, outfile, indent=4)


if __name__ == '__main__':
    dir = pathlib.Path(__file__).parent.absolute()

    windowsInferenceModel = WindowsInferenceModel('experiments/resnet/lr1e-3_x120-90-110_center_b2.yaml',
                                                  'models/resnet18_model_latest.pth.tar')

    windows_corner_list = infer_windows_list()

    write_window_coordinates_to_json(windows_corner_list)
