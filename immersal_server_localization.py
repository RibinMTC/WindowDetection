import requests
import json
import base64


def ConvertToBase64(src_filepath):
    with open(src_filepath, 'rb') as imageFileAsBinary:
        fileContent = imageFileAsBinary.read()
        b64_encoded_img = base64.b64encode(fileContent)

        return b64_encoded_img


def ServerLocalize(url, token, imagePath):
    complete_url = url + '/localizeb64'

    data = {
        "token": token,
        "fx": 488.323212,  # image focal length in pixels on x axis
        "fy": 488.841583,  # image focal length in pixels on y axis
        "ox": 320.453979,  # image principal point on x axis
        "oy": 234.39151,  # image principal point on y axis
        "b64": str(ConvertToBase64(imagePath), 'utf-8'),  # base64 encoded .png image
        "mapIds": [{"id": 35737}]  # a list of map ids to localize against
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    print(r.text)


ServerLocalize('https://api.immersal.com', 'a69f2d9d15628757b83904aa12d14bb46691cb73263cbfbc7e195cbe61079f3b',
               'input/test_facade.png')
