# import json,requests
#
# a = [1.0, 2.0, 5.0]
#
# data = json.dumps({"signature_name": "serving_default", "instances": [a]})
# headers = {"content-type": "application/json"}
# json_response = requests.post('http://129.226.168.183:8501/v1/models/half_plus_two:predict',
#         data=data, headers=headers)
# predictions = json.loads(json_response.text)["predictions"]
# print(predictions)


import json,requests

a =[1]*800
data = json.dumps({"signature_name": "serving_default", "instances": [a]})
headers = {"content-type": "application/json"}
json_response = requests.post('http://192.168.0.104:8501/v1/models/textclassification:predict',
        data=data, headers=headers)
predictions = json.loads(json_response.text)["predictions"]

print(predictions)