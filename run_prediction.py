import json
import requests


def get_prediction():

    # ML Flow Deployment Server URL
    url = input('\nEnter Deployment URL for inference\n')

    # Sample Data for Inference
    json_file = input('\nEnter sample data file path (JSON)\n')
    if not json_file.endswith('json'):
        raise ValueError('Invalid Input')
    input_data = json.dumps(json.load(open(json_file, 'r')))

    # Set the headers for the request
    headers = {"Content-Type": "application/json"}

    # sent request to the given url
    result = requests.post(url, headers=headers, data=input_data)

    # Check the status code
    if result.status_code == 200:
        # If successful, print the prediction result
        prediction = result.json()
        print("Prediction:", prediction)
    else:
        # If there was an error, print the status code
        print(f"Error: {result.status_code}")
        print(result.text)


if __name__ == '__main__':
    get_prediction()