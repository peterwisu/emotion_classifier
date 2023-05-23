import requests

# Test data for requests
test_data = [
    {'text': 'I am very angry right now!'},
    {'text': 'This is utterly disgusting.'},
    {'text': 'I\'m truly scared.'},
    {'text': 'I\'m overjoyed today!'},
    {'text': 'I\'m feeling so much love.'},
    {'text': 'I\'m swelling with pride.'},
    {'text': 'I\'m deeply sad.'},
    {'text': 'I\'m totally surprised!'},
    {'text': 'I\'m feeling quite optimistic.'},
    {'text': 'I\'m totally confused.'},
    {'text': 'I\'ve never been this embarrassed.'},
    {'text': 'I desire that so much.'},
    {'text': 'I\'m really curious about this.'},
    {'text': 'I\'m feeling quite neutral.'}
]

def send_request(text):
    response = requests.post('http://localhost:8000/predict', json={"text": text})
    print(f'Text: {text}\nResponse: {response.json()}')

for data in test_data:
    send_request(data['text'])