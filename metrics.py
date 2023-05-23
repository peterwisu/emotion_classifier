import time
import requests
import matplotlib.pyplot as plt

# Number of requests to test
num_requests = list(range(1, 50001, 1000))

# Test data for requests
test_data = [{'text': 'I am furious right now!'}]  # Add more test data if needed

def send_request(text):
    response = requests.post('http://localhost:8000/predict', json={"text": text})
    return response

elapsed_times = []
cumulative_times = []

for num_req in num_requests:
    start_time = time.time()
    for _ in range(num_req):
        send_request(test_data[0]['text'])
    end_time = time.time()

    elapsed_time = end_time - start_time
    elapsed_times.append(elapsed_time)

    cumulative_time = sum(elapsed_times)
    cumulative_times.append(cumulative_time)

# Plotting the cumulative time
plt.plot(num_requests, cumulative_times, label='Cumulative Time')
plt.xlabel('Number of Requests')
plt.ylabel('Cumulative Time (seconds)')
plt.title('Model Performance - Cumulative Time')
plt.legend()

# Save the graph as an image file
plt.savefig('performance_graph.png')
