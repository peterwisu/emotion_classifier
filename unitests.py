import unittest
import requests
import json


class TestApi(unittest.TestCase):

    def setUp(self):
        self.url = "http://127.0.0.1:8000/predict"
        self.headers = {'Content-Type': 'application/json'}

    def test_text_boundary_short(self):
        # short text
        data = {'text': 'A'}
        response = requests.post(
            self.url, headers=self.headers, data=json.dumps(data))
        self.assertEqual(response.status_code, 200,
                         f"Expected status code 200, but got {response.status_code}")

    def test_text_boundary_long(self):
        # long text (512 length)
        data = {'text': 'A' * 512}
        response = requests.post(
            self.url, headers=self.headers, data=json.dumps(data))
        self.assertEqual(response.status_code, 200,
                         f"Expected status code 200, but got {response.status_code}")

    def test_input_validation(self):
        # no data
        response = requests.post(self.url, headers=self.headers)
        self.assertEqual(response.status_code, 422,
                         f"Expected status code 422, but got {response.status_code}")

    def test_invalid_JSON(self):
        # invalid JSON
        response = requests.post(
            self.url, headers=self.headers, data="this is not valid JSON")
        self.assertEqual(response.status_code, 422,
                         f"Expected status code 422, but got {response.status_code}")

    def test_missing_field(self):
        # missing 'text' field
        data = {'not_text': 'This field should be called text'}
        response = requests.post(
            self.url, headers=self.headers, data=json.dumps(data))
        self.assertEqual(response.status_code, 422,
                         f"Expected status code 422, but got {response.status_code}")

    def test_additional_field(self):
        # additional unexpected field
        data = {'text': 'Hello, world',
                'extra_field': 'This should not be here'}
        response = requests.post(
            self.url, headers=self.headers, data=json.dumps(data))
        self.assertEqual(response.status_code, 200,
                         f"Expected status code 200, but got {response.status_code}")

    def test_output_validation(self):
        data = {'text': 'I am very happy'}
        response = requests.post(
            self.url, headers=self.headers, data=json.dumps(data))

        self.assertEqual(response.status_code, 200,
                         f"Expected status code 200, but got {response.status_code}")
        response_data = response.json()

        # Check 'results' field
        self.assertIn('results', response_data,
                      "Response does not contain 'results'")

        # Check 'text', 'label', and 'score' field in the first result
        result = response_data['results'][0]
        self.assertIn('text', result, "Result does not contain 'text'")
        self.assertIn('label', result, "Result does not contain 'label'")
        self.assertIn('score', result, "Result does not contain 'score'")

        self.assertIn(result['label'], ['anger', 'confusion', 'curiosity', 'desire', 'digust', 'embarrassment',
                                        'fear', 'joy', 'love', 'neutral', 'optimism', 'pride', 'sadness', 'surprise'], "Invalid label")
        self.assertIsInstance(result['score'], float, "Score is not a float")
        self.assertTrue(0 <= result['score'] <= 1,
                        "Score is not between 0 and 1")


if __name__ == "__main__":
    unittest.main()
