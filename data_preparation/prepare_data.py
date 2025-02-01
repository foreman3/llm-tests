import requests

class DataPreparer:
    def retrieve_data(self, search_term):
        url = 'https://api.duckduckgo.com/'
        params = {
            'q': search_term,
            'format': 'json'
        }
        response = requests.get(url, params=params)
        if (response.status_code == 200 or response.status_code == 202) :
            data = response.json()
            if 'RelatedTopics' in data and len(data['RelatedTopics']) > 0:
                return data['RelatedTopics'][0]['Text']
        return None

    def sanitize_data(self, data):
        # Logic to normalize the data
        pass