import requests


def google_custom_search(query, api_key=None, cse_id=None, num=5):
    """
    Perform a Google Custom Search API query.
    
    :param query: The search query string.
    :param api_key: Your Google API key.
    :param cse_id: Your Google Custom Search Engine ID.
    :param num: Number of results to return (default is 5).
    :return: List of search results.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cse_id,
        "num": num
    }
    response = requests.get(url, params=params)
    return response.json().get('items', [])

    
def serpapi_search(query, api_key=None):
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google",
    }
    response = requests.get(url, params=params)
    return response.json().get('organic_results', [])

