import argparse
import os
import requests
from bs4 import BeautifulSoup

#------------------------------------ Wikipedia Link Retrieval --------------------------------------
def get_wikipedia_link(query):
    # Retrieve API key and search engine ID from environment variables
    api_key = os.getenv('GOOGLE_API_KEY')
    search_engine_id = os.getenv('SEARCH_ENGINE_ID')
    
    if not api_key or not search_engine_id:
        raise ValueError("Environment variables for API key or search engine ID are not set.")

    search_url = "https://www.googleapis.com/customsearch/v1"
    
    params = {
        'key': api_key,
        'cx': search_engine_id,
        'q': query + " site:wikipedia.org",
    }

    response = requests.get(search_url, params=params)
    response.raise_for_status()
    search_results = response.json()

    # Extract the first Wikipedia URL from search results
    if 'items' in search_results:
        for result in search_results['items']:
            if "wikipedia.org" in result['link']:
                return result['link']
    return None

#------------------------------------ Wikipedia Content Scraping --------------------------------------
def scrape_wikipedia_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract main content paragraphs from Wikipedia article
    paragraphs = soup.find_all('p')
    content = "\n".join([para.get_text() for para in paragraphs])
    return content

#------------------------------------ Save Content to File --------------------------------------
def save_to_txt(file_name, content):
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(content)
    print(f"Content saved to {file_name}")

#------------------------------------ Main Execution --------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Get closest Wikipedia article for a given topic")
    parser.add_argument('query', type=str, help="Search query for Wikipedia article")

    args = parser.parse_args()
    query = args.query

    print(f"Searching for the closest Wikipedia article for: {query}")
    
    # Step 1: Get the Wikipedia link
    wikipedia_url = get_wikipedia_link(query)

    if not wikipedia_url:
        print("No Wikipedia article found.")
        return

    print(f"Found Wikipedia article: {wikipedia_url}")

    # Step 2: Scrape the content of the Wikipedia article
    article_content = scrape_wikipedia_article(wikipedia_url)

    # Step 3: Save the content to a .txt file
    file_name = query.replace(" ", "_") + ".txt"
    save_to_txt(file_name, article_content)

if __name__ == "__main__":
    main()
