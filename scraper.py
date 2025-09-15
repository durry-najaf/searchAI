import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import concurrent.futures

def scrape_website_parallel(start_url, depth=1, max_links=100):
    visited = {}
    to_visit = {start_url}  # Use a set to avoid duplicates

    for _ in range(depth):
        next_to_visit = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(scrape_page, url): url for url in to_visit}

            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    text, new_links = future.result()
                    if url not in visited and len(visited) < max_links:
                        visited[url] = text
                        next_to_visit.update(new_links)
                except Exception:
                    continue
        to_visit = next_to_visit  # Move to the next depth level
    return visited

def scrape_page(url):
    """ Scrape a single page and return text + new links """
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return "", []
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text().strip()
        new_links = set()
        for link in soup.find_all("a", href=True):
            absolute_link = urljoin(url, link["href"])
            new_links.add(absolute_link)
        return text, new_links
    except requests.RequestException:
        return "", [] 
