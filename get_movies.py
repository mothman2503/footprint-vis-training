import requests
import csv
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlencode

BASE_URL = "https://www.imdb.com/list/ls050782187/"
HEADERS = {"Accept-Language": "en-US,en;q=0.9","User-Agent": "Mozilla/5.0"}

def fetch_page(url, params=None):
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    return r.text

def parse_titles(html):
    soup = BeautifulSoup(html, "html.parser")
    titles = []
    for h3 in soup.select("h3.lister-item-header a"):
        titles.append(h3.text.strip())
    return titles

def main():
    all_titles = []
    # Strategy 1: Pagination via `?start=`
    for start in range(1, 501, 100):  # pages of up to 100 movies
        html = fetch_page(BASE_URL, params={"start": start})
        titles = parse_titles(html)
        if not titles:
            break
        all_titles += titles

    # If infinite scroll, fallback to clicking Load More via Selenium
    if len(all_titles) < 500:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        opts = Options()
        opts.headless = True
        driver = webdriver.Chrome(options=opts)
        driver.get(BASE_URL)
        wait = WebDriverWait(driver, 10)

        while True:
            try:
                btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.ipc-see-more__button")))
                btn.click()
                wait.until(EC.staleness_of(btn))
            except:
                break

        soup = BeautifulSoup(driver.page_source, "html.parser")
        all_titles = parse_titles(soup.prettify())
        driver.quit()

    all_titles = all_titles[:500]
    print(f"Found {len(all_titles)} titles.")
    
    with open("imdb_top500_titles.csv", "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["title"])
        for title in all_titles:
            writer.writerow([title])
        print("✅ CSV saved: imdb_top500_titles.csv")

if __name__ == "__main__":
    main()
