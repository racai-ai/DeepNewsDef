import requests
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_random_wikipedia_extract(lang: str = "ro", chars_limit: int = 1500, max_retries: int = 5) -> tuple:
    """
    Fetches a random article extract from Wikipedia for the specified language and its URL.
    Useful for providing factual seeds (e.g., based_on_real_article).
    """
    url = f"https://{lang}.wikipedia.org/w/api.php"
    
    params = {
        "action": "query",
        "format": "json",
        "generator": "random",
        "grnnamespace": 0,  # Only target main articles
        "prop": "extracts|info",
        "inprop": "url",
        "exchars": chars_limit,
        "explaintext": 1,   # Get plaintext, not HTML
        "exintro": 1        # Only fetch the intro part
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait_s = int(retry_after)
                else:
                    wait_s = min(60, 2 ** attempt)
                logger.warning(
                    "Wikipedia rate-limited (429). Waiting %ss before retry %s/%s.",
                    wait_s,
                    attempt,
                    max_retries,
                )
                time.sleep(wait_s)
                continue

            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return "", ""

            # Get the first (and only) random page returned
            for page_id, page_data in pages.items():
                extract = page_data.get("extract", "").strip()
                article_url = page_data.get("fullurl", "")
                if not extract:
                    return "", ""
                paragraphs = [p.strip() for p in extract.split("\n") if p.strip()]
                return paragraphs[0] if paragraphs else "", article_url

        except Exception as e:
            if attempt >= max_retries:
                logger.error(f"Failed to fetch Wikipedia extract: {e}")
                return "", ""
            wait_s = min(60, 2 ** attempt)
            logger.warning(
                "Wikipedia fetch failed (%s). Retrying in %ss (%s/%s).",
                e,
                wait_s,
                attempt,
                max_retries,
            )
            time.sleep(wait_s)

if __name__ == "__main__":
    extract, article_url = fetch_random_wikipedia_extract()
    print("--- Wiki Extract ---")
    print(extract)
    print("URL:", article_url)
    print("--------------------")
