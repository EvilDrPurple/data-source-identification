import json
import ssl
import urllib3
import urllib.request
from concurrent.futures import as_completed, ThreadPoolExecutor
import traceback

import requests
from tqdm import tqdm
import bs4
from bs4 import BeautifulSoup


# Define the list of header tags we want to extract
header_tags = ["h1", "h2", "h3", "h4", "h5", "h6"]


def process_urls(urls):
    """Process a list of urls and retrieve their HTML tags.

    Args:
        urls (list): List of urls.

    Returns:
        list: List of dicts with HTML tags.
    """
    results = []
    new_urls = ["https://" + url if not url.startswith("http") else url for url in urls]

    with ThreadPoolExecutor(max_workers=100) as executor:
        print("Retrieving HTML tags...")
        future_to_url = [executor.submit(get_response, url, i) for i, url in enumerate(new_urls)]

        for future in tqdm(as_completed(future_to_url), total=len(future_to_url)):
            data = future.result()
            results.append(data)

    results.sort(key=lambda d: d["index"])
    urls_and_responses = [{"url": urls[i], "response": result["response"]} for i, result in enumerate(results)]

    urls_and_headers = []
    tags = None

    print("Parsing responses...")
    # TODO: May want to parallelize this as well, it tends to take a while
    for row in tqdm(urls_and_responses):
        if tags is not None:
            urls_and_headers.append(tags)

        tags = {}
        res = row["response"]
        tags["url"] = row["url"]

        if res is None:
            tags["http_response"] = "Request failed"
            continue

        tags["http_response"] = res.status_code
        if not res.ok:
            continue

        try:
            soup = BeautifulSoup(res.content, "html.parser", from_encoding="iso-8859-1")
        except (bs4.builder.ParserRejectedMarkup, AssertionError):
            continue

        tags["html_title"] = soup.title.string if soup.title is not None else ""

        meta_tag = soup.find("meta", attrs={"name": "description"})
        try:
            tags["meta_description"] = meta_tag["content"] if meta_tag is not None else ""
        except KeyError:
            tags["meta_description"] = ""

        html_headers = []
        for header_tag in header_tags:
            headers = soup.find_all(header_tag)
            header_content = [header.text for header in headers]
            html_headers.append({header_tag: header_content})
        tags["html_headers"] = html_headers

    urls_and_headers.append(tags)

    return urls_and_headers


def get_response(url, index):
    """Retrieves GET response for given url.

    Args:
        url (str): Url to request.

    Returns:
        dict(int, Response): Dictionary of the url's index value and Response object, None if an error occurred.
    """

    headers = {
        # Some websites refuse the connection of automated requests, setting the User-Agent will circumvent that
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        # Make sure there's no pre-mature closing of responses before a redirect completes
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    }
    response = None
    debug = True  # Set to True to enable debug output
    url = url.removesuffix(".json")

    try:
        response = requests.get(url, headers=headers, timeout=60)
    except requests.exceptions.SSLError:
        # This error is raised when the website uses a legacy SSL version, which is not supported by requests
        if debug:
            print("SSLError:", url)
        # Retry using legacy SSL session
        response = get_legacy_session().get(url, headers=headers, timeout=60)
    except requests.exceptions.ConnectionError:
        # Sometimes this error is raised because the provided url uses http when it should be https and the website does not handle it properly
        if debug:
            print("MaxRetryError:", url)

        if not url[4] == "s":
            url = url[:4] + "s" + url[4:]
            # Retry with https
            response = requests.get(url, headers=headers, timeout=60)
    except (urllib3.exceptions.LocationParseError, requests.exceptions.ReadTimeout) as e:
        if debug:
            print(f"{type(e).__name__}: {url}")
    except Exception as e:
        if debug:
            print("Exception:", url)
            print(traceback.format_exc())
            print(str(e))
    finally:
        if debug:
            print(url, response)

        return {"index": index, "response": response}


# The following adapter code was shamelessly stolen from Harry Mallon on Stack Overflow:
# https://stackoverflow.com/a/71646353/14045691
class CustomHttpAdapter(requests.adapters.HTTPAdapter):
    # "Transport adapter" that allows us to use custom ssl_context.
    def __init__(self, ssl_context=None, **kwargs):
        self.ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = urllib3.poolmanager.PoolManager(
            num_pools=connections, maxsize=maxsize, block=block, ssl_context=self.ssl_context
        )


def get_legacy_session():
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
    session = requests.session()
    session.mount("https://", CustomHttpAdapter(ctx))
    return session


# Define a function to process a URL and update the JSON object
def process_url(url, d={}):
    try:
        response = requests.get(url)
    except (requests.exceptions.ConnectionError, requests.exceptions.RequestException):
        print(f"Error: Invalid URL - {url}")
        print("Invalid URLs removed from output data")
        return False

    # Add the HTTP response status code as a new property
    d["http_response"] = response.status_code

    # If the response status code is not in the 200 range, skip adding other properties
    if not response.ok:
        return

    # Check if the URL ends with .pdf
    if url.endswith(".pdf"):
        return

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract the title tag and its content

    html_title = soup.title.string

    # Extract the meta description tag and its content
    meta_tag = soup.find("meta", attrs={"name": "description"})
    meta_description = meta_tag["content"] if meta_tag is not None else ""

    # Extract the header tags and their content
    html_headers = []
    for header_tag in header_tags:
        headers = soup.find_all(header_tag)
        header_content = [header.text for header in headers]
        html_headers.append({header_tag: header_content})

    # Add the new properties to the JSON object
    d["html_title"] = html_title
    d["meta_description"] = meta_description
    d["html_headers"] = html_headers

    return d


def main():
    # Open the input file and load the JSON data
    with open("urls.json") as f:
        data = json.load(f)

    # Loop over each URL in the data
    valid_data = []
    for d in data:
        url = d["url"]
        if process_url(url, d) != False:
            valid_data.append(d)

    # Write the updated JSON data to a new file
    with open("urls_and_headers.json", "w") as f:
        json.dump(valid_data, f, indent=4)


if __name__ == "__main__":
    main()