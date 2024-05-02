import ast
import csv
import sys
import os
from urllib.parse import urlparse

csv.field_size_limit(sys.maxsize)
FILE = "train-urls-keywords.csv"

with open(FILE, newline="") as readFile, open("new.csv", "w", newline="") as writeFile:
    reader = csv.DictReader(readFile)
    fieldnames = ["url", "url_path", "label", "html_title", "meta_description", "root_page_title", "http_response", "keywords", "h1", "h2", "h3", "h4", "h5", "h6", "div_text"]
    writer = csv.DictWriter(writeFile, fieldnames=fieldnames)
    writer.writeheader()
    
    for row in reader:
        write_row = row

        """for key, value in write_row.items():
            try:
                l = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                continue
            
            #if key == "keywords":
            #    l = l[0:2]
            
            try:
                value = " ".join(l)
            except TypeError:
                continue

            write_row.update({key:value})"""

        url = write_row["url"]
        if not url.startswith("http"):
            url = "https://" + url
        write_row["url_path"] = urlparse(url).path[1:]

        writer.writerow(write_row)
    
    os.rename("new.csv", FILE)