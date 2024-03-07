import requests
from bs4 import BeautifulSoup
def write_file(prof_name):
    # URL of the faculty webpage
    url = f'https://lti.cs.cmu.edu/people/faculty/{prof_name}.html'
    doc_name = f'../{prof_name}.txt'
    # Send a GET request to the webpage
    response = requests.get(url)

    # Initialize a variable to hold the extracted data
    extracted_data = ""

    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract and append title tag
        title = soup.find('title').get_text()
        extracted_data += f"Title: {title}\n\n"
        
        # Extract and append all meta tags
        meta_tags = soup.find_all('meta')
        extracted_data += "Meta Tags:\n"
        for meta in meta_tags:
            extracted_data += str(meta) + "\n"
        extracted_data += "\n"
        
        # Extract and append all text within <div class='content'>
        content_div = soup.find('div', class_='content')
        if content_div:
            extracted_data += "Content:\n" + content_div.get_text(separator='\n', strip=True) + "\n\n"
        
        # Extract and append hrefs for email addresses or websites
        extracted_data += "Links:\n"
        for a_tag in content_div.find_all('a', href=True):
            href = a_tag['href']
            if "mailto:" in href or "http" in href:
                extracted_data += href + "\n"

        # Save the extracted data to a file
        with open(doc_name, 'w', encoding='utf-8') as file:
            file.write(extracted_data)
        print(f"Data extracted and saved to {doc_name}")
    else:
        print("Failed to retrieve the webpage. Status code:", response.status_code)

def extract_url_list():
    urls = set()
    filename = '../faculty_details.txt'
    try:
        # Open the file containing the faculty details
        with open(filename, 'r', encoding='utf-8') as file:
            # Read the file line by line
            for line in file:
                # Check if the line contains 'Links:'
                if line.startswith("Links:"):
                    # Extract URLs from the line, assuming they are comma-separated
                    extracted_urls = line.split("Links:")[1].strip().split(", ")
                    # Extend our list of URLs with the extracted URLs
                    for url in extracted_urls:
                        urls.add(url.split('.html')[0])
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return urls

if __name__ == '__main__':
    prof_name_list = extract_url_list()
    for prof_name in prof_name_list:
        write_file(prof_name)
