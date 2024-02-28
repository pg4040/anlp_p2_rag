from bs4 import BeautifulSoup
import requests
import json
import os

# References
# https://beautiful-soup-4.readthedocs.io/en/latest/: Basic syntax
# ChatGPT 3.5: Metadata extraction

"""
This function extracts the text and metadata part from a url and puts into text file.
Params:
    url = url of website
    id = document id you want assigned to the document
    topic_category = the category of the topic
Ensures:
    Creates a <topicCategory_docId.txt> document for the text of 
    the webpage and a <topicCategory_docId_metadata.txt> that contains the metadata and
    topic_category in /data/documents/.
Returns:
    Nothing
"""
def html_to_text(url, id, topic_category):
    # Get response from url
    response = requests.get(url)
    if response.status_code==200:
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        base_dir = '../data/documents/'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # Create doc txt file
        doc_file_name = os.path.join(base_dir, f'{topic_category}_{id}.txt')
        text = soup.get_text().strip()
        processed_text = further_process(text)
        with open(doc_file_name,"w", encoding="utf-8") as file:
            file.write(processed_text)
        
        # Get metadata
        metadata_tags = soup.find_all('meta')
        metadata_json = {}
        for tag in metadata_tags:
            name = tag.get('name')
            if name:
                metadata_json[name] = tag.get('content')
            else:
                property_attr = tag.get('property')
                if property_attr:
                    metadata_json[property_attr] = tag.get('content')
                    
        # Create metadata txt file
        meta_file_name = os.path.join(base_dir, f'{topic_category}_{id}_metadata.txt')
        with open(meta_file_name, "w", encoding="utf-8") as meta_file:
            json.dump(metadata_json, meta_file, indent=2)
            # Append the category information to the metadata file
            meta_file.write('\nTopic category is ' + topic_category)

"""Add further processing as needed"""
def further_process(text):
    return text

# Uncomment below lines and run with required values
# url = 'https://lti.cmu.edu/'
# id = 'd1'
# topic_category = 'lti'
# html_to_text(url, id, topic_category)
    
    