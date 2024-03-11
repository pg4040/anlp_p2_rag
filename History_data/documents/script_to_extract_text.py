from bs4 import BeautifulSoup
import requests
import json
import os
import re
import fitz
# References
# https://beautiful-soup-4.readthedocs.io/en/latest/: Basic syntax
# ChatGPT 3.5
# Copy of /data_processing/scraping_and_beautify.py

"""
This function extracts the text and metadata part from a url and puts into text file.
Params:
    url = url of website
    id = document id you want assigned to the document
    topic_category = the category of the topic
    further_processing = function that performs further processing of text
Ensures:
    Creates a <topicCategory_docId.txt> document for the text of 
    the webpage and a <topicCategory_docId_metadata.txt> that contains the metadata and
    topic_category in /data/documents/.
Returns:
    Nothing
"""
def html_to_text(url, id, topic_category, further_processing):
    # Get response from url
    response = requests.get(url)

    if response.status_code==200:
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        base_dir = '../History_data/documents'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # Create doc txt file
        doc_file_name = os.path.join(base_dir, f'{topic_category}_{id}.txt')
        text = soup.get_text().strip()
        text = re.sub('\s+', ' ', text)
        # print("text: ", text)

        # text_elements = soup.find_all(text=True)

        # text = ' '.join(text_elements)

        # text = re.sub('\s+', ' ', text).strip()

        processed_text = further_processing(text)
        with open(doc_file_name,"w", encoding="utf-8") as file:
            file.write(processed_text)
        
        # Get metadata
       
        metadata_json = {}
        metadata_json['title']=soup.title.text
        metadata_tags = soup.find_all('meta')

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
            
def html_file_to_text(webpage, id, topic_category, further_processing):
    # Get response from url
    with open(webpage, 'r', encoding='utf-8') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    base_dir = '../History_data/documents'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

        # Create doc txt file
    doc_file_name = os.path.join(base_dir, f'{topic_category}_{id}.txt')
    # text = soup.get_text().strip()
    # text = re.sub('\s+', ' ', text)
    # print("text: ", text)

    # text_elements = soup.find_all(text=True)

    # text = ' '.join(text_elements)

    # text = re.sub('\s+', ' ', text).strip()
    
    article_body_div = soup.find('div', class_='article-body')

    if article_body_div:
        article_body_text = article_body_div.get_text(separator='\n')

    text = '\n'.join(line.strip() for line in article_body_text.splitlines() if line.strip())

    processed_text = further_processing(text)
    with open(doc_file_name,"w", encoding="utf-8") as file:
        file.write(processed_text)
        
    metadata_json = {}
    metadata_json['title']=soup.title.text
    metadata_tags = soup.find_all('meta')
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

def pdf_to_text(page, id, topic_category, further_processing):
    try:
        # Open the PDF file
        with fitz.open(page) as pdf_doc:
            text = ""

            # Iterate over pages
            for page_num in range(pdf_doc.page_count):
                # Get the page
                page = pdf_doc[page_num]

                # Extract text from the page
                text += page.get_text()

            # Specify the base directory
            base_dir = '../History_data/documents'

            # Create the base directory if it doesn't exist
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            # Create the doc txt file
            doc_file_name = os.path.join(base_dir, f'{topic_category}_{id}.txt')

            # Process the text
            processed_text = further_processing(text)

            # Write the processed text to the doc txt file
            with open(doc_file_name, "w", encoding="utf-8") as file:
                file.write(processed_text)

            # Create metadata dictionary
            metadata_json = {'title': 'CMU fact sheet'}

            # Create the metadata file
            meta_file_name = os.path.join(base_dir, f'{topic_category}_{id}_metadata.txt')

            # Write metadata to the metadata file
            with open(meta_file_name, "w", encoding="utf-8") as meta_file:
                json.dump(metadata_json, meta_file, indent=2)

                # Append the category information to the metadata file
                meta_file.write('\nTopic category is ' + topic_category)

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")


"""Add further processing as needed"""
def further_process(text):
    return text




# Uncomment below lines and run with required values
# url = 'https://www.cs.cmu.edu/scs25/25things'
# id = 'd400'
# topic_category = '25things'
# further_processing = further_process # Implement other processing 
#                       # functions as needed and pass them as parameters.

# url = 'https://www.cs.cmu.edu/scs25/history'
# id = 'd401'
# topic_category = 'history'
# further_processing = further_process # Implement other processing 
# #                       # functions as needed and pass them as parameters.

# webpage = 'Tartan Facts - Carnegie Mellon University Athletics.html'
# id = 'd404'
# topic_category = 'tartanfacts'
# further_processing = further_process

# webpage = 'About - Carnegie Mellon University Athletics.html'
# id = 'd405'
# topic_category = 'mascot'
# further_processing = further_process

# webpage = 'The Kiltie Band - Carnegie Mellon University Athletics.html'
# id = 'd406'
# topic_category = 'kiltieband'
# further_processing = further_process

# html_to_text(url, id, topic_category, further_processing)
# html_file_to_text(webpage, id, topic_category, further_processing)
# page = 'cmu_fact_sheet.pdf'
# id = 'd407'
# topic_category = 'fact_sheet'
# further_processing = further_process

#pdf_to_text(page, id , topic_category, further_processing)
    