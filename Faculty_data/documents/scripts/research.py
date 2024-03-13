import requests
from doc_extractor import extract_url_list
import time

def fetch_open_access_papers(author_query):
    """
    Fetches titles and years of open access papers published in 2023 for a given author.

    :param author_query: The query string to search for the author, e.g., "bisk yonatan".
    :return: A list of papers that match the criteria.
    """
    # Define the base URL for the author search
    url = f"https://api.semanticscholar.org/graph/v1/author/search?query={author_query}&fields=name,aliases,url,papers.title,papers.year,papers.paperId,papers.isOpenAccess"

    try:
        # Perform the search query
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()
        # Placeholder for additional filtering by open access and year
        papers_of_interest = []
        paperIdsList = []
        papers = []
        # Assume each author has papers listed in the response
        for author in data.get('data', []):
            for paper in author.get('papers', []):
                if paper['year'] == 2023 and paper['isOpenAccess']==True:
                    # For each paper of 2023, fetch more details to check open access status
                    paperIdsList.append(paper['paperId'])
                    papers.append(paper)
        print(author_query)
        print(len(paperIdsList))        
        write_author_papers_file(author_query, papers)
        papers_metadata = fetch_papers_metadata(paperIdsList)
        write_paper_metadata_files(author_query, papers_metadata)
        #TODO: extract the paper also
        return papers_of_interest

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []

def write_author_papers_file(author_query, papers):
    start = f"List of 2023 Open Access papers by {author_query} are:\n"
    for paper in papers:
        start+= paper['title']+'\n'
    with open(f'../{author_query}_papers.txt', 'w') as file:
        file.write(start)
    
def sanitize_filename(filename):
    return "".join([c if c.isalnum() or c in " .-_" else "_" for c in filename])

def write_paper_metadata_files(author_name, papers_metadata):
    #papers_metadata is a list of metadata dictionaries.
    #For each metadata dictionary, write one file with the content
    #TODO: write a file <author_name>_<paper_title>_metadata.txt
    #Write content - Faculty Name  - <authorname>,  metadata-<metadata> parsed into human format'
    print(author_name, len(papers_metadata))
    for paper in papers_metadata:
        paper_title = paper.get('paperId', 'No paperId available')
        filename = f'{author_name}_{paper_title}_metadata.txt' 
        filename = '../'+sanitize_filename(filename)
        with open(filename, 'w', encoding='utf-8') as file:
            # Write the author name at the top
            file.write(f"Faculty Name: {author_name}\n")
            # Iterate over the metadata dictionary and write each key-value pair
            file.write("Metadata:\n")
            for key, value in paper.items():
                # For authors, since it's a list, join the names with commas
                if key == 'authors':
                    file.write(f"{key.capitalize()}: {', '.join(value)}\n")
                else:
                    file.write(f"{key.capitalize()}: {value}\n")
            print(filename)
        
def fetch_papers_metadata(ids_list):
    """
    Fetches metadata for a list of paper IDs from Semantic Scholar.
    
    :param ids_list: A list of Semantic Scholar paper IDs.
    :return: A list of dictionaries, each containing metadata for one of the requested papers.
    """
    url = 'https://api.semanticscholar.org/graph/v1/paper/batch'
    headers = {'Content-Type': 'application/json'}
    params = {'fields': 'title,year,abstract,authors,venue,tldr'}
    data = {"ids": ids_list}
    
    try:
        time.sleep(3)
        response = requests.post(url, params=params, json=data, headers=headers)
        response.raise_for_status()  # Check for HTTP request errors
        papers_data = response.json()
        papers_metadata = []
        for paper in papers_data:
            metadata = {
                'paperId': paper.get('paperId', 'No paperId Available'),
                'title': paper.get('title', 'No Title Available'),
                'year': paper.get('year', 'No Year Provided'),
                'abstract': paper.get('abstract', 'No Abstract Available'),
                'authors': [author.get('name') for author in paper.get('authors', [])] if paper.get('authors') else [],
                'venue': paper.get('venue', 'No Venue Provided'),
                'tldr': paper.get('tldr', 'No TLDR Available'),  # Assuming 'tldr' is available
            }
            papers_metadata.append(metadata)
        
        return papers_metadata
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == '__main__':
    #Get list of authors
    prof_names_list = extract_url_list()
    for prof_name in prof_names_list:
        if prof_name == 'bio':
            prof_name = 'li-lei'
        elif prof_name != 'neubig-graham':
            continue
        split = prof_name.split('-')
        if len(split) > 2:
            prof_name_spaced = ' '.join(split)
        else:
            prof_name_spaced = ' '.join(reversed(split))
        print(prof_name_spaced)
        fetch_open_access_papers(prof_name_spaced)
