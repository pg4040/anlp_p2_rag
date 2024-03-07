import requests
from bs4 import BeautifulSoup

# The URL from which we're scraping data
url = 'https://lti.cs.cmu.edu/people/faculty/index.html'

# Sending a GET request to the webpage
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parsing the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extracting title and meta tags
    title = soup.find('title').get_text()
    meta_tags = [meta.attrs for meta in soup.find_all('meta')]
    
    # Extracting faculty information
    faculty_bios = soup.find('div', {'class': 'section-id', 'data-filter-id': 'bio-index-1', 'id': 'faculty-bios'})

    #faculty_bios = soup.find('div', id='faculty-bios')
    faculty_details = []
    faculty_bios_list = faculty_bios.find_all('div', class_='filterable')
    # Loop through each faculty member entry
    for faculty in faculty_bios_list:
        # Extract name, which is within a <h2> tag and the associated <a> tag
        name = faculty.find('h2').get_text(strip=True)
        # Extract designation, which is within a <h3> tag
        designation = faculty.find('h3').get_text(strip=True) if faculty.find('h3') else 'No designation provided'
        # Extract links to profile and emails, which are within <a> tags with href attribute
        links = [a['href'] for a in faculty.find_all('a', href=True)]
        
        # Add the extracted information to our list
        faculty_details.append({
            'name': name,
            'designation': designation,
            'links': links
        })

    # Now, we'll write this information to a file
    with open('faculty_details.txt', 'w', encoding='utf-8') as file:
        for faculty in faculty_details:
            file.write(f"Name: {faculty['name']}\n")
            file.write(f"Designation: {faculty['designation']}\n")
            file.write(f"Links: {', '.join(faculty['links'])}\n")
            file.write("\n")  # Adding a new line for readability between entries

    print("Faculty details have been extracted and saved to faculty_details.txt.")
