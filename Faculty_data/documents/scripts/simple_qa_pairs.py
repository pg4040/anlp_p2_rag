import re

def extract_info(text):
    # Generic patterns for extracting information
    name_pattern = r"(\w+\s+\w+)(?=\n\s*[^,]+,)"
    position_pattern = r"([^,\n]+),\s*Language Technologies Institute"
    research_area_pattern = r"Research Area\n([^\n]+)"
    office_location_pattern = r"Contact\n([^\n]+)"
    contact_pattern = r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
    website_pattern = r"(https?://[^\s]+)"
    
    info = {
        "name": re.search(name_pattern, text).group(1) if re.search(name_pattern, text) else None,
        "position": re.search(position_pattern, text).group(1) if re.search(position_pattern, text) else None,
        "research_area": re.search(research_area_pattern, text).group(1) if re.search(research_area_pattern, text) else None,
        "office_location": re.search(office_location_pattern, text).group(1) if re.search(office_location_pattern, text) else None,
        "contact": re.search(contact_pattern, text).group(1) if re.search(contact_pattern, text) else None,
        "website": re.search(website_pattern, text).group(1) if re.search(website_pattern, text) else None,
    }
    
    return info

def generate_qa_pairs(info):
    qa_pairs = []
    if info['name'] and info['position']:
        qa_pairs.append((f"Who is {info['name']}?", f"{info['name']} is a {info['position']} at Carnegie Mellon University."))
    if info['research_area']:
        qa_pairs.append((f"What is {info['name']}'s research area?", f"The research area includes {info['research_area']}."))
    if info['office_location']:
        qa_pairs.append((f"Where is {info['name']}'s office located?", f"The office is located at {info['office_location']}."))
    if info['contact']:
        qa_pairs.append((f"How can someone contact {info['name']}?", f"They can be contacted via email at {info['contact']}."))
    if info['website']:
        qa_pairs.append((f"Does {info['name']} have a personal website?", f"Yes, the personal website can be found at {info['website']}."))

    return qa_pairs


if __name__ == '__main__':
    prof_name = 'bisk-yonatan'
    file_path = f"../{prof_name}.txt"  # Update this to your actual file path
    with open(file_path, "r") as file:
        text = file.read()

    info = extract_info(text)
    qa_pairs = generate_qa_pairs(info)

    for q, a in qa_pairs:
        print(f"Q: {q}\nA: {a}\n")

