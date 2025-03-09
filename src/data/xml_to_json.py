from xml.etree import ElementTree as ET
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_xml_to_json(xml_file, json_file):
    """Convert Wikipedia XML dump to JSON format"""
    logger.info(f"Converting {xml_file} to JSON...")
    
    articles = []
    context = ET.iterparse(xml_file, events=('end',))
    
    for event, elem in context:
        if elem.tag.endswith('page'):
            # Extract text content from the page
            text = None
            title = None
            for child in elem:
                if child.tag.endswith('revision'):
                    for rev_child in child:
                        if rev_child.tag.endswith('text'):
                            text = rev_child.text
                elif child.tag.endswith('title'):
                    title = child.text
            
            if text and title:
                articles.append({
                    'title': title,
                    'text': text
                })
            
            # Clear element to save memory
            elem.clear()
    
    logger.info(f"Writing {len(articles)} articles to {json_file}")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input XML file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()
    
    convert_xml_to_json(args.input, args.output) 