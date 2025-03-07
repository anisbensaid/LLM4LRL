import argparse
import logging
from pathlib import Path
from datasets import load_dataset
import requests
from tqdm import tqdm
from datetime import datetime
import json
from huggingface_hub import list_datasets
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, language: str, output_dir: str):
        self.language = language
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_latest_dump_date(self):
        """Get the latest available dump date for the language"""
        base_url = f"https://dumps.wikimedia.org/{self.language}wiki"
        
        try:
            # Get the list of available dumps
            response = requests.get(f"{base_url}/")
            if response.status_code != 200:
                raise Exception(f"Failed to get dump list: {response.status_code}")
            
            # Parse dates from the response
            dates = []
            for line in response.text.split('\n'):
                if 'href' in line and 'latest' not in line:
                    try:
                        date_str = line.split('href="')[1].split('/')[0]
                        date = datetime.strptime(date_str, "%Y%m%d")
                        dates.append(date_str)
                    except:
                        continue
            
            if not dates:
                raise Exception("No dumps found")
                
            # Return the most recent date
            return sorted(dates)[-1]
            
        except Exception as e:
            logger.error(f"Error getting latest dump date: {e}")
            return None

    def collect_oscar(self):
        """Collect data from OSCAR corpus"""
        logger.info(f"Collecting OSCAR data for language: {self.language}")
        
        try:
            dataset = load_dataset("oscar-corpus/OSCAR-2301", 
                                 language=self.language,
                                 trust_remote_code=True)
            output_file = self.output_dir / f"oscar_{self.language}.json"
            dataset.save_to_disk(output_file)
            logger.info(f"Saved OSCAR dataset to {output_file}")
            return dataset
        except Exception as e:
            logger.error(f"Error collecting OSCAR data: {e}")
            return None

    def process_wikipedia_dump(self, xml_file):
        """Process Wikipedia XML dump into text"""
        try:
            import bz2
            import xml.etree.ElementTree as ET
            from tqdm import tqdm
            
            logger.info("Processing Wikipedia XML dump...")
            output_file = self.output_dir / f"{self.language}wiki-articles.json"
            articles = []
            
            with bz2.open(xml_file, 'rt', encoding='utf-8') as f:
                # Read line by line to avoid loading entire file into memory
                page = []
                in_page = False
                for line in tqdm(f):
                    if '<page>' in line:
                        in_page = True
                    if in_page:
                        page.append(line)
                    if '</page>' in line:
                        in_page = False
                        # Process the page
                        page_xml = ''.join(page)
                        try:
                            root = ET.fromstring(page_xml)
                            title = root.find('.//title').text
                            text = root.find('.//text').text
                            if text and not any(x in title.lower() for x in ['template:', 'wikipedia:', 'file:']):
                                articles.append({
                                    'title': title,
                                    'text': text
                                })
                        except:
                            pass
                        page = []
            
            # Save as JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Extracted {len(articles)} articles to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error processing Wikipedia dump: {e}")
            return None

    def collect_wikipedia(self):
        """Collect data from Wikipedia dumps"""
        logger.info(f"Collecting Wikipedia data for language: {self.language}")
        
        try:
            # Download dump
            wiki_url = f"https://dumps.wikimedia.org/{self.language}wiki/latest/{self.language}wiki-latest-pages-articles.xml.bz2"
            output_file = self.output_dir / f"{self.language}wiki-latest-pages-articles.xml.bz2"
            
            logger.info(f"Downloading from: {wiki_url}")
            
            response = requests.get(wiki_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_file, 'wb') as f, tqdm(
                desc=output_file.name,
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
            
            logger.info(f"Downloaded Wikipedia dump to {output_file}")
            
            # Process the dump
            return self.process_wikipedia_dump(output_file)
            
        except Exception as e:
            logger.error(f"Error downloading Wikipedia dump: {e}")
            return None

    def collect_common_crawl(self):
        """Collect data from Common Crawl"""
        logger.info(f"Collecting Common Crawl data for language: {self.language}")
        try:
            # Using the c4 dataset which is derived from Common Crawl
            dataset = load_dataset("allenai/c4", f"en", trust_remote_code=True)
            output_file = self.output_dir / f"common_crawl_{self.language}.json"
            dataset.save_to_disk(output_file)
            logger.info(f"Saved Common Crawl dataset to {output_file}")
            return dataset
        except Exception as e:
            logger.error(f"Error collecting Common Crawl data: {e}")
            return None
     

    def collect_mc4(self):
        """Collect data from mC4 dataset"""
        logger.info(f"Collecting mC4 data for language: {self.language}")
        try:
            dataset = load_dataset("mc4", languages=[self.language], trust_remote_code=True)
            output_file = self.output_dir / f"mc4_{self.language}.json"
            dataset.save_to_disk(output_file)
            logger.info(f"Saved mC4 dataset to {output_file}")
            return dataset
        except Exception as e:
            logger.error(f"Error collecting mC4 data: {e}")
            return None

def process_myanmar_wiki_data(json_file_path):
    """
    Process the Myanmar Wikipedia JSON data to extract useful text content
    """
    with open(json_file_path, 'r', encoding='utf-8') as file:
        wiki_data = json.load(file)
    
    processed_data = []
    for article in wiki_data:
        # Extract clean text content
        text = article['text']
        # Remove wiki markup and special characters
        clean_text = clean_wiki_text(text)
        # Split into sentences
        sentences = split_into_sentences(clean_text)
        
        processed_data.extend(sentences)
    
    return processed_data

def clean_wiki_text(text):
    """
    Clean Wikipedia text by removing markup, references, etc.
    """
    # Remove wiki markup patterns
    text = re.sub(r'\{\{.*?\}\}', '', text)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'<.*?>', '', text)
    # Remove references
    text = re.sub(r'\[\d+\]', '', text)
    return text.strip()

def split_into_sentences(text):
    """
    Split Myanmar text into sentences
    """
    # Basic sentence splitting for Myanmar text
    # You might need to adjust these patterns based on Myanmar language rules
    sentences = re.split(r'[။၊]', text)
    return [s.strip() for s in sentences if s.strip()]

def save_processed_data(processed_data, output_file):
    """
    Save the processed data to a file
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(processed_data, file, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Collect data for low-resource language")
    parser.add_argument("--language", type=str, required=True, help="Language code (e.g., myw for Myanmar)")
    parser.add_argument("--source", type=str, default="wikipedia", choices=["wikipedia", "oscar", "common_crawl"],
                      help="Data source to collect from")
    parser.add_argument("--output_dir", type=str, default="data/raw",
                      help="Output directory for collected data")
    parser.add_argument("--list-languages", action="store_true",
                      help="List available Wikipedia language codes")
    parser.add_argument("--check-sources", action="store_true",
                      help="Check available data sources for the language")
    
    args = parser.parse_args()
    
    if args.list_languages:
        print("Fetching available Wikipedia language codes...")
        response = requests.get("https://dumps.wikimedia.org/backup-index.html")
        if response.status_code == 200:
            langs = [line.split('href="')[1].split('wiki/')[0] 
                    for line in response.text.split('\n') 
                    if 'href="' in line and 'wiki/' in line]
            print("\nAvailable language codes:")
            for lang in sorted(set(langs)):
                print(f"- {lang}")
        return

    if args.check_sources:
        print(f"\nChecking available data sources for language: {args.language}")
        # Check Wikipedia
        collector = DataCollector(args.language, args.output_dir)
        dump_date = collector.get_latest_dump_date()
        print(f"Wikipedia: {'Available' if dump_date else 'Not available'}")
        # Check OSCAR
        try:
            oscar_info = load_dataset("oscar-corpus/OSCAR-2301", language=args.language, trust_remote_code=True)
            print("OSCAR: Available")
        except:
            print("OSCAR: Not available")
        return

    collector = DataCollector(args.language, args.output_dir)
    
    if args.source == "wikipedia":
        collector.collect_wikipedia()
    elif args.source == "oscar":
        collector.collect_oscar()
    elif args.source == "common_crawl":
        collector.collect_common_crawl()

if __name__ == "__main__":
    main()

# Load the processed articles
with open('data/raw/mywiki-articles.json', 'r', encoding='utf-8') as f:
    articles = json.load(f)

# Basic statistics
print(f"Total articles: {len(articles)}")
print(f"Sample article title: {articles[0]['title']}")
print(f"Sample text preview: {articles[0]['text'][:200]}...")

# Count total words
total_words = sum(len(article['text'].split()) for article in articles)
print(f"Total words: {total_words}") 