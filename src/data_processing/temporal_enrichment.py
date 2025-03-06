# src/data_processing/temporal_enrichment.py

import json
import pandas as pd
import datetime
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import re

def extract_publication_date(url):
    """Extract publication date from a webpage or use archive.org to find first appearance"""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try common metadata patterns for dates
        meta_tags = [
            soup.find('meta', property='article:published_time'),
            soup.find('meta', property='og:published_time'),
            soup.find('meta', itemprop='datePublished'),
            soup.find('time'),
            soup.find(class_=re.compile('date|publish|time', re.I))
        ]
        
        for tag in meta_tags:
            if tag and tag.get('content', tag.get('datetime', None)):
                date_str = tag.get('content', tag.get('datetime', None))
                try:
                    return datetime.datetime.fromisoformat(date_str)
                except:
                    pass
        
        # If we can't find a date, return None
        return None
    except:
        return None

def enrich_dataset():
    """Add temporal information to the COVID-Fact dataset"""
    with open('../covidfact/COVIDFACT_dataset.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    
    enriched_data = []
    for item in tqdm(data):
        # Extract publication date
        publication_date = extract_publication_date(item['gold_source'])
        
        # If we couldn't find a date, try to estimate from content
        if not publication_date:
            # Look for date patterns in the evidence
            date_patterns = [
                r'(\d{1,2}\/\d{1,2}\/\d{2,4})',
                r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',
                r'(\d{4}-\d{2}-\d{2})'
            ]
            
            for evidence in item['evidence']:
                for pattern in date_patterns:
                    matches = re.findall(pattern, evidence)
                    if matches:
                        try:
                            # Try to parse the date
                            publication_date = datetime.datetime.strptime(matches[0], '%Y-%m-%d')
                            break
                        except:
                            continue
        
        # Add COVID timeline context
        pandemic_phase = get_pandemic_phase(publication_date)
        
        # Add the temporal information to the item
        item['temporal_info'] = {
            'publication_date': publication_date.isoformat() if publication_date else None,
            'pandemic_phase': pandemic_phase
        }
        
        enriched_data.append(item)
    
    # Save the enriched dataset
    with open('../data/processed/temporal_covidfact.jsonl', 'w') as f:
        for item in enriched_data:
            f.write(json.dumps(item) + '\n')

def get_pandemic_phase(date):
    """Determine the phase of the pandemic based on date"""
    if not date:
        return "unknown"
    
    # Define pandemic phases (simplified)
    phases = [
        (datetime.datetime(2019, 12, 1), datetime.datetime(2020, 3, 10), "early_outbreak"),
        (datetime.datetime(2020, 3, 11), datetime.datetime(2020, 12, 31), "first_wave"),
        (datetime.datetime(2021, 1, 1), datetime.datetime(2021, 6, 30), "vaccination_phase"),
        (datetime.datetime(2021, 7, 1), datetime.datetime(2022, 12, 31), "variant_phase")
    ]
    
    for start, end, phase in phases:
        if start <= date <= end:
            return phase
    
    return "unknown"

if __name__ == "__main__":
    enrich_dataset()