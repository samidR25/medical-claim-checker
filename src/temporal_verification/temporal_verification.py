"""
Module for checking the temporal validity of claims related to COVID-19.
Some claims might have been true at some point but became false later, or vice versa.
"""

import re
import datetime
from typing import Dict, Any, Optional, Tuple

def extract_temporal_info(claim: str) -> Dict[str, Any]:
    """
    Extract temporal information from a claim.
    
    Args:
        claim: The claim to analyze
        
    Returns:
        Dictionary with temporal information
    """
    info = {
        "has_temporal_marker": False,
        "time_expressions": [],
        "is_current": False,
        "is_past": False,
        "is_future": False,
        "specific_date": None
    }
    
    # Check for current time markers
    current_markers = ["now", "currently", "present", "today", "at this time"]
    if any(marker in claim.lower() for marker in current_markers):
        info["has_temporal_marker"] = True
        info["is_current"] = True
        info["time_expressions"].append("current")
    
    # Check for past time markers
    past_markers = ["previously", "earlier", "before", "past", "last year", "last month", "last week"]
    if any(marker in claim.lower() for marker in past_markers):
        info["has_temporal_marker"] = True
        info["is_past"] = True
        info["time_expressions"].append("past")
    
    # Check for future time markers
    future_markers = ["will", "future", "next", "upcoming", "soon"]
    if any(marker in claim.lower() for marker in future_markers):
        info["has_temporal_marker"] = True
        info["is_future"] = True
        info["time_expressions"].append("future")
    
    # Extract specific dates
    date_patterns = [
        # mm/dd/yyyy, mm-dd-yyyy
        r'\b(0?[1-9]|1[0-2])[/\-](0?[1-9]|[12][0-9]|3[01])[/\-](19|20)\d{2}\b',
        # Month name, day, year
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?,?\s+(19|20)\d{2}\b',
        # Year only
        r'\b(19|20)\d{2}\b'
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, claim, re.IGNORECASE)
        if matches:
            info["has_temporal_marker"] = True
            for match in matches:
                if isinstance(match, tuple):
                    # Handle tuple results (from capturing groups)
                    date_str = " ".join(str(m) for m in match if m)
                else:
                    date_str = match
                info["time_expressions"].append(date_str)
                # Try to parse the date
                parsed_date = _parse_date(date_str)
                if parsed_date and (info["specific_date"] is None or parsed_date > info["specific_date"]):
                    info["specific_date"] = parsed_date
    
    return info

def _parse_date(date_str: str) -> Optional[datetime.datetime]:
    """
    Try to parse a date string into a datetime object.
    
    Args:
        date_str: The date string to parse
        
    Returns:
        Datetime object or None if parsing fails
    """
    formats = [
        "%m/%d/%Y", "%m-%d-%Y",  # mm/dd/yyyy, mm-dd-yyyy
        "%B %d, %Y", "%B %d %Y",  # Month day, year
        "%Y"  # Year only
    ]
    
    for fmt in formats:
        try:
            return datetime.datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Try to extract year only
    year_match = re.search(r'(19|20)\d{2}', date_str)
    if year_match:
        try:
            return datetime.datetime(int(year_match.group(0)), 1, 1)
        except ValueError:
            pass
    
    return None

def check_temporal_validity(claim: str, 
                          counter_claim: str, 
                          reference_date: Optional[datetime.datetime] = None) -> Dict[str, Any]:
    """
    Check if a claim or counter-claim might be valid depending on the time.
    
    Args:
        claim: The original claim
        counter_claim: The generated counter-claim
        reference_date: Optional reference date (defaults to current date)
        
    Returns:
        Dictionary with temporal validity assessment
    """
    if reference_date is None:
        reference_date = datetime.datetime.now()
    
    # Extract temporal info
    claim_temporal = extract_temporal_info(claim)
    counter_temporal = extract_temporal_info(counter_claim)
    
    # COVID-19 timeline key dates (simplified)
    covid_timeline = {
        "first_reported": datetime.datetime(2019, 12, 31),
        "who_emergency": datetime.datetime(2020, 1, 30),
        "pandemic_declared": datetime.datetime(2020, 3, 11),
        "first_vaccine_approved": datetime.datetime(2020, 12, 11),
        "delta_variant": datetime.datetime(2021, 5, 1),
        "omicron_variant": datetime.datetime(2021, 11, 1)
    }
    
    # Check if the claims might both be true but at different times
    result = {
        "temporal_conflict": False,
        "claim_time_dependent": False,
        "counter_claim_time_dependent": False,
        "claim_validity_period": None,
        "counter_claim_validity_period": None,
        "explanation": None
    }
    
    # Detect claims about vaccine effectiveness (time-dependent)
    vaccine_patterns = [
        r'vaccine(s)?\s+(is|are|was|were)\s+(effective|ineffective)',
        r'vaccine(s)?\s+(does|do|did)\s+not\s+\w+',
        r'vaccine(s)?\s+(protect|prevents|protected|prevented)'
    ]
    
    for pattern in vaccine_patterns:
        if re.search(pattern, claim, re.IGNORECASE):
            result["claim_time_dependent"] = True
            # Vaccine statements could only be true after first vaccine approval
            result["claim_validity_period"] = (covid_timeline["first_vaccine_approved"], None)
            
        if re.search(pattern, counter_claim, re.IGNORECASE):
            result["counter_claim_time_dependent"] = True
            result["counter_claim_validity_period"] = (covid_timeline["first_vaccine_approved"], None)
    
    # Detect claims about variants (time-dependent)
    variant_patterns = {
        "delta": r'delta\s+variant',
        "omicron": r'omicron\s+variant'
    }
    
    for variant, pattern in variant_patterns.items():
        if re.search(pattern, claim, re.IGNORECASE):
            result["claim_time_dependent"] = True
            start_date = covid_timeline.get(f"{variant}_variant")
            if start_date:
                result["claim_validity_period"] = (start_date, None)
                
        if re.search(pattern, counter_claim, re.IGNORECASE):
            result["counter_claim_time_dependent"] = True
            start_date = covid_timeline.get(f"{variant}_variant")
            if start_date:
                result["counter_claim_validity_period"] = (start_date, None)
    
    # If both claims have different validity periods, they might not contradict
    if (result["claim_validity_period"] and result["counter_claim_validity_period"] and
        result["claim_validity_period"] != result["counter_claim_validity_period"]):
        result["temporal_conflict"] = True
        result["explanation"] = "The claim and counter-claim might both be valid but at different points in time."
    
    return result