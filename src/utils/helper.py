"""
Utility helper functions
"""

import json
import jsonlines
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSONL file
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries
    """
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for obj in reader:
            data.append(obj)
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save data to JSONL file
    
    Args:
        data: List of dictionaries
        file_path: Path to save file
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(file_path, 'w') as writer:
        writer.write_all(data)


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
    """Save data to JSON file"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def compute_hash(text: str) -> str:
    """Compute SHA256 hash of text"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def count_tokens(text: str) -> int:
    """
    Approximate token count (rough estimation)
    More accurate: use tiktoken library
    """
    return len(text.split())


def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def parse_answer(answer: str) -> List[str]:
    """
    Parse answer string to list of options
    
    Args:
        answer: Answer string like "A", "A,B", "B,D"
        
    Returns:
        List of options like ["A"], ["A", "B"]
    """
    return [opt.strip() for opt in answer.split(',')]


def format_answer(options: List[str]) -> str:
    """
    Format list of options to answer string
    
    Args:
        options: List like ["A", "B"]
        
    Returns:
        String like "A,B"
    """
    return ','.join(sorted(options))