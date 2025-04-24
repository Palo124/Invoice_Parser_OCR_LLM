import json
from collections import Counter

def normalize_value(value):
    """
    Convert unhashable values (list/dict) into a JSON string representation.
    For all other values, return them as is.
    """
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(value)
    return value

def denormalize_value(value):
    """
    If the normalized value is a JSON string representing a list or dict,
    convert it back to its original type.
    Otherwise, return the value unchanged.
    """
    if isinstance(value, str):
        stripped = value.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
    return value

def triple_modular_redundancy(dict1: dict, dict2: dict, dict3: dict) -> dict:
    """
    Perform triple modular redundancy (TMR) on three dictionaries.
    For each key, if at least two values (from the three dictionaries) are equal,
    that value is chosen. If no majority exists, the value from the first dictionary is used.
    
    This function handles unhashable types (like lists/dicts) by normalizing them
    to a JSON string for comparison.
    """
    final = {}
    all_keys = set(dict1.keys()) | set(dict2.keys()) | set(dict3.keys())
    
    for key in all_keys:
        # Extract values (using None if the key is missing in any dictionary).
        values = [dict1.get(key), dict2.get(key), dict3.get(key)]
        # Convert values to a normalized (hashable) form.
        norm_values = [normalize_value(v) for v in values]
        
        # Count the occurrences of each normalized value.
        counts = Counter(norm_values)
        majority_norm = None
        for norm_val, freq in counts.items():
            if freq >= 2:
                majority_norm = norm_val
                break
        
        # If no majority exists, default to the value from the first dictionary.
        if majority_norm is None:
            majority_norm = norm_values[0]
        
        # Convert the normalized value back to its original type.
        final[key] = denormalize_value(majority_norm)
    
    return final

# Optional simple test when running this module directly.
if __name__ == '__main__':
    dict_a = {
        "Invoice NO": "16FVO117",
        "Company": "BHIT CZ s.r.o",
        "Description": "Školení",
        "Item Details": [
            {"Popis": "Školení", "Množství": 780.00},
            {"Popis": "Školení", "Množství": 560.00}
        ]
    }
    dict_b = {
        "Invoice NO": "16FVO117",
        "Company": "BHIT CZ s.r.o",
        "Description": "Školení",
        "Item Details": [
            {"Popis": "Školení", "Množství": 780.00},
            {"Popis": "Školení", "Množství": 560.00}
        ]
    }
    dict_c = {
        "Invoice NO": "16FVO117",
        "Company": "BHIT CZ s.r.o",
        "Description": "Other",  # Different description
        "Item Details": [
            {"Popis": "Školení", "Množství": 780.00},
            {"Popis": "Školení", "Množství": 560.00}
        ]
    }
    final_dict = triple_modular_redundancy(dict_a, dict_b, dict_c)
    print("Final merged dict:", json.dumps(final_dict, indent=4, ensure_ascii=False))
