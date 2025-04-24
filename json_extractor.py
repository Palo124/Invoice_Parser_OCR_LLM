import re
import json
from typing import Union

def convert_to_float(s: str) -> float:
    """
    Convert numeric string to float handling both comma as a thousand separator
    and comma as a decimal separator.
    """
    s = s.strip()
    # If there's one comma and no period, assume it's a decimal separator.
    if s.count(',') == 1 and '.' not in s:
        s = s.replace(',', '.')
    else:
        s = s.replace(',', '')
    return float(s)

class JSONExtractor:
    @staticmethod
    def clean_json_string(json_str: str) -> str:
        """
        Clean the JSON string by removing common issues like trailing commas and unescaped control characters.
        """
        json_str = json_str.strip()
        # Remove trailing commas before a closing brace or bracket.
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        # Replace literal "\n" sequences with a space.
        json_str = json_str.replace("\\n", " ")
        # Remove unescaped control characters (newlines, tabs, etc.).
        json_str = re.sub(r'[\x00-\x1F]+', ' ', json_str)
        return json_str


    # Updated sanitize_data method:
    @staticmethod
    def sanitize_data(data):
        """
        Recursively sanitize JSON data, converting strings with percentage signs or numeric strings
        to floats where appropriate.
        """
        if isinstance(data, dict):
            return { key: JSONExtractor.sanitize_data(val) for key, val in data.items() }
        elif isinstance(data, list):
            return [JSONExtractor.sanitize_data(item) for item in data]
        elif isinstance(data, str):
            if "%" in data:
                try:
                    return float(data.replace("%", "").strip())
                except ValueError:
                    return data
            else:
                try:
                    return convert_to_float(data)
                except ValueError:
                    return data
        else:
            return data

    @staticmethod
    def extract_json(text: str, return_string: bool = False, sanitize: bool = True) -> Union[dict, str]:
        """
        Extracts the first complete JSON object found in the text by ensuring balanced braces,
        taking into account markdown code blocks starting with ```json.
        
        Parameters:
            text (str): The text containing the JSON.
            return_string (bool): If True, returns a pretty-printed JSON string with unescaped characters.
                                  If False (default), returns the parsed dictionary.
            sanitize (bool): If True, sanitizes the JSON data by converting numeric strings into floats.
        
        Returns:
            Union[dict, str]: The parsed (and optionally sanitized) JSON object or a JSON string with original Unicode characters.
        """
        # Check if the text contains a markdown code block starting with ```json.
        if "```json" in text:
            pattern = r'```json\s*(\{.*?\})\s*```'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                raise ValueError("No valid JSON code block found in the text.")
        else:
            # Fallback: extract the first balanced JSON object.
            start = text.find('{')
            if start == -1:
                raise ValueError("No valid JSON block found in the text.")
            open_braces = 0
            end = None
            for idx in range(start, len(text)):
                char = text[idx]
                if char == '{':
                    open_braces += 1
                elif char == '}':
                    open_braces -= 1
                    if open_braces == 0:
                        end = idx + 1
                        break
            if end is None:
                raise ValueError("No matching closing brace found for JSON object.")
            json_str = text[start:end]
        
        # Clean the extracted JSON string.
        json_str = JSONExtractor.clean_json_string(json_str)

        try:
            parsed_json = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Debug print: show a snippet around the error.
            error_pos = e.pos
            snippet = json_str[max(0, error_pos-20):error_pos+20]
            raise ValueError(f"Error parsing JSON at position {error_pos}: {e.msg}. Snippet: {snippet}")
        
        if sanitize:
            parsed_json = JSONExtractor.sanitize_data(parsed_json)
        if return_string:
            return json.dumps(parsed_json, indent=4, ensure_ascii=False)
        return parsed_json
