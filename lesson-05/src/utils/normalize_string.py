import unicodedata
import re


def normalize_string(input_string):
    """
    Normalizes a string by:
    - Converting 'đ' to 'd'
    - Removing Vietnamese diacritics
    - Replacing special characters with underscores
    - Converting capital letters to lowercase

    Args:
        input_string: The string to normalize.

    Returns:
        The normalized string.
    """
    # Convert 'đ' and 'Đ' to 'd' and 'D' respectively
    input_string = input_string.replace('đ', 'd').replace('Đ', 'D')

    # Remove Vietnamese diacritics
    nfkd_form = unicodedata.normalize('NFKD', input_string)
    only_ascii = nfkd_form.encode('ascii', 'ignore').decode('utf-8')

    # Replace special characters with underscores. \W is the inverse of \w
    normalized_string = re.sub(r'\W+', '_', only_ascii)

    # Convert to lowercase
    normalized_string = normalized_string.lower()

    # Remove leading and trailing underscores
    normalized_string = normalized_string.strip('_')

    return normalized_string


