import random

def simulate_ocr_noise(text: str, noise_probability: float = 0.1) -> str:
    """
    Simulates OCR noise such as character substitution, insertion, deletion,
    and space removal to test semantic robustness.
    """
    if not text:
        return text
        
    chars = list(text)
    noisy_chars = []
    
    # Common OCR confusions
    confusions = {
        '0': ['O', 'Q', 'D'],
        'O': ['0', 'D', 'Q'],
        '1': ['l', 'I', 'i', '|'],
        'l': ['1', 'I', 'i', '|'],
        'I': ['1', 'l', 'i', '|'],
        '5': ['S'],
        'S': ['5'],
        '8': ['B'],
        'B': ['8'],
        '2': ['Z'],
        'Z': ['2'],
        'c': ['e', 'o'],
        'e': ['c', 'o'],
        'n': ['m', 'h'],
        'm': ['n', 'rn']
    }
    
    for c in chars:
        if random.random() < noise_probability:
            noise_type = random.choice(['substitute', 'delete', 'insert', 'drop_space'])
            
            if noise_type == 'substitute':
                if c in confusions:
                    noisy_chars.append(random.choice(confusions[c]))
                else:
                    # Random ASCII letter/digit substitution
                    noisy_chars.append(random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
            elif noise_type == 'delete':
                continue # drop character
            elif noise_type == 'insert':
                noisy_chars.append(c)
                noisy_chars.append(random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,'))
            elif noise_type == 'drop_space' and c == ' ':
                continue # drop space
            else:
                noisy_chars.append(c)
        else:
            noisy_chars.append(c)
            
    return "".join(noisy_chars)

if __name__ == "__main__":
    sample = "MCDONALDS BURGER 15.00"
    print("Original:", sample)
    print("Noisy (p=0.1):", simulate_ocr_noise(sample, 0.1))
    print("Noisy (p=0.3):", simulate_ocr_noise(sample, 0.3))
