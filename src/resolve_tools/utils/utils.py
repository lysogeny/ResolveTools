from datetime import datetime


def printwtime(text):
    """Print a text with a timestamp"""
    print(datetime.now().strftime("%H:%M:%S"), "-", text)
