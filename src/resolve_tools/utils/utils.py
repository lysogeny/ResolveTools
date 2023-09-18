from datetime import datetime

def printwtime(text):
    print(datetime.now().strftime("%H:%M:%S"),"-",text)