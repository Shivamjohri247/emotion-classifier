import pandas as pd

def create_test_data():
    test_data = {
        'text': [
            "I am happy",
            "I am sad",
            "I am angry",
            "I am scared"
        ],
        'Emotion': ['joy', 'sadness', 'anger', 'fear']
    }
    
    df = pd.DataFrame(test_data)
    df.to_csv('tests/data/test_data.csv', index=False)

if __name__ == "__main__":
    create_test_data() 