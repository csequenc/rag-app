import json

def load_gold_dataset():
    
    try:
        with open("evaluation/gold_dataset.json", "r") as file:
            gold_dataset = json.load(file)
        return gold_dataset
    
    except FileNotFoundError:
        print("Gold dataset file not found.")
        return None
    
    except json.JSONDecodeError:
        print("Error decoding JSON from gold dataset file.")
        return None
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    