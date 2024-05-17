import jsonlines
import pandas as pd

def ExtractUserContent(file_path):
    print("Extracting user content...")
    user_contents = []

    with jsonlines.open(file_path) as reader:
        for line in reader:
            messages = line.get('messages', [])
            for message in messages:
                if message.get('role') == 'user':
                    user_contents.append(message['content'])
    print("Extracted user content.")
    return user_contents

# file_path = 'data.jsonl'
# contents = extract_user_content(file_path)
# print(contents)

def GetLabel(file_path):
    print("Getting labels...")
    labels = []
    with jsonlines.open(file_path) as reader:
        for line in reader:
            messages = line.get('messages', [])
            for message in messages:
                if message.get('role') == 'assistant':
                    labels.append(message['content'])
    print("Got labels.")
    return labels