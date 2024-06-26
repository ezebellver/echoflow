import json
from collections import defaultdict

# Read the JSON data from the file
file_path = "C:\\Users\\bellv\\EchoFlowDataset\\dataset\\spottings\\spottings\\spottings\\verified_dict_spottings.json"
with open(file_path, 'r') as file:
    data = json.load(file)

# Initialize dictionaries to store words and episodes
words_data = {}
episodes_data = defaultdict(lambda: {"words": [], "timestamps": []})

# Iterate through the data to organize words and episodes
# print(data["test"].items())
for category, episodes in data["test"].items():
    words_data[category] = {"episodes": episodes["names"]}
    for i in range(len(episodes["names"])):
        episodes_data[episodes["names"][i]]["words"].append(category)
        episodes_data[episodes["names"][i]]["timestamps"].append(episodes["global_times"][i])

        # if not episodes_data[episodes["names"][i]]:
        #     episodes_data[episodes["names"][i]] = {"words": []}


# Write the words data to a JSON file
with open('../words_data.json', 'w') as file:
    json.dump(words_data, file, indent=4)

# Write the episodes data to a JSON file
with open('../episodes_data.json', 'w') as file:
    json.dump(episodes_data, file, indent=4)

# Get the top 10 episodes with the most words
top_10_episodes = sorted(episodes_data.items(), key=lambda x: len(x[1]["words"]), reverse=True)[:10]

# Keep only the top 10 episodes
episodes_data = dict(top_10_episodes)

# Print the names of the top 10 episodes
print("Top 10 episodes with the most words:")
for episode_name, episode_data in episodes_data.items():
    print(episode_name)

# Save the top 10 episodes data to a JSON file
with open('../top_10_episodes.json', 'w') as file:
    json.dump(episodes_data, file, indent=4)