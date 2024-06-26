import json

def get_unique_words_and_episodes(data):
    unique_words = set()
    word_episodes_map = {}

    for category, category_data in data.items():
        for word, word_data in category_data.items():
            for episode_name, global_time in zip(word_data['names'], word_data['global_times']):
                if word not in word_episodes_map:
                    word_episodes_map[word] = []

                if episode_name not in word_episodes_map[word]:
                    word_episodes_map[word].append((episode_name, global_time))

            unique_words.add(word)

    return unique_words, word_episodes_map


def get_episodes_for_words(unique_words, word_episodes_map, max_episodes_per_word=5):
    episodes_for_words = {}

    for word in unique_words:
        episodes_for_words[word] = []

        if word in word_episodes_map:
            episodes = word_episodes_map[word]
            unique_episodes = []

            for episode, time in sorted(episodes, key=lambda x: x[1]):
                if episode not in unique_episodes:
                    unique_episodes.append(episode)

                    if len(unique_episodes) >= max_episodes_per_word:
                        break

                    episodes_for_words[word].append((episode, time))

    return episodes_for_words


def get_episodes_to_download(episodes_for_words):
    episodes_to_download = set()
    for word, episodes in episodes_for_words.items():
        for episode, _ in episodes:
            episodes_to_download.add(episode)
    return episodes_to_download


def find_overlaps(episodes_for_words):
    overlaps = {}
    for word, episodes in episodes_for_words.items():
        for episode, _ in episodes:
            if episode not in overlaps:
                overlaps[episode] = set()
            overlaps[episode].add(word)
    return overlaps


# Read data from file
with open('data\\dataset\\spottings\\spottings\\spottings\\verified_dict_spottings.json', 'r') as file:
    data = json.load(file)

# Process the data
unique_words, word_episodes_map = get_unique_words_and_episodes(data)
episodes_for_words = get_episodes_for_words(unique_words, word_episodes_map)

# Get episodes to download
episodes_to_download = get_episodes_to_download(episodes_for_words)

# Find overlaps between episodes for different words
overlaps = find_overlaps(episodes_for_words)

# Printing the episodes to download
print("Episodes to download:")
for episode in episodes_to_download:
    print(f"- Episode: {episode}")

print(episodes_to_download)

# Printing the overlaps
print("\nOverlaps between episodes for different words:")
for episode, words in overlaps.items():
    print(f"- Episode: {episode}, Words: {words}")
