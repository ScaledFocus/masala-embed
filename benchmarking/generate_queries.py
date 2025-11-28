import random

moods = [
    "relaxing",
    "rainy-day",
    "comforting",
    "cozy",
    "quick",
    "spicy",
    "light",
    "rich",
    "tangy",
    "smoky",
    "creamy",
    "crunchy",
    "hearty",
    "buttery",
    "fragrant",
]
intents = ["Need a", "I want a", "Craving a", "Looking for a"]
cuisines = [
    "Italian",
    "Indian",
    "Mexican",
    "Thai",
    "Japanese",
    "French",
    "Mediterranean",
    "Korean",
    "Chinese",
    "Middle Eastern",
]
dishes = [
    "dosa",
    "pasta",
    "curry",
    "ramen",
    "taco",
    "salad",
    "wrap",
    "noodle",
    "steak",
    "sandwich",
    "biryani",
    "pizza",
    "dumpling",
]

with open("payloads.txt", "w") as f:
    for _ in range(3000):
        intent = random.choice(intents)
        mood_or_sensory = random.choice(moods)
        cuisine = random.choice(cuisines)
        dish = random.choice(dishes)
        query = f"{intent} {mood_or_sensory} {cuisine} {dish}"
        f.write(query + "\n")

print("Generated 3,000-line food queries dataset in 'payloads.txt'")
