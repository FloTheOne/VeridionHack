# Your categorized data
noun_categories = [
    # Physical Objects (non-living)
    {"noun": "hammer", "physical": "yes", "type": "object"},
    {"noun": "rock", "physical": "yes", "type": "object"},
    {"noun": "candle", "physical": "yes", "type": "object"},
    {"noun": "paper", "physical": "yes", "type": "object"},
    {"noun": "gun", "physical": "yes", "type": "object"},
    {"noun": "ice", "physical": "yes", "type": "object"},
    {"noun": "volcano", "physical": "yes", "type": "object"},
    {"noun": "moon", "physical": "yes", "type": "object"},
    {"noun": "sword", "physical": "yes", "type": "object"},
    {"noun": "shield", "physical": "yes", "type": "object"},
    {"noun": "pebble", "physical": "yes", "type": "object"},
    {"noun": "robot", "physical": "yes", "type": "object"},
    {"noun": "laser", "physical": "yes", "type": "object"},
    {"noun": "nuclear bomb", "physical": "yes", "type": "object"},

    # Physical Living Beings
    {"noun": "lion", "physical": "yes", "type": "living_being"},
    {"noun": "bacteria", "physical": "yes", "type": "living_being"},
    {"noun": "virus", "physical": "yes", "type": "living_being"},
    {"noun": "whale", "physical": "yes", "type": "living_being"},
    {"noun": "human", "physical": "yes", "type": "living_being"},
    {"noun": "bird", "physical": "yes", "type": "living_being"},
    {"noun": "tree", "physical": "yes", "type": "living_being"},
    {"noun": "dog", "physical": "yes", "type": "living_being"},
    {"noun": "plant", "physical": "yes", "type": "living_being"},
    {"noun": "insect", "physical": "yes", "type": "living_being"},

    # Non-Physical Feelings/Sentiments
    {"noun": "love", "physical": "no", "type": "feeling"},
    {"noun": "happiness", "physical": "no", "type": "feeling"},
    {"noun": "sadness", "physical": "no", "type": "feeling"},
    {"noun": "anger", "physical": "no", "type": "feeling"},
    {"noun": "peace", "physical": "no", "type": "feeling"},
    {"noun": "hope", "physical": "no", "type": "feeling"},
    {"noun": "fear", "physical": "no", "type": "feeling"},
    {"noun": "excitement", "physical": "no", "type": "feeling"},
    {"noun": "envy", "physical": "no", "type": "feeling"},
    {"noun": "gratitude", "physical": "no", "type": "feeling"},

    # Non-Physical Events
    {"noun": "war", "physical": "no", "type": "event"},
    {"noun": "earthquake", "physical": "no", "type": "event"},
    {"noun": "tsunami", "physical": "no", "type": "event"},
    {"noun": "storm", "physical": "no", "type": "event"},
    {"noun": "pandemic", "physical": "no", "type": "event"},
    {"noun": "flood", "physical": "no", "type": "event"},
    {"noun": "explosion", "physical": "no", "type": "event"},
    {"noun": "revolution", "physical": "no", "type": "event"},
    {"noun": "party", "physical": "no", "type": "event"},
    {"noun": "conference", "physical": "no", "type": "event"}
]

# Decision tree function
def get_weakness(noun):
    noun = noun.lower()
    for item in noun_categories:
        if item["noun"].lower() == noun:
            if item["type"] == "object":
                return "flame"
            elif item["type"] == "living_being":
                return "gun"
            elif item["type"] == "feeling":
                return "time"
            elif item["type"] == "event":
                return None
    return None  # Unknown noun

# Examples:
print(get_weakness("soldier"))        # gun
print(get_weakness("peace"))       # time
print(get_weakness("volcano"))     # flame
print(get_weakness("earthquake"))  # None
print(get_weakness("hope"))        # time
