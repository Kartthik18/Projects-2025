# config.py

CLASSES = ["pikachu", "charizard", "bulbasaur", "mewtwo"]
label2id = {c: i for i, c in enumerate(CLASSES)}
id2label = {i: c for c, i in label2id.items()}

POKEMON_SYNONYMS = {
    "electric rat": "pikachu", "yellow mouse": "pikachu", "electric mouse": "pikachu",
    "fire-breathing lizard": "charizard", "orange dragon": "charizard", "fire dragon": "charizard",
    "seed pokemon": "bulbasaur", "plant toad": "bulbasaur", "grass frog": "bulbasaur",
    "genetic pokemon": "mewtwo", "psychic cat": "mewtwo", "clone pokemon": "mewtwo",
    # (… keep full synonym dict here …)
}

ACTION_WORDS = ["kill", "eliminate", "destroy", "attack", "neutralize", "defeat", "obliterate"]
NEGATION_WORDS = ["not", "don't", "avoid", "never", "spare", "protect", "defend", "guard"]

TACTICAL_FILLER = [
    "HQ REPORT: Situation analysis regarding unusual activity",
    "Scouts described sightings of",
    "Maintain operational secrecy. HQ will expect a full after-action report.",
    # (… keep rest of filler here …)
]
