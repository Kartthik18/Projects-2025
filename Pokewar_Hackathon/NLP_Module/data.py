# data.py
import json, random
from typing import List, Dict
from config import CLASSES, POKEMON_SYNONYMS, ACTION_WORDS, NEGATION_WORDS, TACTICAL_FILLER

class PromptProcessor:
    """Processes and generates training data for the NeoBERT classifier."""

    def __init__(self):
        self.filler_phrases = TACTICAL_FILLER

    # -------- Load original prompts --------
    def load_train_prompts(self, path: str) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed = []
        for item in data:
            prompt = item['prompt']
            target = (
                prompt.split(": ")[1].strip().lower()
                if ": " in prompt else prompt.split(" ")[1].strip().lower()
            )
            if target in CLASSES:
                processed.append({
                    "prompt": prompt,
                    "target": target,
                    "image_id": item.get("image_id", "")
                })
        return processed

    # -------- Synthetic generators (ALL types) --------
    def generate_synthetic_prompts(self, num_prompts: int = 10000) -> List[Dict]:
        """Balanced classes; diversified strategies."""
        print(f"🧠 Generating {num_prompts} diverse tactical prompts...")
        strategies = [
            self._gen_tactical_report,
            self._gen_long_tactical_with_distractors,
            self._gen_buried_instruction,
            self._gen_multiple_mentions_with_target,
            self._gen_ambiguous_prompt,
            self._gen_negated_prompt,
            self._gen_instruction_emphasis,
            self._gen_many_distractors_one_target,
            self._gen_complex_scenario,
        ]
        per_class = num_prompts // len(CLASSES)
        prompts = []
        for target in CLASSES:
            for _ in range(per_class):
                prompts.append(random.choice(strategies)(target))
        while len(prompts) < num_prompts:
            prompts.append(random.choice(strategies)(random.choice(CLASSES)))
        random.shuffle(prompts)
        print(f"✅ Generated {len(prompts)} prompts.")
        return prompts

    # -------- Helpers --------
    def _get_synonyms(self, pokemon: str) -> List[str]:
        return [k for k, v in POKEMON_SYNONYMS.items() if v == pokemon] + [pokemon]

    def _get_other_pokemon_mentions(self, target_pokemon: str) -> List[str]:
        others = [p for p in CLASSES if p != target_pokemon]
        mentions = []
        for other in others:
            syns = self._get_synonyms(other)
            if syns:
                mentions.extend(random.sample(syns, min(3, len(syns))))
        return mentions

    # -------- Strategies (same coverage as your original) --------
    def _gen_tactical_report(self, target: str) -> Dict:
        alias = random.choice(self._get_synonyms(target))
        action = random.choice(ACTION_WORDS)
        filler = random.choice(self.filler_phrases)
        return {"prompt": f"{filler}. Mission objective: {action} the {alias}. Maintain operational secrecy.",
                "target": target}

    def _gen_long_tactical_with_distractors(self, target: str) -> Dict:
        alias = random.choice(self._get_synonyms(target))
        action = random.choice(ACTION_WORDS)
        parts = ["HQ REPORT: Situation analysis regarding unusual activity in this operational zone."]
        for _ in range(random.randint(6, 10)):
            parts.append(random.choice(self.filler_phrases))
            if random.random() < 0.7:
                d = random.choice(self._get_other_pokemon_mentions(target))
                parts.append(f"Scouts described sightings of {d} moving in small clusters.")
        parts.insert(len(parts)//2, f"Priority: {action} the {alias} groups and prevent them from regrouping.")
        parts.append("Maintain operational secrecy. HQ will expect a full after-action report.")
        return {"prompt": " ".join(parts), "target": target}

    def _gen_buried_instruction(self, target: str) -> Dict:
        alias = random.choice(self._get_synonyms(target))
        action = random.choice(ACTION_WORDS)
        filler_text = " ".join([random.choice(self.filler_phrases) for _ in range(6)])
        instruction = f"After all that analysis, the final order from high command is to {action} the {alias}."
        more_filler = " ".join([random.choice(self.filler_phrases) for _ in range(4)])
        return {"prompt": f"{filler_text} {instruction} {more_filler}", "target": target}

    def _gen_multiple_mentions_with_target(self, target: str) -> Dict:
        alias = random.choice(self._get_synonyms(target))
        action = random.choice(ACTION_WORDS)
        distractors = self._get_other_pokemon_mentions(target)
        parts = ["HQ REPORT: Multiple contacts detected in the operational zone."]
        for d in distractors[:2]:
            parts.append(f"Additional activity has been noted from {d} groups nearby, though they do not appear hostile at present.")
        parts.extend([random.choice(self.filler_phrases) for _ in range(2)])
        parts.append(f"Directives: {action} all {alias} in the AO and maintain perimeter security.")
        parts.append("Maintain operational secrecy. HQ will expect a full after-action report.")
        return {"prompt": " ".join(parts), "target": target}

    def _gen_ambiguous_prompt(self, target: str) -> Dict:
        alias = random.choice(self._get_synonyms(target))
        action = random.choice(ACTION_WORDS)
        distractors = self._get_other_pokemon_mentions(target)
        parts = ["HQ REPORT: Multiple high-priority targets detected in the operational zone."]
        for d in distractors[:2]:
            a2 = random.choice(ACTION_WORDS)
            parts.append(f"Priority: {a2} all {d} groups in sector 7.")
        parts.append(f"Secondary objective: {action} any {alias} encountered in the area.")
        parts.append("Note: Target priorities may change based on real-time intelligence.")
        parts.append("Maintain operational secrecy. HQ will expect a full after-action report.")
        return {"prompt": " ".join(parts), "target": target}

    def _gen_negated_prompt(self, target: str) -> Dict:
        alias = random.choice(self._get_synonyms(target))
        action = random.choice(ACTION_WORDS)
        distractors = self._get_other_pokemon_mentions(target)
        parts = ["HQ REPORT: Complex tactical situation developing in the operational zone."]
        for d in distractors[:2]:
            neg = random.choice(NEGATION_WORDS)
            a2 = random.choice(ACTION_WORDS)
            parts.append(f"Important: {neg} {a2} {d} units; they are not hostile.")
        parts.append(f"Primary directive: {action} all {alias} forces on sight.")
        parts.append("Maintain operational secrecy. HQ will expect a full after-action report.")
        return {"prompt": " ".join(parts), "target": target}

    def _gen_instruction_emphasis(self, target: str) -> Dict:
        alias = random.choice(self._get_synonyms(target))
        action = random.choice(ACTION_WORDS)
        distractors = self._get_other_pokemon_mentions(target)
        parts = ["HQ REPORT: Multiple contacts detected in the operational zone."]
        for _ in range(random.randint(8, 12)):
            parts.append(random.choice(self.filler_phrases))
            parts.append(f"Scouts described sightings of {random.choice(distractors)} moving in small clusters, often accompanied by subtle disruptions.")
        parts.append(f"CRITICAL ORDER: {action.upper()} ALL {alias.upper()} UNITS IMMEDIATELY.")
        for _ in range(random.randint(3, 5)):
            parts.append(random.choice(self.filler_phrases))
            parts.append(f"Additional activity has been noted from {random.choice(distractors)} groups nearby.")
        parts.append("Maintain operational secrecy. HQ will expect a full after-action report.")
        return {"prompt": " ".join(parts), "target": target}

    def _gen_many_distractors_one_target(self, target: str) -> Dict:
        alias = random.choice(self._get_synonyms(target))
        action = random.choice(ACTION_WORDS)
        distractors = self._get_other_pokemon_mentions(target)
        parts = ["HQ REPORT: Complex multi-species activity detected across the operational zone."]
        for _ in range(random.randint(10, 15)):
            parts.append(random.choice(self.filler_phrases))
            parts.append(f"Scouts described sightings of {random.choice(distractors)} moving in coordinated patterns.")
        parts.append(f"Despite multiple contacts, priority remains: {action} all {alias} encountered.")
        parts.append("Maintain operational secrecy. HQ will expect a full after-action report.")
        return {"prompt": " ".join(parts), "target": target}

    def _gen_complex_scenario(self, target: str) -> Dict:
        alias = random.choice(self._get_synonyms(target))
        action = random.choice(ACTION_WORDS)
        distractors = self._get_other_pokemon_mentions(target)
        parts = ["HQ REPORT: Multi-layered tactical scenario unfolding in the operational zone."]
        for _ in range(random.randint(8, 12)):
            parts.append(random.choice(self.filler_phrases))
            if random.random() < 0.6:
                parts.append(f"Scouts described sightings of {random.choice(distractors)} moving in small clusters, often accompanied by subtle disruptions in the environment.")
            if random.random() < 0.4:
                parts.append(f"Scouts described sightings of {random.choice(self._get_synonyms(target))} near the treeline.")
        for pos in [len(parts)//3, len(parts)//2, 2*len(parts)//3]:
            if random.random() < 0.7:
                parts.insert(pos, f"Priority: {action} the {alias} groups and prevent them from regrouping.")
        parts.append("Maintain operational secrecy. HQ will expect a full after-action report.")
        return {"prompt": " ".join(parts), "target": target}
