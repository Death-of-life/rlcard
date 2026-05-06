"""Deck-pool helpers for PTCG training and evaluation."""

import json
import os
import random
from collections import defaultdict


def load_deck_pool(path, split=None, validate_paths=True):
    """Load deck entries from a JSON deck-pool file."""
    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    raw_decks = data.get("decks") if isinstance(data, dict) else data
    if not isinstance(raw_decks, list):
        raise ValueError("Deck pool must be a list or an object with a 'decks' list")

    decks = []
    for index, entry in enumerate(raw_decks):
        if not isinstance(entry, dict):
            raise ValueError(f"Deck entry {index} must be an object")
        missing = [key for key in ("name", "path") if not entry.get(key)]
        if missing:
            raise ValueError(f"Deck entry {index} missing required field(s): {', '.join(missing)}")
        deck = {
            "name": str(entry["name"]),
            "archetype": str(entry.get("archetype") or entry["name"]),
            "split": str(entry.get("split") or "train"),
            "path": str(entry["path"]),
        }
        if split and split != "all" and deck["split"] != split:
            continue
        if validate_paths and not os.path.isfile(deck["path"]):
            raise FileNotFoundError(f"Deck '{deck['name']}' not found: {deck['path']}")
        decks.append(deck)

    if not decks:
        selected = split or "all"
        raise ValueError(f"Deck pool has no decks for split '{selected}'")
    return decks


def sample_deck_pair(decks, rng=None, mode="archetype-balanced"):
    """Sample an ordered pair of deck entries."""
    rng = rng or random
    if not decks:
        raise ValueError("Cannot sample from an empty deck pool")
    if mode == "uniform":
        return rng.choice(decks), rng.choice(decks)
    if mode != "archetype-balanced":
        raise ValueError(f"Unknown deck sample mode: {mode}")

    grouped = defaultdict(list)
    for deck in decks:
        grouped[deck["archetype"]].append(deck)
    archetypes = sorted(grouped)

    def sample_one():
        archetype = rng.choice(archetypes)
        return rng.choice(grouped[archetype])

    return sample_one(), sample_one()


def deck_label(deck):
    return deck["name"]


def deck_path(deck):
    return deck["path"]


def deck_archetype(deck):
    return deck.get("archetype", deck["name"])


def deck_split(deck):
    return deck.get("split", "manual")


def manual_deck(name, path):
    return {
        "name": name,
        "archetype": name,
        "split": "manual",
        "path": path,
    }
