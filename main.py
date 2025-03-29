import requests
from time import sleep
import random

from Categorize_with_glove import load_resources, get_best_word

# API settings
host = "http://172.18.4.158:8000"
post_url = f"{host}/submit-word"
get_url = f"{host}/get-word"
status_url = f"{host}/status"

NUM_ROUNDS = 5

# Load resources only once at the beginning
glove, df_excel, df = load_resources()

def what_beats(word):
    sleep(random.randint(1, 3))  # Simulează un delay
    best_word = get_best_word(word, glove, df_excel, df)
    print(f"[Decision] Weakness for '{word}': {best_word}")

    # Caută rândul care corespunde cuvântului
    matched = df_excel[df_excel["Word"] == best_word.lower()]
    
    if not matched.empty:
        return int(matched.iloc[0]["#"])  # Folosește coloana "#" ca ID

    # Dacă nu se găsește nimic, alege un fallback safe
    return random.randint(23, 50)

def play_game(player_id):
    print(f"\n=== Starting game as player: {player_id} ===")

    for round_id in range(1, NUM_ROUNDS + 1):
        print(f"\n--- Waiting for Round {round_id} ---")
        round_num = -1

        while round_num != round_id:
            response = requests.get(get_url)
            data = response.json()
            round_num = data["round"]
            sys_word = data["word"]
            print(f"[System] Word: '{sys_word}' | Round: {round_num}")
            sleep(1)

        # Always check status first
        status = requests.get(status_url).json().get("status", {})
        print("\n[Status]")
        print(f"  P1 word: {status.get('p1_word')}")
        print(f"  P2 word: {status.get('p2_word')}")
        print(f"  System word: {status.get('system_word')}")
        print(f"  Round: {status.get('round')}")
        print(f"  Game over: {status.get('game_over')}")

        # Pick a word using AI logic
        chosen_word_id = what_beats(sys_word)
        print(f"[Decision] Chosen word_id: {chosen_word_id}")

        payload = {
            "player_id": player_id,
            "word_id": chosen_word_id,
            "round_id": round_id
        }

        response = requests.post(post_url, json=payload)
        print(f"[Submit] Sent: {payload}")
        print(f"[Response] {response.json()}")

    print(f"\n=== Game Finished for {player_id} ===")

# Run it
what_beats("broom")
play_game("ynngS2cuuV")
