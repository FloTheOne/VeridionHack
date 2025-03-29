import requests
import csv
from time import sleep
import random

host = "http://172.18.4.158:8000"
post_url = f"{host}/submit-word"
get_url = f"{host}/get-word"
status_url = f"{host}/status"

NUM_ROUNDS = 5
CSV_FILE = "updated_sick_nouns_logic (1).csv"

def find_best_counter_category(csv_filename, input_word):
    best_match = None
    
    with open(csv_filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if input_word in row["word"]:
                if best_match is None or len(row["word"]) > len(best_match["word"]):
                    best_match = row
    
    return best_match["Best_Counter_Category"] if best_match else "Peace"

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

        # Pick the best counter category
        best_category = find_best_counter_category(CSV_FILE, sys_word)
        print(f"[Decision] Best Counter Category: {best_category}")

        payload = {
            "player_id": player_id,
            "word_id": best_category,
            "round_id": round_id
        }

        response = requests.post(post_url, json=payload)
        print(f"[Submit] Sent: {payload}")
        print(f"[Response] {response.json()}")

    print(f"\n=== Game Finished for {player_id} ===")

# Run the game
play_game("Bombardilo")