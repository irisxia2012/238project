import numpy as np
import random
from tqdm import tqdm

TOTAL_POINTS = 99
ACTIONS = ['4', '3', '9', '10', 'Ace', 'king', 'queen', 'jack'] + [str(i) for i in range(2, 11) if i not in [3, 4, 9, 10]]
POINTS_DICT = {'4': 0, '3': 0, '9': 99, '10': 10, 'Ace': 1, 'king': 0, 'queen': 10, 'jack': 10}

GAMMA = .9  
THETA = 1e-4  
STATE_SPACE = [(total, tokens) for total in range(100) for tokens in range(4)]
VALUE_FUNCTION = {state: 0 for state in STATE_SPACE}

def initialize_deck():
    deck = ACTIONS * 4
    random.shuffle(deck)
    return deck

def play_card(card, current_total):
    if card == 'Ace':
        new_total = current_total + 11 if current_total + 11 <= TOTAL_POINTS else current_total + 1
    elif card == '10':
        new_total = current_total + 10 if current_total + 10 <= TOTAL_POINTS else current_total - 10
    elif card == '9':
        new_total = 99
    elif card in POINTS_DICT:
        new_total = current_total + POINTS_DICT[card]
    else:
        new_total = current_total + int(card)
    return new_total

def reset_game(agent_tokens, opponent_tokens):
    deck = initialize_deck()
    current_total = 0
    state = (current_total, agent_tokens)
    agent_hand = [deck.pop() for _ in range(3)]
    opponents_hands = [[deck.pop() for _ in range(3)], [deck.pop() for _ in range(3)]]
    return deck, current_total, state, agent_hand, opponents_hands, opponent_tokens[:]

def opponent_turn(opponent_hand, current_total, deck):
    if not opponent_hand:
        if deck:
            opponent_hand.append(deck.pop())
        else:
            return current_total

    if '9' in opponent_hand:
        choice = '9'
    else:
        valid_cards = [
            card for card in opponent_hand
            if current_total + POINTS_DICT.get(card, int(card) if card.isdigit() else 0) <= TOTAL_POINTS
        ]
        if valid_cards:
            choice = min(valid_cards, key=lambda card: POINTS_DICT.get(card, int(card) if card.isdigit() else 0))
        else:
            choice = opponent_hand[0]
    choice = max(opponent_hand)
    opponent_hand.remove(choice)
    new_total = play_card(choice, current_total)

    if deck:
        opponent_hand.append(deck.pop())

    return new_total

def value_iteration():
    global VALUE_FUNCTION
    delta = float('inf')
    while delta > THETA:
        delta = 0
        for state in STATE_SPACE:
            current_total, tokens = state
            if tokens == 0:
                continue

            max_value = float('-inf')
            for action in ACTIONS:
                new_total = play_card(action, current_total)
                if new_total > TOTAL_POINTS:
                    continue

                next_state = (new_total, tokens)
                reward = max(1, 99 - abs(99 - new_total))
                value = reward + GAMMA * VALUE_FUNCTION[next_state]
                max_value = max(max_value, value)

            delta = max(delta, abs(VALUE_FUNCTION[state] - max_value))
            VALUE_FUNCTION[state] = max_value
    return VALUE_FUNCTION

def derive_policy(value_function):
    policy = {}
    for state in STATE_SPACE:
        current_total, tokens = state
        if tokens == 0:
            continue

        best_action = None
        best_value = float('-inf')
        for action in ACTIONS:
            new_total = play_card(action, current_total)
            if new_total > TOTAL_POINTS:
                continue

            next_state = (new_total, tokens)
            reward = max(1, 99 - abs(99 - new_total))
            value = reward + GAMMA * value_function[next_state]
            if value > best_value:
                best_action = action
                best_value = value
        policy[state] = best_action
    return policy

def evaluate_policy(policy, episodes=1000):
    wins = 0

    for episode in tqdm(range(episodes), desc="Evaluating agent"):
        agent_tokens = 3
        opponent_tokens = [3, 3]
        deck, current_total, state, agent_hand, opponents_hands, opponent_tokens = reset_game(agent_tokens, opponent_tokens)
        current_player = 'agent'

        print(f"\nStarting Episode {episode + 1}")
        while agent_tokens > 0 and any(t > 0 for t in opponent_tokens):
            print(f"\nCurrent Total: {current_total}")
            print(f"Agent Tokens: {agent_tokens}, Opponent Tokens: {opponent_tokens}")
            print(f"Agent Hand: {agent_hand}")

            if current_player == 'agent':
                valid_actions = [action for action in agent_hand if play_card(action, current_total) <= TOTAL_POINTS]
                if not valid_actions:
                    print("Agent exceeded 99 points. Losing a token.")
                    agent_tokens -= 1
                    if agent_tokens > 0:
                        deck, current_total, state, agent_hand, opponents_hands, opponent_tokens = reset_game(agent_tokens, opponent_tokens)
                        print("Game reset after agent token loss.")
                    else:
                        break
                    continue

                best_action = max(
                    valid_actions,
                    key=lambda action: VALUE_FUNCTION.get((play_card(action, current_total), agent_tokens), 0)
                )

                print(f"Agent plays: {best_action}")
                agent_hand.remove(best_action)
                new_total = play_card(best_action, current_total)

                if new_total > TOTAL_POINTS:
                    print("Agent exceeded 99 points. Losing a token.")
                    agent_tokens -= 1
                    if agent_tokens > 0:
                        deck, current_total, state, agent_hand, opponents_hands, opponent_tokens = reset_game(agent_tokens, opponent_tokens)
                        print("Game reset after agent token loss.")
                    else:
                        break
                    continue

                current_total = new_total
                state = (current_total, agent_tokens)

                if deck:
                    agent_hand.append(deck.pop())

                current_player = 'opponent_1'

            elif current_player == 'opponent_1':
                if opponent_tokens[0] > 0:
                    print(f"Opponent 1 Hand: {opponents_hands[0]}")
                    current_total = opponent_turn(opponents_hands[0], current_total, deck)
                    print(f"Opponent 1 plays. New Total: {current_total}")
                    if current_total > TOTAL_POINTS:
                        print("Opponent 1 exceeded 99 points. Losing a token.")
                        opponent_tokens[0] -= 1
                        if opponent_tokens[0] == 0 and opponent_tokens[1] == 0:
                            break
                        if opponent_tokens[0] > 0:
                            deck, current_total, state, agent_hand, opponents_hands, opponent_tokens = reset_game(agent_tokens, opponent_tokens)
                            print("Game reset after Opponent 1 token loss.")
                    current_player = 'opponent_2'
                else:
                    current_player = 'opponent_2'

            elif current_player == 'opponent_2':
                if opponent_tokens[1] > 0:
                    print(f"Opponent 2 Hand: {opponents_hands[1]}")
                    current_total = opponent_turn(opponents_hands[1], current_total, deck)
                    print(f"Opponent 2 plays. New Total: {current_total}")
                    if current_total > TOTAL_POINTS:
                        print("Opponent 2 exceeded 99 points. Losing a token.")
                        opponent_tokens[1] -= 1
                        if opponent_tokens[0] == 0 and opponent_tokens[1] == 0:
                            break
                        if opponent_tokens[1] > 0:
                            deck, current_total, state, agent_hand, opponents_hands, opponent_tokens = reset_game(agent_tokens, opponent_tokens)
                            print("Game reset after Opponent 2 token loss.")
                    current_player = 'agent'
                else:
                    current_player = 'agent'

        if agent_tokens > 0 and all(t <= 0 for t in opponent_tokens):
            print("Agent wins this episode!")
            wins += 1
        else:
            print("Agent loses this episode!")

    win_rate = wins / episodes
    print(f"\nAgent Win Rate: {win_rate:.2f}")

print("Running Value Iteration...")
optimal_value_function = value_iteration()
optimal_policy = derive_policy(optimal_value_function)
print("Value Iteration complete!")

evaluate_policy(optimal_policy, episodes=1000)
