import numpy as np
import random
from tqdm import tqdm  
import matplotlib.pyplot as plt
import pandas as pd

TOTAL_POINTS = 99
ACTIONS = ['4', '3', '9', '10', 'Ace', 'king', 'queen', 'jack'] + [str(i) for i in range(2, 11) if i not in [3, 4, 9, 10]]
POINTS_DICT = {'4': 0, '3': 0, '9': 99, '10': 10, 'Ace': 1, 'king': 0, 'queen': 10, 'jack': 10}

LEARNING_RATE = 0.15
DISCOUNT_FACTOR = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.6
EPISODES = 100000

# set opponent strategy
strategy = "greedy"

q_table = {}

def initialize_q_table(state):
    if state not in q_table:
        q_table[state] = {action: 0 for action in ACTIONS}

def choose_action(state, hand):
    if not hand:
        return None
    if np.random.rand() < EPSILON:
        return random.choice(hand)
    else:
        return max(hand, key=lambda action: q_table[state].get(action, 0))

def update_q_value(state, action, reward, next_state):
    max_next_q_value = max(q_table[next_state].values()) if next_state in q_table else 0
    q_table[state][action] += LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q_value - q_table[state][action])

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

def initialize_deck():
    deck = ACTIONS * 4
    random.shuffle(deck)
    return deck

def reset_game(agent_tokens, opponent_tokens):
    """Reset the game by reinitializing the deck and dealing new hands to players."""
    deck = initialize_deck()
    current_total = 0
    state = (current_total, agent_tokens)
    agent_hand = [deck.pop() for _ in range(3)]
    opponents_hands = [[deck.pop() for _ in range(3)], [deck.pop() for _ in range(3)]]
    return deck, current_total, state, agent_hand, opponents_hands, opponent_tokens[:]

def opponent_strategy(strategy, valid_cards, opponent_hand):
    if strategy == "dumb":
        return max(opponent_hand, key=lambda card: POINTS_DICT.get(card, int(card) if card.isdigit() else 0))
    elif strategy == "random":
        return np.random.choice(opponent_hand)
    elif strategy == "defensive" and valid_cards:
        if '9' in opponent_hand: # brings total to 99
            return '9'
        else:
            return min(valid_cards, key=lambda card: POINTS_DICT.get(card, int(card) if card.isdigit() else 0))
    elif strategy == "greedy" and valid_cards:
        return max(valid_cards, key=lambda card: POINTS_DICT.get(card, int(card) if card.isdigit() else 0))
    return opponent_hand[0]

def train_opponent_turn_1(opponent_hand, current_total, deck):
    if not opponent_hand:
        if deck:
            opponent_hand.append(deck.pop())
        else:
            return current_total
    
    # pick choice based on strategy
    valid_cards = [card for card in opponent_hand if current_total + POINTS_DICT.get(card, int(card) if card.isdigit() else 0) <= 99]
    choice = opponent_strategy(strategy, valid_cards, opponent_hand)

    opponent_hand.remove(choice)
    new_total = play_card(choice, current_total)

    return new_total

def train_opponent_turn_2(opponent_hand, current_total, deck):
    if not opponent_hand:
        if deck:
            opponent_hand.append(deck.pop())
        else:
            return current_total
    
    # pick choice based on strategy
    valid_cards = [card for card in opponent_hand if current_total + POINTS_DICT.get(card, int(card) if card.isdigit() else 0) <= 99]
    choice = opponent_strategy(strategy, valid_cards, opponent_hand)

    opponent_hand.remove(choice)
    new_total = play_card(choice, current_total)

    return new_total

def opponent_turn(opponent_hand, current_total, deck):
    if not opponent_hand:
        if deck:
            opponent_hand.append(deck.pop())
        else:
            return current_total
        
    # pick choice based on strategy
    valid_cards = [card for card in opponent_hand if current_total + POINTS_DICT.get(card, int(card) if card.isdigit() else 0) <= 99]
    choice = opponent_strategy(strategy, valid_cards, opponent_hand)
    opponent_hand.remove(choice)
    new_total = play_card(choice, current_total)

    return new_total

rewards = []
for episode in tqdm(range(EPISODES), desc="Training Q-learning agent"):
    cum_reward = 0
    agent_tokens = 3
    opponent_tokens = [3, 3]
    deck, current_total, state, agent_hand, opponents_hands, opponent_tokens = reset_game(agent_tokens, opponent_tokens)
    current_player = 'agent'

    while agent_tokens > 0 and any(t > 0 for t in opponent_tokens):
        if current_player == 'agent':
            initialize_q_table(state)
            action = choose_action(state, agent_hand)
            if action is None:
                break
            agent_hand.remove(action)
            new_total = play_card(action, current_total)

            if new_total > TOTAL_POINTS:
                reward = -10
                agent_tokens -= 1
                if agent_tokens > 0:
                    deck, current_total, state, agent_hand, opponents_hands, opponent_tokens = reset_game(agent_tokens, opponent_tokens)
                else:
                    break
                continue
            else:
                reward = max(1, 99 - abs(99 - new_total))
                current_total = new_total

            cum_reward += reward
            next_state = (current_total, agent_tokens)
            initialize_q_table(next_state)
            update_q_value(state, action, reward, next_state)
            state = next_state
            if deck:
                agent_hand.append(deck.pop())
            current_player = 'opponent_1'

        elif current_player == 'opponent_1':
            if opponent_tokens[0] > 0:
                current_total = train_opponent_turn_1(opponents_hands[0], current_total, deck)
                if current_total > TOTAL_POINTS:
                    opponent_tokens[0] -= 1
                    if opponent_tokens[0] == 0 and opponent_tokens[1] == 0:
                        break
                    if opponent_tokens[0] > 0:
                        deck, current_total, state, agent_hand, opponents_hands, opponent_tokens = reset_game(agent_tokens, opponent_tokens)
                current_player = 'opponent_2'
            else:
                current_player = 'opponent_2'

        elif current_player == 'opponent_2':
            if opponent_tokens[1] > 0:
                current_total = train_opponent_turn_1(opponents_hands[1], current_total, deck)
                if current_total > TOTAL_POINTS:
                    opponent_tokens[1] -= 1
                    if opponent_tokens[0] == 0 and opponent_tokens[1] == 0:
                        break
                    if opponent_tokens[1] > 0:
                        deck, current_total, state, agent_hand, opponents_hands, opponent_tokens = reset_game(agent_tokens, opponent_tokens)
                current_player = 'agent'
            else:
                current_player = 'agent'

    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
    rewards.append((episode, cum_reward))
print("Training complete. Q-learning agent is ready.")

def evaluate_agent(episodes=10000):
    global EPSILON
    wins = 0
    original_epsilon = EPSILON
    EPSILON = 0

    for _ in tqdm(range(episodes), desc="Evaluating agent"):
        agent_tokens = 3
        opponent_tokens = [3, 3]
        deck, current_total, state, agent_hand, opponents_hands, opponent_tokens = reset_game(agent_tokens, opponent_tokens)
        current_player = 'agent'

        while agent_tokens > 0 and any(t > 0 for t in opponent_tokens):
            if current_player == 'agent':
                action = choose_action(state, agent_hand)
                if action is None:
                    break
                agent_hand.remove(action)
                new_total = play_card(action, current_total)

                if new_total > TOTAL_POINTS:
                    agent_tokens -= 1
                    if agent_tokens > 0:
                        deck, current_total, state, agent_hand, opponents_hands, opponent_tokens = reset_game(agent_tokens, opponent_tokens)
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
                    current_total = opponent_turn(opponents_hands[0], current_total, deck)
                    if current_total > TOTAL_POINTS:
                        opponent_tokens[0] -= 1
                        if opponent_tokens[0] == 0 and opponent_tokens[1] == 0:
                            break
                        if opponent_tokens[0] > 0:
                            deck, current_total, state, agent_hand, opponents_hands, opponent_tokens = reset_game(agent_tokens, opponent_tokens)
                    current_player = 'opponent_2'
                else:
                    current_player = 'opponent_2'

            elif current_player == 'opponent_2':
                if opponent_tokens[1] > 0:
                    current_total = opponent_turn(opponents_hands[1], current_total, deck)
                    if current_total > TOTAL_POINTS:
                        opponent_tokens[1] -= 1
                        if opponent_tokens[0] == 0 and opponent_tokens[1] == 0:
                            break
                        if opponent_tokens[1] > 0:
                            deck, current_total, state, agent_hand, opponents_hands, opponent_tokens = reset_game(agent_tokens, opponent_tokens)
                    current_player = 'agent'
                else:
                    current_player = 'agent'

        if agent_tokens > 0 and all(t <= 0 for t in opponent_tokens):
            wins += 1

    EPSILON = original_epsilon
    win_rate = wins / episodes
    print(f"Agent Win Rate: {win_rate:.2f}")

evaluate_agent(episodes=100000)

# plot cumulative reward
reward_df = pd.DataFrame(rewards, columns = ["episode", "reward"])
reward_df.to_csv("q_learning.csv")