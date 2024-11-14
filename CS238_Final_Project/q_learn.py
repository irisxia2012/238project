import numpy as np
import random

# Game configuration
TOTAL_POINTS = 99
ACTIONS = ['4', '3', '9', '10+', '10-', 'ace1', 'ace11', 'king'] + [str(i) for i in range(2, 11) if i not in [3, 4, 9, 10]]
POINTS_DICT = {'4': 0, '3': 0, '9': 99, '10+': 10, '10-': -10, 'ace1': 1, 'ace11': 11, 'king': 0}

# Q-learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.1
EPISODES = 5000

# Q-table initialization
q_table = {}

def initialize_q_table(state):
    if state not in q_table:
        q_table[state] = {action: 0 for action in ACTIONS}

def choose_action(state):
    if np.random.rand() < EPSILON:
        return random.choice(ACTIONS)  # Random action for exploration
    else:
        return max(q_table[state], key=q_table[state].get)  # Best action for exploitation

def update_q_value(state, action, reward, next_state):
    max_next_q_value = max(q_table[next_state].values()) if next_state in q_table else 0
    q_table[state][action] += LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q_value - q_table[state][action])

def play_card(card, current_total):
    if card == '9':
        return 99
    elif card == '10+':
        return current_total + 10
    elif card == '10-':
        return current_total - 10
    elif card in POINTS_DICT:
        return current_total + POINTS_DICT[card]
    return current_total + int(card)  # Non-special cards add their numeric value

def initialize_deck():
    deck = ACTIONS * 4  # 4 of each card to simulate a standard 52-card deck
    random.shuffle(deck)
    return deck

# Training loop
for episode in range(EPISODES):
    deck = initialize_deck()
    agent_tokens = 3
    current_total = 0
    state = (current_total, agent_tokens)

    # Draw 3 cards for the agent and 3 for each opponent
    agent_hand = [deck.pop() for _ in range(3)]
    opponents_hands = [[deck.pop() for _ in range(3)], [deck.pop() for _ in range(3)]]

    while agent_tokens > 0:
        initialize_q_table(state)

        # Agent's turn: choose an action from available cards in hand
        action = choose_action(state) if agent_hand else 'king'  # Play king if hand is empty
        if action in agent_hand:
            agent_hand.remove(action)  # Remove played card from hand
            new_total = play_card(action, current_total)

            # Determine reward and next state
            if new_total > TOTAL_POINTS:
                reward = -1
                agent_tokens -= 1
            else:
                reward = 1
                current_total = new_total

            # Update Q-values
            next_state = (current_total, agent_tokens)
            initialize_q_table(next_state)
            update_q_value(state, action, reward, next_state)
            state = next_state

            # Draw a new card if the deck is not empty
            if deck:
                agent_hand.append(deck.pop())

        # Opponents' turns (random strategy)
        for opponent_hand in opponents_hands:
            if opponent_hand and current_total < TOTAL_POINTS:
                opponent_card = random.choice(opponent_hand)
                opponent_hand.remove(opponent_card)
                current_total = play_card(opponent_card, current_total)
                if current_total > TOTAL_POINTS:
                    break  # Stop if opponents cause the game to exceed 99

                # Draw a new card if the deck is not empty
                if deck:
                    opponent_hand.append(deck.pop())

    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

print("Training complete. Q-learning agent is ready.")

# Evaluation function
def evaluate_agent(episodes=100):
    wins = 0
    for _ in range(episodes):
        deck = initialize_deck()
        agent_tokens = 3
        current_total = 0
        state = (current_total, agent_tokens)

        agent_hand = [deck.pop() for _ in range(3)]
        opponents_hands = [[deck.pop() for _ in range(3)], [deck.pop() for _ in range(3)]]

        while agent_tokens > 0:
            action = max(q_table[state], key=q_table[state].get) if state in q_table else 'king'
            if action in agent_hand:
                agent_hand.remove(action)
                new_total = play_card(action, current_total)

                if new_total > TOTAL_POINTS:
                    agent_tokens -= 1
                    break
                current_total = new_total
                state = (current_total, agent_tokens)

                if deck:
                    agent_hand.append(deck.pop())

            # Opponents' turns
            for opponent_hand in opponents_hands:
                if opponent_hand and current_total < TOTAL_POINTS:
                    opponent_card = random.choice(opponent_hand)
                    opponent_hand.remove(opponent_card)
                    current_total = play_card(opponent_card, current_total)
                    if current_total > TOTAL_POINTS:
                        break
                    if deck:
                        opponent_hand.append(deck.pop())

        if agent_tokens > 0:
            wins += 1

    win_rate = wins / episodes
    print(f"Agent Win Rate: {win_rate:.2f}")

# Play a single game and observe
def play_single_game(verbose=True):
    deck = initialize_deck()
    agent_tokens = 3
    current_total = 0
    state = (current_total, agent_tokens)

    agent_hand = [deck.pop() for _ in range(3)]
    opponents_hands = [[deck.pop() for _ in range(3)], [deck.pop() for _ in range(3)]]

    while agent_tokens > 0:
        if verbose:
            print(f"Current Total: {current_total}, Agent Tokens: {agent_tokens}")
            print(f"Agent's Hand: {agent_hand}")

        action = max(q_table[state], key=q_table[state].get) if state in q_table else 'king'
        if action in agent_hand:
            agent_hand.remove(action)
            new_total = play_card(action, current_total)

            if verbose:
                print(f"Agent plays {action}, new total: {new_total}")

            if new_total > TOTAL_POINTS:
                agent_tokens -= 1
                if verbose:
                    print("Agent loses a token!")
                break

            current_total = new_total
            state = (current_total, agent_tokens)

            if deck:
                agent_hand.append(deck.pop())

        # Opponents' turns
        for opponent_hand in opponents_hands:
            if opponent_hand and current_total < TOTAL_POINTS:
                opponent_card = random.choice(opponent_hand)
                opponent_hand.remove(opponent_card)
                current_total = play_card(opponent_card, current_total)
                if verbose:
                    print(f"Opponent plays {opponent_card}, new total: {current_total}")
                if current_total > TOTAL_POINTS:
                    if verbose:
                        print("Opponent loses the game!")
                    break
                if deck:
                    opponent_hand.append(deck.pop())

# Evaluate the trained agent
evaluate_agent(episodes=100)

# Play a single game to observe the agent in action
play_single_game()
