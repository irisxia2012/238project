import numpy as np
import random
from tqdm import tqdm
from abc import ABC, abstractmethod

# Abstract RLMDP class
class RLMDP(ABC):
    def __init__(self, A: list[int], gamma: float):
        self.A = A
        self.gamma = gamma

    @abstractmethod
    def lookahead(self, s: int, a: int) -> float:
        pass

    @abstractmethod
    def update(self, s: int, a: int, r: float, s_prime: int):
        pass

class ModelFreeMDP(RLMDP):
    def __init__(self, A: list[int], gamma: float, Q: np.ndarray, alpha: float):
        super().__init__(A, gamma)
        self.Q = Q
        self.alpha = alpha

class Sarsa(ModelFreeMDP):
    def __init__(self, S: list[int], A: list[int], gamma: float, Q: np.ndarray, alpha: float, ell: tuple[int, int, float]):
        super().__init__(A, gamma, Q, alpha)
        self.S = S
        self.ell = ell

    def lookahead(self, s: int, a: int):
        return self.Q[s, a]

    def update(self, s: int, a: int, r: float, s_prime: int):
        if self.ell is not None:
            s_prev, a_prev, r_prev = self.ell
            self.Q[s_prev, a_prev] += self.alpha * (r_prev + (self.gamma * self.Q[s, a]) - self.Q[s_prev, a_prev])
        self.ell = (s, a, r)

# game setup
TOTAL_POINTS = 99
ACTIONS = ['4', '3', '9', '10', 'Ace', 'king', 'queen', 'jack'] + [str(i) for i in range(2, 11) if i not in [3, 4, 9, 10]]
POINTS_DICT = {'4': 0, '3': 0, '9': 99, '10': 10, 'Ace': 1, 'king': 0, 'queen': 10, 'jack': 10}

STATE_SPACE = 100
ACTION_SPACE = len(ACTIONS)
GAMMA = .90
ALPHA = 0.15
EPSILON = 1.0
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.6
EPISODES = 100000

Q = np.zeros((STATE_SPACE, ACTION_SPACE))
sarsa = Sarsa(S=list(range(STATE_SPACE)), A=list(range(ACTION_SPACE)), gamma=GAMMA, Q=Q, alpha=ALPHA, ell=None)

def initialize_deck():
    deck = ACTIONS * 4
    random.shuffle(deck)
    return deck

def replenish_deck(deck, agent_hand, opponents_hands):
    """Replenish the deck by shuffling the remaining cards, excluding players' hands."""
    all_hands = set(agent_hand + [card for hand in opponents_hands for card in hand])
    remaining_deck = [card for card in ACTIONS * 4 if card not in all_hands]
    random.shuffle(remaining_deck)
    return remaining_deck

def reset_after_token_loss(agent_tokens, opponent_tokens):
    """Reset the point total and redeal cards for all players."""
    deck = initialize_deck()
    agent_hand = [deck.pop() for _ in range(3)]
    opponents_hands = [[deck.pop() for _ in range(3)], [deck.pop() for _ in range(3)]]
    current_total = 0
    return deck, current_total, agent_hand, opponents_hands, agent_tokens, opponent_tokens

def play_card(card, current_total):
    """Determine the new total after playing a card."""
    if card == 'Ace':
        return current_total + 11 if current_total + 11 <= TOTAL_POINTS else current_total + 1
    elif card == '10':
        return current_total + 10 if current_total + 10 <= TOTAL_POINTS else current_total - 10
    elif card == '9':
        return 99
    elif card in POINTS_DICT:
        return current_total + POINTS_DICT[card]
    else:
        return current_total + int(card)

def reset_game(agent_tokens, opponent_tokens):
    """Initialize the game with a new deck and hands for players."""
    deck = initialize_deck()
    current_total = 0
    agent_hand = [deck.pop() for _ in range(3)]
    opponents_hands = [[deck.pop() for _ in range(3)], [deck.pop() for _ in range(3)]]
    return deck, current_total, agent_hand, opponents_hands, opponent_tokens[:]

def opponent_turn(opponent_hand, current_total, deck):
    """Opponent plays a card, and their hand is replenished if needed."""
    if not opponent_hand:
        if not deck:
            deck = replenish_deck(deck, [], [opponent_hand])
        if deck:
            opponent_hand.append(deck.pop())
    if '9' in opponent_hand:
       choice = '9'
    else:
        valid_cards = [card for card in opponent_hand if play_card(card, current_total) <= TOTAL_POINTS]
        choice = max(valid_cards) if valid_cards else opponent_hand[0]
    
    opponent_hand.remove(choice)
    #print(f"Opponent plays card: {choice}")
    new_total = play_card(choice, current_total)
    if not deck:
        deck = replenish_deck(deck, [], [opponent_hand])
    if deck:
        opponent_hand.append(deck.pop())
    return new_total, deck

def choose_action(s, hand):
    """epsilon-greedy strategy."""
    valid_indices = [i for i, card in enumerate(ACTIONS) if card in hand]
    if np.random.rand() < EPSILON:  # Explore
        return random.choice(valid_indices)
    return max(valid_indices, key=lambda i: sarsa.Q[s, i])

# Training loop
for episode in tqdm(range(EPISODES), desc="Training SARSA agent"):
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
    agent_tokens = 3
    opponent_tokens = [3, 3]
    deck, current_total, agent_hand, opponents_hands, opponent_tokens = reset_game(agent_tokens, opponent_tokens)
    s = current_total
    a = choose_action(s, agent_hand)

    while agent_tokens > 0 and any(t > 0 for t in opponent_tokens):
        card = ACTIONS[a]
        if card not in agent_hand:
            a = choose_action(s, agent_hand)
            continue
        agent_hand.remove(card)
        new_total = play_card(card, current_total)
        if new_total > TOTAL_POINTS:
            r = -10
            agent_tokens -= 1
            if agent_tokens > 0:
                deck, current_total, agent_hand, opponents_hands, agent_tokens, opponent_tokens = reset_after_token_loss(agent_tokens, opponent_tokens)
            else:
                break
        else:
            r = max(1, 99 - abs(99 - new_total))
            current_total = new_total
        s_prime = current_total
        a_prime = choose_action(s_prime, agent_hand)
        sarsa.update(s, a, r, s_prime)
        s, a = s_prime, a_prime
        if not deck:
            deck = replenish_deck(deck, agent_hand, opponents_hands)
        if deck:
            agent_hand.append(deck.pop())
        for i, opponent_hand in enumerate(opponents_hands):
            if opponent_tokens[i] > 0:  
                current_total, deck = opponent_turn(opponent_hand, current_total, deck)
                if current_total > TOTAL_POINTS:
                    opponent_tokens[i] -= 1
                    if opponent_tokens[i] > 0:
                        deck, current_total, agent_hand, opponents_hands, agent_tokens, opponent_tokens = reset_after_token_loss(agent_tokens, opponent_tokens)

print("Training complete. SARSA agent is ready.")

def simulate_game():
    """Simulate a game using the trained SARSA agent."""
    agent_tokens = 3
    opponent_tokens = [3, 3]
    deck, current_total, agent_hand, opponents_hands, opponent_tokens = reset_game(agent_tokens, opponent_tokens)
    current_player = 'agent'
    step = 0

    while agent_tokens > 0 and any(t > 0 for t in opponent_tokens):
        print(f"\nStep {step}: Current Total: {current_total}")
        print(f"Agent Tokens: {agent_tokens}, Opponent Tokens: {opponent_tokens}")
        print(f"Agent Hand: {agent_hand}")
        for i, hand in enumerate(opponents_hands):
            print(f"Opponent {i + 1} Hand: {hand}")
        print(f"Deck Size: {len(deck)}")

        if current_player == 'agent':
            a = choose_action(current_total, agent_hand)
            card = ACTIONS[a]
            print(f"Agent chooses card: {card}")
            if card not in agent_hand:
                print("Invalid card chosen (not in hand).")
                break
            agent_hand.remove(card)
            new_total = play_card(card, current_total)
            if new_total > TOTAL_POINTS:
                print("Agent exceeded 99 points. Losing a token.")
                agent_tokens -= 1
                if agent_tokens > 0:
                    deck, current_total, agent_hand, opponents_hands, agent_tokens, opponent_tokens = reset_after_token_loss(agent_tokens, opponent_tokens)
                else:
                    break
                continue
            current_total = new_total
            if not deck:
                deck = replenish_deck(deck, agent_hand, opponents_hands)
            if deck:
                agent_hand.append(deck.pop())
            current_player = 'opponent_1'
        elif current_player == 'opponent_1':
            if opponent_tokens[0] > 0:  
                current_total, deck = opponent_turn(opponents_hands[0], current_total, deck)
                print(f"Opponent 1 plays. New Total: {current_total}")
                if current_total > TOTAL_POINTS:
                    print(f"Opponent 1 exceeded 99 points. Losing a token.")
                    opponent_tokens[0] -= 1
                    if opponent_tokens[0] > 0:
                        deck, current_total, agent_hand, opponents_hands, agent_tokens, opponent_tokens = reset_after_token_loss(agent_tokens, opponent_tokens)
            current_player = 'opponent_2'
        elif current_player == 'opponent_2':
            if opponent_tokens[1] > 0:  
                current_total, deck = opponent_turn(opponents_hands[1], current_total, deck)
                print(f"Opponent 2 plays. New Total: {current_total}")
                if current_total > TOTAL_POINTS:
                    print(f"Opponent 2 exceeded 99 points. Losing a token.")
                    opponent_tokens[1] -= 1
                    if opponent_tokens[1] > 0:
                        deck, current_total, agent_hand, opponents_hands, agent_tokens, opponent_tokens = reset_after_token_loss(agent_tokens, opponent_tokens)
                    else:
                        current_total = 0
            current_player = 'agent'
        step += 1

    print("\nGame Over!")
    if agent_tokens > 0:
        print("Agent Wins!")
    else:
        print("Agent Loses!")
    print(f"Final Agent Tokens: {agent_tokens}")
    print(f"Final Opponent Tokens: {opponent_tokens}")

def evaluate_agent(episodes=10000):
    """Evaluate the trained SARSA agent by simulating games."""
    wins = 0
    global EPSILON
    original_epsilon = EPSILON  
    EPSILON = 0  

    for episode in tqdm(range(episodes), desc="Evaluating SARSA agent"):
        agent_tokens = 3
        opponent_tokens = [3, 3]
        deck, current_total, agent_hand, opponents_hands, opponent_tokens = reset_game(agent_tokens, opponent_tokens)
        current_player = 'agent'

        while agent_tokens > 0 and any(t > 0 for t in opponent_tokens):
            if current_player == 'agent':
                a = choose_action(current_total, agent_hand)
                card = ACTIONS[a]
                if card not in agent_hand:
                    break
                agent_hand.remove(card)
                new_total = play_card(card, current_total)
                if new_total > TOTAL_POINTS:
                    agent_tokens -= 1
                    if agent_tokens > 0:
                        deck, current_total, agent_hand, opponents_hands, agent_tokens, opponent_tokens = reset_after_token_loss(agent_tokens, opponent_tokens)
                    else:
                        break
                else:
                    current_total = new_total
                    if not deck:
                        deck = replenish_deck(deck, agent_hand, opponents_hands)
                    if deck:
                        agent_hand.append(deck.pop())
                current_player = 'opponent_1'
            elif current_player == 'opponent_1':
                if opponent_tokens[0] > 0:
                    current_total, deck = opponent_turn(opponents_hands[0], current_total, deck)
                    if current_total > TOTAL_POINTS:
                        opponent_tokens[0] -= 1
                        if opponent_tokens[0] > 0:
                            deck, current_total, agent_hand, opponents_hands, agent_tokens, opponent_tokens = reset_after_token_loss(agent_tokens, opponent_tokens)
                current_player = 'opponent_2'
            elif current_player == 'opponent_2':
                if opponent_tokens[1] > 0:
                    current_total, deck = opponent_turn(opponents_hands[1], current_total, deck)
                    if current_total > TOTAL_POINTS:
                        opponent_tokens[1] -= 1
                        if opponent_tokens[1] > 0:
                            deck, current_total, agent_hand, opponents_hands, agent_tokens, opponent_tokens = reset_after_token_loss(agent_tokens, opponent_tokens)
                        else:
                            current_total = 0
                current_player = 'agent'

        if agent_tokens > 0 and all(t <= 0 for t in opponent_tokens):
            wins += 1

    EPSILON = original_epsilon  
    win_rate = wins / episodes
    print(f"Agent Win Rate: {win_rate:.2f}")

evaluate_agent(episodes=10000)
#simulate_game()
