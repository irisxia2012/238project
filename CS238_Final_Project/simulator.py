import random


class Card99:
    def __init__(self):
        self.deck = self.initialize_deck()
        random.shuffle(self.deck)
        self.dead_pile = []
        self.counter = 0

    def initialize_deck(self):
        # Initialize a standard 52-card deck, represented as a list of tuples (value, suit)
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # Where Ace=1/11, Jack=11, Queen=12, King=13
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        return [(value, suit) for value in values for suit in suits]

    def draw_card(self):
        # Draw a card, reshuffle if the deck is empty
        if not self.deck:
            self.reshuffle()
        return self.deck.pop()

    def reshuffle(self):
        # Add dead pile back to deck and shuffle
        self.deck = self.initialize_deck()
        random.shuffle(self.deck)
        self.dead_pile = []

    def add_to_dead_pile(self, card):
        self.dead_pile.append(card)


class Card99Player:
    def __init__(self):
        self.current_hand = []
        self.num_tokens = 3

    def get_hand(self, card_deck):
        # Draw three cards from the deck for a new hand
        self.current_hand = [card_deck.draw_card() for _ in range(3)]

    def get_card(self, card_deck):
        self.current_hand.append(card_deck.draw_card())

    def play_card(self, card_deck, card, counter):
        # Play a card, apply its effect, and draw a replacement
        value, _ = card
        card_deck.add_to_dead_pile(card)
        self.current_hand.remove(card)

        # Update counter based on card value with special rules
        if value == 4:
            # Reverse play or take another turn if only 2 players
            counter+=0  # Doesn't change counter, only affects turn order
        elif value == 1:  # Ace
            counter += 1 if counter + 1 <= 99 else 11
        elif value == 3:
            counter+=0  # Skip next player (handle in game logic)
        elif value == 9:
            counter = 99
        elif value == 10:
            choice = input("Play 10 to add or subtract 10? (+ or -): ")
            counter += 10 if choice == '+' else -10
        elif value == 13:  # King
            counter+=0# King adds 0, so counter remains the same
        else:
            counter += value  # Regular card adds its face value

        # Draw a replacement card to keep the hand at three cards
        self.current_hand.append(card_deck.draw_card())
        return counter


class Card99Game:
    def __init__(self, num_players):
        self.card_deck = Card99()
        self.players = [Card99Player() for _ in range(num_players)]
        self.current_player = 0
        self.direction = 1  # Controls play order; reverse with -1

    def start_game(self):
        for player in self.players:
            player.get_hand(self.card_deck)
            print(player.current_hand)

        # Game loop
        while len([p for p in self.players if p.num_tokens > 0]) > 1:
            input()
            player = self.players[self.current_player]

            # Display current state
            print(f"\nPlayer {self.current_player + 1}'s turn")
            print(f"Current counter: {self.card_deck.counter}")
            print(f"Player's hand: {player.current_hand}")

            # Choose a card to play
            card = self.choose_card(player)
            print(card)
            self.card_deck.counter = player.play_card(self.card_deck, card, self.card_deck.counter)
            # player.get_card(self.card_deck)
            # Check for losing condition
            if self.card_deck.counter > 99:
                print(f"Player {self.current_player + 1} exceeded 99 and loses a token!")
                player.num_tokens -= 1
                self.card_deck.counter = 0  # Reset for next round
                # Refill the player's hand
                if player.num_tokens > 0:
                    player.get_hand(self.card_deck)
                else:
                    self.players.remove(player)

            # Update current player and handle skipping/reversing
            self.next_player(card)

        # Announce winner
        winner = next(p for p in self.players if p.num_tokens > 0)
        print(f"Player {self.players.index(winner) + 1} wins the game!")

    def choose_card(self, player):
        # For now, pick a random card from the player's hand (you could improve this logic)
        return random.choice(player.current_hand)

    def next_player(self, card):
        # Determine the next player considering skips and reverses
        if card[0] == 3:  # Skip next player
            self.current_player = (self.current_player + 2 * self.direction) % len(self.players)
        elif card[0] == 4:  # Reverse play direction
            self.direction *= -1
            self.current_player = (self.current_player + self.direction) % len(self.players)
        else:
            self.current_player = (self.current_player + self.direction) % len(self.players)
