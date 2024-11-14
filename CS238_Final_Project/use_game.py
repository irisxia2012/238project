from simulator import Card99Game
def main():
    print("Welcome to the card game 99!")

    # Get number of players
    num_players = int(input("Enter the number of players: "))
    if num_players < 2:
        print("The game requires at least 2 players.")
        return

    # Initialize the game
    game = Card99Game(num_players)

    # Start the game loop
    game.start_game()


if __name__ == "__main__":
    main()
