# My Chess Project

I made this project to experiment with Python, chess, and a small neural network. I got some help from GitHub Copilot while building and improving it, especially with working through ideas and code. I still wanted to understand what the parts do, so this README is my explanation of how it works and how to run it.

The project has a few things you can try:

- Play chess against the agent in a window.
- Watch the agent play itself and learn from each game.
- Train the model on completed games in a PGN file.
- Ask a saved model to suggest a move for a position.

This is mainly a learning project, not a powerful chess engine. It does not search ahead like Stockfish, and a model with random weights will not play well.

## Getting started

You need Python 3.10 or later. The project uses `python-chess`, `numpy`, `torch`, and `pygame`.

Install them with:

```bash
python -m pip install python-chess numpy torch pygame
```

Use the same Python environment to run the project. To see the commands:

```bash
python chess_project.py --help
```

## Play against the agent

Run the project with no extra arguments:

```bash
python chess_project.py
```

That opens the graphical game. You play White, and the agent plays Black. Click one of your pieces, then click where you want it to move. Pawn promotion defaults to a queen.

- Press **R** or click **New game** to restart.
- Press **Esc** to close the window.

The game loads `self_play_model.pth` if that file exists. If it does not, the program creates a model with random weights, so the agent's moves may not be very good.

You can also start the game explicitly and choose the checkpoint or how long the agent waits before moving:

```bash
python chess_project.py play --checkpoint self_play_model.pth --agent-delay 0.35
```

## Watch self-play

To watch the agent play both sides, run:

```bash
python chess_project.py self-play
```

The window shows the board, game count, results, and latest training loss. Press **Space** to pause or **Esc** to quit. After each finished game, the program trains the model using that game's result and winning moves, then saves the weights to `self_play_model.pth`.

If a game has not ended after 30 moves per side, the program decides the result based on the pieces left on the board. This keeps games from going on for too long. Games usually take around one or two seconds, depending on the computer and game.

You can change the checkpoint, device, and time between moves:

```bash
python chess_project.py self-play --checkpoint self_play_model.pth --move-delay 0.015
```

The model gets updated after games, but that does not mean it will get better every time. It is an experiment in learning from self-play, not a guarantee of stronger chess.

## Train using PGN games

PGN is a format for saving chess games. To train using a file named `games.pgn`, run:

```bash
python chess_project.py train --pgn games.pgn
```

By default, the program uses up to 5,000 positions and trains for five epochs. It saves the resulting model as `chess_model.pth`. You can change those settings:

```bash
python chess_project.py train --pgn games.pgn --positions 20000 --epochs 10 --batch-size 32 --checkpoint output\my_model.pth
```

The games need to have a known result (`1-0`, `0-1`, or `1/2-1/2`). Unfinished games are skipped because they do not have a final result to learn from. Training will report an error if it cannot find any usable positions.

If you have a CUDA-enabled PyTorch setup and a supported GPU, you can try:

```bash
python chess_project.py train --pgn games.pgn --device cuda
```

## Get a move prediction

After you have a saved model, you can ask it to suggest a move:

```bash
python chess_project.py predict --checkpoint chess_model.pth
```

You can also pass a FEN position:

```bash
python chess_project.py predict --checkpoint chess_model.pth --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

The program only chooses from legal moves. The model's move policy does not distinguish between the different kinds of pawn promotion, so when a position has only promotion moves, it returns the first available promotion.

## Try the model demo

Run:

```bash
python chess_project.py demo
```

This makes a sample board position and passes it through the network. It prints the input and output shapes, a value estimate, and the number of model parameters. It is mostly there to check that the model works; it is not a measure of how well the agent plays.

## A little about how it works

The chess rules and legal moves come from the `python-chess` package. The neural network is built with PyTorch.

### Turning the board into numbers

The network cannot use a chessboard directly, so `encode_board(board)` turns it into a NumPy array with shape `(12, 8, 8)`. There is one plane for each piece type for White and another six for Black. A square is marked `1.0` if it contains that piece and `0.0` otherwise.

`board_to_tensor(board)` converts that array into a PyTorch tensor with shape `(1, 12, 8, 8)`. The first number is the batch size.

This simple board encoding only includes piece locations. It does not tell the network about castling rights, en passant, repetition, or the move counters.

### Choosing moves

The policy part of the network has 4,096 outputs: one for every possible starting-square and destination-square combination (`64 * 64`). The program masks out illegal moves before choosing, so the agent should only make a legal move.

Promotion choices are not included in the policy output. That is one of the model's current limitations.

### The value estimate

The other output is a value between `-1` and `1`:

- `+1` means a win for the player whose turn it is.
- `-1` means a loss for the player whose turn it is.
- `0` means a draw.

The model has convolution layers and residual blocks to process the board, then separate policy and value outputs. During training, the program compares its predictions with the moves and results from the PGN games.

## Project files and useful functions

Most of the program is in [chess_project.py](./chess_project.py). It contains the board helpers, neural-network classes, PGN dataset, training code, prediction command, and both graphical modes.

For example, the board helper functions can be used from another Python file:

```python
import chess
from chess_project import create_environment, encode_board

board = create_environment()
board.push_uci("e2e4")
planes = encode_board(board)

print(board)
print(planes.shape)
```

Some useful names in the project are:

- `create_environment` creates a new chessboard.
- `encode_board` and `board_to_tensor` prepare the board for the model.
- `ChessNet` is the neural network.
- `ChessDataset` reads positions from PGN games.
- `choose_move` chooses the model's highest-scoring legal move.
- `train_model` trains and saves a model.

## Things I know could be better

- The agent does not search ahead, so it is not a replacement for a real chess engine.
- The model does not have a complete representation for special chess rules.
- Promotion moves are not fully represented in the model's policy.
- PGN training reports training loss but does not use a validation set.
- Self-play is a simple experiment; the agent's strength can go up or down.
- Larger training jobs can take a while, especially on a CPU.

## If something goes wrong

### A package cannot be imported

Make sure the packages are installed in the same Python environment you use to run the script:

```bash
python -m pip install python-chess numpy torch pygame
```

### The PGN file cannot be found

Check the filename or give its full path:

```bash
python chess_project.py train --pgn "C:\data\games.pgn"
```

### There are no training positions

Check that the PGN file contains games with a known result and usable moves.

### The model checkpoint cannot be found

Train a model first, or pass the path to an existing checkpoint with `--checkpoint`.

## Quick commands

```bash
python chess_project.py --help
python chess_project.py
python chess_project.py play
python chess_project.py self-play
python chess_project.py demo
python chess_project.py train --pgn games.pgn
python chess_project.py predict --checkpoint chess_model.pth
```
