

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


BOARD_SIZE = 8
PIECE_TYPES = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)
INPUT_CHANNELS = 12
POLICY_SIZE = 64 * 64
SELF_PLAY_MAX_PLIES = 60


def create_environment() -> chess.Board:
    """Create a new chess game in the standard starting position."""
    return chess.Board()


def display_board(board: chess.Board) -> None:
    """Print a position, the player to move, and its FEN representation."""
    print(board)
    print("\nTurn:", "White" if board.turn == chess.WHITE else "Black")
    print("FEN:", board.fen())


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode pieces as 12 binary planes with shape (12, 8, 8).

    Planes 0-5 represent White pawn through king; planes 6-11
    represent Black pawn through king. Row zero is rank 8.
    """
    encoded = np.zeros(
        (INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE),
        dtype=np.float32,
    )

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        piece_plane = PIECE_TYPES.index(piece.piece_type)
        if piece.color == chess.BLACK:
            piece_plane += len(PIECE_TYPES)

        row = BOARD_SIZE - 1 - chess.square_rank(square)
        column = chess.square_file(square)
        encoded[piece_plane, row, column] = 1.0

    return encoded


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Return one encoded position with shape (1, 12, 8, 8)."""
    return torch.from_numpy(encode_board(board)).unsqueeze(0)


def move_to_index(move: chess.Move) -> int:
    """Map a non-promotion move to its index in the 4096-way policy."""
    if move.promotion is not None:
        raise ValueError("Promotion moves are not supported by this policy.")
    return move.from_square * 64 + move.to_square


def index_to_move(index: int) -> chess.Move:
    """Convert a policy index back into a non-promotion chess move."""
    if not 0 <= index < POLICY_SIZE:
        raise ValueError(f"Policy index must be between 0 and {POLICY_SIZE - 1}.")
    return chess.Move(index // 64, index % 64)


def create_legal_mask(
    board: chess.Board,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, dict[int, chess.Move]]:
    """Return a boolean 4096-entry mask and indexes for legal non-promotions."""
    mask = torch.zeros(POLICY_SIZE, dtype=torch.bool, device=device)
    indexed_moves: dict[int, chess.Move] = {}

    for move in board.legal_moves:
        if move.promotion is not None:
            continue
        index = move_to_index(move)
        mask[index] = True
        indexed_moves[index] = move

    return mask, indexed_moves


class ResidualBlock(nn.Module):
    """Two convolution layers with an identity skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        return F.relu(output + identity)


class ChessNet(nn.Module):
    """Convolutional policy/value network for an 8-by-8 chess board."""

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        channels: int = 64,
        num_res_blocks: int = 4,
    ) -> None:
        super().__init__()
        if input_channels != INPUT_CHANNELS:
            raise ValueError(f"ChessNet expects {INPUT_CHANNELS} input channels.")
        if channels <= 0 or num_res_blocks < 0:
            raise ValueError("channels must be positive and num_res_blocks non-negative.")

        self.conv = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(num_res_blocks))
        )

        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, POLICY_SIZE)

        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or tuple(x.shape[1:]) != (
            INPUT_CHANNELS,
            BOARD_SIZE,
            BOARD_SIZE,
        ):
            raise ValueError(
                f"Expected input shape (batch, {INPUT_CHANNELS}, 8, 8); "
                f"received {tuple(x.shape)}."
            )

        features = F.relu(self.bn(self.conv(x)))
        features = self.res_blocks(features)

        policy = F.relu(self.policy_conv(features)).flatten(start_dim=1)
        policy = self.policy_fc(policy)

        value = F.relu(self.value_conv(features)).flatten(start_dim=1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        return policy, value


class ChessDataset(Dataset):
    """Read labeled positions from PGN games with known results."""

    def __init__(self, pgn_path: str | Path, max_positions: int = 5000) -> None:
        if max_positions <= 0:
            raise ValueError("max_positions must be greater than zero.")

        self.samples: list[
            tuple[torch.Tensor, int, torch.Tensor, float]
        ] = []
        self.games_read = 0
        self.skipped_games = 0

        with Path(pgn_path).open("r", encoding="utf-8") as pgn_file:
            while len(self.samples) < max_positions:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                self.games_read += 1
                result = game.headers.get("Result", "*")
                result_values = {
                    "1-0": 1.0,
                    "0-1": -1.0,
                    "1/2-1/2": 0.0,
                }
                if result not in result_values:
                    self.skipped_games += 1
                    continue

                white_result = result_values[result]
                board = game.board()

                for move in game.mainline_moves():
                    if move.promotion is None:
                        mask, _ = create_legal_mask(board)
                        policy_target = move_to_index(move)
                        if not mask[policy_target]:
                            raise ValueError(
                                f"PGN target {move.uci()} is not a legal policy move."
                            )

                        value_target = (
                            white_result
                            if board.turn == chess.WHITE
                            else -white_result
                        )
                        self.samples.append(
                            (
                                board_to_tensor(board).squeeze(0),
                                policy_target,
                                mask,
                                value_target,
                            )
                        )

                    board.push(move)
                    if len(self.samples) >= max_positions:
                        break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        board, policy_target, legal_mask, value_target = self.samples[index]
        return (
            board,
            torch.tensor(policy_target, dtype=torch.long),
            legal_mask,
            torch.tensor(value_target, dtype=torch.float32),
        )


def choose_move(
    board: chess.Board,
    model: ChessNet,
    device: torch.device | str = "cpu",
) -> tuple[chess.Move | None, float | None]:
    """Choose the highest-scoring legal policy move and return its value estimate.

    Promotion moves are not represented in the policy. If every legal move is a
    promotion, return the first legal promotion rather than returning no move.
    """
    if board.is_game_over():
        return None, None

    legal_moves = list(board.legal_moves)
    mask, indexed_moves = create_legal_mask(board, device)
    if not indexed_moves:
        promotion_moves = [move for move in legal_moves if move.promotion is not None]
        return (promotion_moves[0], None) if promotion_moves else (None, None)

    model.eval()
    with torch.no_grad():
        policy, value = model(board_to_tensor(board).to(device))
        legal_policy = policy[0].masked_fill(~mask, float("-inf"))
        best_index = int(legal_policy.argmax().item())
    return indexed_moves[best_index], float(value.item())


def train_model(
    pgn_path: str | Path,
    checkpoint_path: str | Path,
    *,
    max_positions: int = 5000,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 0.0003,
    device: torch.device | str = "cpu",
) -> None:
    """Train the policy/value network on labeled games and save its weights."""
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")

    dataset = ChessDataset(pgn_path, max_positions=max_positions)
    print("Games read:", dataset.games_read)
    print("Games skipped (unknown result):", dataset.skipped_games)
    print("Training positions:", len(dataset))
    if not dataset:
        raise ValueError(
            "The PGN file contained no usable positions with known game results."
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ChessNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for boards, target_moves, legal_masks, target_values in loader:
            boards = boards.to(device)
            target_moves = target_moves.to(device)
            legal_masks = legal_masks.to(device)
            target_values = target_values.to(device)

            optimizer.zero_grad()
            policy, value = model(boards)
            policy = policy.masked_fill(~legal_masks, -1e9)

            policy_loss = policy_loss_fn(policy, target_moves)
            value_loss = value_loss_fn(value.squeeze(-1), target_values)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        print(f"Epoch {epoch + 1}/{epochs}: loss = {total_loss / len(loader):.4f}")

    output_path = Path(checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print("Training complete. Weights saved to:", output_path)


def load_model(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> ChessNet:
    """Load a model state dictionary saved by ``train_model``."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")

    model = ChessNet().to(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_demo() -> None:
    """Show board encoding and the shapes of the network's two outputs."""
    board = create_environment()
    for move in ("e2e4", "e7e5", "g1f3"):
        board.push_uci(move)

    display_board(board)
    model = ChessNet().eval()
    with torch.no_grad():
        policy, value = model(board_to_tensor(board))

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("\nPolicy shape:", tuple(policy.shape))
    print("Value:", round(float(value.item()), 4))
    print("Parameters:", f"{parameter_count:,}")
    print("Encoded board shape:", tuple(board_to_tensor(board).shape))


def run_prediction(
    checkpoint_path: str | Path,
    fen: str | None,
    device: torch.device | str,
) -> None:
    """Print the model's best legal move for a FEN or the starting position."""
    board = chess.Board(fen) if fen else create_environment()
    model = load_model(checkpoint_path, device)
    move, value = choose_move(board, model, device)

    if move is None:
        print("This position is already over.")
        print("Result:", board.result())
        return

    print("Position:")
    display_board(board)
    print("\nSelected move:", move.uci())
    if value is not None:
        print("Value estimate (side to move):", f"{value:.3f}")
    if move.promotion is not None:
        print("Note: the network cannot rank promotions; this is the first legal promotion.")


def _device_from_argument(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch installation has no available CUDA device."
        )
    return torch.device(device_name)


def choose_self_play_move(
    board: chess.Board,
    model: ChessNet,
    device: torch.device | str = "cpu",
) -> tuple[chess.Move | None, torch.Tensor]:
    if board.is_game_over(claim_draw=True):
        return None, torch.zeros(POLICY_SIZE, dtype=torch.bool, device=device)

    legal_moves = list(board.legal_moves)
    mask, indexed_moves = create_legal_mask(board, device)
    if not indexed_moves:
        return (
            next(
                (move for move in legal_moves if move.promotion is not None),
                None,
            ),
            mask,
        )

    model.eval()
    with torch.no_grad():
        policy, _ = model(board_to_tensor(board).to(device))
        legal_policy = policy[0].masked_fill(~mask, float("-inf"))
        probabilities = torch.softmax(legal_policy / 0.8, dim=0)
        move_index = int(torch.multinomial(probabilities, 1).item())
    return indexed_moves[move_index], mask


def learn_from_self_play_game(
    model: ChessNet,
    optimizer: torch.optim.Optimizer,
    samples: list[tuple[torch.Tensor, int | None, torch.Tensor, bool]],
    winner: bool | None,
    device: torch.device | str,
) -> float:
    if not samples:
        return 0.0

    if len(samples) > 16:
        sample_indexes = [
            round(index * (len(samples) - 1) / 15)
            for index in range(16)
        ]
        samples = [samples[index] for index in sample_indexes]

    positions = torch.stack([sample[0] for sample in samples]).to(device)
    value_targets = torch.tensor(
        [
            0.0 if winner is None else (1.0 if sample[3] == winner else -1.0)
            for sample in samples
        ],
        dtype=torch.float32,
        device=device,
    )
    policy_samples = [
        (index, sample)
        for index, sample in enumerate(samples)
        if winner is not None
        and sample[3] == winner
        and sample[1] is not None
    ]

    model.train()
    optimizer.zero_grad()
    policy, value = model(positions)
    value_loss = F.mse_loss(value.squeeze(-1), value_targets)
    loss = value_loss

    if policy_samples:
        sample_indexes = torch.tensor(
            [index for index, _ in policy_samples],
            dtype=torch.long,
            device=device,
        )
        move_targets = torch.tensor(
            [sample[1] for _, sample in policy_samples],
            dtype=torch.long,
            device=device,
        )
        legal_masks = torch.stack(
            [sample[2] for _, sample in policy_samples]
        ).to(device)
        legal_policy = policy[sample_indexes].masked_fill(~legal_masks, -1e9)
        loss = loss + F.cross_entropy(legal_policy, move_targets)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.item())


def adjudicate_self_play_material(board: chess.Board) -> bool | None:
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    score = sum(
        piece_values.get(piece.piece_type, 0)
        * (1 if piece.color == chess.WHITE else -1)
        for piece in board.piece_map().values()
    )
    return chess.WHITE if score > 0 else chess.BLACK if score < 0 else None


def run_human_game_window(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    agent_delay: float = 0.35,
) -> None:
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError(
            "The graphical chess game requires pygame. Install it with "
            "'python -m pip install pygame'."
        ) from exc

    if agent_delay < 0:
        raise ValueError("agent_delay must be zero or greater.")

    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cpu":
        torch.set_num_threads(1)

    checkpoint = Path(checkpoint_path)
    model = load_model(checkpoint, device) if checkpoint.is_file() else ChessNet().to(device)
    model.eval()
    board = create_environment()
    selected_square: int | None = None
    move_history: list[str] = []
    next_agent_move_at = 0

    pygame.init()
    screen = pygame.display.set_mode((880, 660))
    pygame.display.set_caption("Chess - Play Against the Agent")
    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont("arial", 30, bold=True)
    info_font = pygame.font.SysFont("arial", 20)
    piece_font = pygame.font.SysFont("arial", 44, bold=True)
    small_font = pygame.font.SysFont("arial", 16)

    square_size = 72
    board_x = 20
    board_y = 20
    board_rect = pygame.Rect(board_x, board_y, square_size * 8, square_size * 8)
    board_colors = ((238, 216, 181), (119, 149, 86))
    restart_rect = pygame.Rect(620, 520, 210, 42)
    board_surface = pygame.Surface((square_size * 8, square_size * 8))
    for row in range(8):
        for column in range(8):
            pygame.draw.rect(
                board_surface,
                board_colors[(row + column) % 2],
                pygame.Rect(
                    column * square_size,
                    row * square_size,
                    square_size,
                    square_size,
                ),
            )

    piece_images: dict[tuple[int, bool], pygame.Surface] = {}
    for color in (chess.WHITE, chess.BLACK):
        for piece_type in PIECE_TYPES:
            piece = chess.Piece(piece_type, color)
            image = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
            center = (square_size // 2, square_size // 2)
            circle_color = (246, 242, 228) if color == chess.WHITE else (55, 61, 68)
            text_color = (25, 30, 35) if color == chess.WHITE else (245, 245, 240)
            pygame.draw.circle(image, circle_color, center, 25)
            glyph = piece_font.render(piece.symbol().upper(), True, text_color)
            image.blit(glyph, glyph.get_rect(center=center))
            piece_images[(piece_type, color)] = image

    def reset_game() -> None:
        nonlocal board, selected_square, next_agent_move_at
        board = create_environment()
        selected_square = None
        move_history.clear()
        next_agent_move_at = pygame.time.get_ticks() + 250

    def apply_human_move(move: chess.Move) -> None:
        nonlocal selected_square, next_agent_move_at
        move_history.append(f"{board.fullmove_number}. {board.san(move)}")
        board.push(move)
        selected_square = None
        next_agent_move_at = pygame.time.get_ticks() + int(agent_delay * 1000)

    def select_square(square: int) -> None:
        nonlocal selected_square
        if board.turn != chess.WHITE or board.is_game_over(claim_draw=True):
            return

        if selected_square is not None:
            candidates = [
                move
                for move in board.legal_moves
                if move.from_square == selected_square and move.to_square == square
            ]
            if candidates:
                move = next(
                    (candidate for candidate in candidates if candidate.promotion == chess.QUEEN),
                    candidates[0],
                )
                apply_human_move(move)
                return

        piece = board.piece_at(square)
        if piece is not None and piece.color == chess.WHITE:
            selected_square = square
        else:
            selected_square = None

    def draw_text(
        text: str,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        screen.blit(font.render(text, True, color), position)

    running = True
    while running:
        now = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    reset_game()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if restart_rect.collidepoint(event.pos):
                    reset_game()
                elif board_rect.collidepoint(event.pos):
                    column = (event.pos[0] - board_x) // square_size
                    row = (event.pos[1] - board_y) // square_size
                    select_square(chess.square(column, 7 - row))

        if (
            running
            and board.turn == chess.BLACK
            and not board.is_game_over(claim_draw=True)
            and now >= next_agent_move_at
        ):
            move, _ = choose_move(board, model, device)
            if move is None:
                raise RuntimeError("The chess agent found no legal move.")
            move_history.append(f"{board.fullmove_number}... {board.san(move)}")
            board.push(move)
            selected_square = None

        screen.fill((31, 35, 40))
        screen.blit(board_surface, (board_x, board_y))

        if board.move_stack:
            last_move = board.peek()
            for square in (last_move.from_square, last_move.to_square):
                column = chess.square_file(square)
                row = 7 - chess.square_rank(square)
                pygame.draw.rect(
                    screen,
                    (189, 190, 105),
                    pygame.Rect(
                        board_x + column * square_size,
                        board_y + row * square_size,
                        square_size,
                        square_size,
                    ),
                )

        legal_targets: set[int] = set()
        if selected_square is not None and board.turn == chess.WHITE:
            legal_targets = {
                move.to_square
                for move in board.legal_moves
                if move.from_square == selected_square
            }
            column = chess.square_file(selected_square)
            row = 7 - chess.square_rank(selected_square)
            pygame.draw.rect(
                screen,
                (246, 246, 105),
                pygame.Rect(
                    board_x + column * square_size,
                    board_y + row * square_size,
                    square_size,
                    square_size,
                ),
                width=4,
            )

        for square in legal_targets:
            column = chess.square_file(square)
            row = 7 - chess.square_rank(square)
            center = (
                board_x + column * square_size + square_size // 2,
                board_y + row * square_size + square_size // 2,
            )
            if board.piece_at(square) is None:
                pygame.draw.circle(screen, (45, 50, 45, 120), center, 10)
            else:
                pygame.draw.circle(screen, (190, 55, 55), center, 31, width=4)

        for square, piece in board.piece_map().items():
            column = chess.square_file(square)
            row = 7 - chess.square_rank(square)
            screen.blit(
                piece_images[(piece.piece_type, piece.color)],
                (board_x + column * square_size, board_y + row * square_size),
            )

        panel_x = 620
        draw_text("Play the Agent", (panel_x, 38), title_font, (240, 240, 240))
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            if outcome is None:
                raise RuntimeError("The finished game has no outcome.")
            result = board.result(claim_draw=True)
            if outcome.winner is None:
                status = f"Draw ({result})"
            elif outcome.winner == chess.WHITE:
                status = "You win!"
            else:
                status = "Agent wins"
            draw_text(status, (panel_x, 100), info_font, (246, 214, 125))
            draw_text(
                outcome.termination.name.replace("_", " ").title(),
                (panel_x, 132),
                small_font,
                (210, 210, 210),
            )
        elif board.turn == chess.WHITE:
            draw_text("Your turn (White)", (panel_x, 100), info_font, (230, 230, 230))
            if board.is_check():
                draw_text("Check!", (panel_x, 132), small_font, (246, 150, 125))
        else:
            draw_text("Agent thinking...", (panel_x, 100), info_font, (230, 230, 230))

        draw_text("Moves", (panel_x, 190), info_font, (230, 230, 230))
        history_lines = move_history[-12:]
        for index, move_text in enumerate(history_lines):
            draw_text(
                move_text,
                (panel_x, 226 + index * 23),
                small_font,
                (205, 210, 215),
            )

        pygame.draw.rect(screen, (73, 110, 75), restart_rect, border_radius=6)
        draw_text("New game (R)", (panel_x + 24, 529), info_font, (250, 250, 250))
        draw_text("Click a piece, then a square", (panel_x, 590), small_font, (180, 190, 200))
        draw_text("Esc: quit", (panel_x, 616), small_font, (180, 190, 200))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def run_self_play_window(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    move_delay: float = 0.0,
) -> None:
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError(
            "The self-play window requires pygame. Install it with "
            "'python -m pip install pygame'."
        ) from exc

    if move_delay < 0:
        raise ValueError("move_delay must be zero or greater.")

    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cpu":
        torch.set_num_threads(1)

    checkpoint = Path(checkpoint_path)
    model = load_model(checkpoint, device) if checkpoint.is_file() else ChessNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    board = create_environment()
    samples: list[tuple[torch.Tensor, int | None, torch.Tensor, bool]] = []
    games_played = 0
    white_wins = 0
    black_wins = 0
    draws = 0
    last_result = "Learning from self-play"
    last_loss: float | None = None
    game_processed = False
    reset_at = 0
    next_move_at = 0
    move_delay_ms = int(move_delay * 1000)

    pygame.init()
    screen = pygame.display.set_mode((880, 660))
    pygame.display.set_caption("Chess Self-Play Learning")
    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont("arial", 30, bold=True)
    info_font = pygame.font.SysFont("arial", 21)
    piece_font = pygame.font.SysFont("arial", 44, bold=True)
    small_font = pygame.font.SysFont("arial", 16)

    square_size = 72
    board_x = 20
    board_y = 20
    board_colors = ((238, 216, 181), (119, 149, 86))
    running = True
    paused = False

    def draw_text(
        text: str,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        screen.blit(font.render(text, True, color), position)

    board_surface = pygame.Surface((square_size * 8, square_size * 8))
    for row in range(8):
        for column in range(8):
            rect = pygame.Rect(
                column * square_size,
                row * square_size,
                square_size,
                square_size,
            )
            pygame.draw.rect(board_surface, board_colors[(row + column) % 2], rect)

    piece_images: dict[tuple[int, bool], pygame.Surface] = {}
    for color in (chess.WHITE, chess.BLACK):
        for piece_type in PIECE_TYPES:
            piece = chess.Piece(piece_type, color)
            image = pygame.Surface(
                (square_size, square_size),
                pygame.SRCALPHA,
            )
            center = (square_size // 2, square_size // 2)
            circle_color = (246, 242, 228) if color == chess.WHITE else (55, 61, 68)
            text_color = (25, 30, 35) if color == chess.WHITE else (245, 245, 240)
            pygame.draw.circle(image, circle_color, center, 25)
            glyph = piece_font.render(piece.symbol().upper(), True, text_color)
            image.blit(glyph, glyph.get_rect(center=center))
            piece_images[(piece_type, color)] = image

    while running:
        now = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused

        if running and not paused:
            if game_processed and now >= reset_at:
                board = create_environment()
                samples.clear()
                game_processed = False
                next_move_at = now
            elif not game_processed and (
                board.is_game_over(claim_draw=True)
                or len(board.move_stack) >= SELF_PLAY_MAX_PLIES
            ):
                outcome = board.outcome(claim_draw=True)
                if outcome is None and len(board.move_stack) < SELF_PLAY_MAX_PLIES:
                    raise RuntimeError("The self-play game ended without an outcome.")
                winner = (
                    outcome.winner
                    if outcome is not None
                    else adjudicate_self_play_material(board)
                )
                games_played += 1
                if winner is None:
                    draws += 1
                    suffix = " at move limit" if outcome is None else ""
                    last_result = f"Game {games_played}: draw{suffix}"
                elif winner == chess.WHITE:
                    white_wins += 1
                    suffix = " by material" if outcome is None else ""
                    last_result = f"Game {games_played}: White wins{suffix}"
                else:
                    black_wins += 1
                    suffix = " by material" if outcome is None else ""
                    last_result = f"Game {games_played}: Black wins{suffix}"

                last_loss = learn_from_self_play_game(
                    model,
                    optimizer,
                    samples,
                    winner,
                    device,
                )
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint)
                game_processed = True
                reset_at = now + 500
            elif not game_processed and now >= next_move_at:
                move, legal_mask = choose_self_play_move(board, model, device)
                if move is None:
                    raise RuntimeError("The self-play agent found no legal move.")

                move_index = move_to_index(move) if move.promotion is None else None
                samples.append(
                    (
                        board_to_tensor(board).squeeze(0),
                        move_index,
                        legal_mask,
                        board.turn,
                    )
                )
                board.push(move)
                next_move_at = pygame.time.get_ticks() + move_delay_ms

        screen.fill((31, 35, 40))
        screen.blit(board_surface, (board_x, board_y))
        if board.move_stack:
            for square in (board.peek().from_square, board.peek().to_square):
                column = chess.square_file(square)
                row = 7 - chess.square_rank(square)
                pygame.draw.rect(
                    screen,
                    (189, 190, 105),
                    pygame.Rect(
                        board_x + column * square_size,
                        board_y + row * square_size,
                        square_size,
                        square_size,
                    ),
                )
        for square, piece in board.piece_map().items():
            column = chess.square_file(square)
            row = 7 - chess.square_rank(square)
            screen.blit(
                piece_images[(piece.piece_type, piece.color)],
                (board_x + column * square_size, board_y + row * square_size),
            )

        panel_x = 620
        draw_text("Self-play", (panel_x, 38), title_font, (240, 240, 240))
        draw_text(f"Games: {games_played}", (panel_x, 100), info_font, (230, 230, 230))
        draw_text(f"White wins: {white_wins}", (panel_x, 138), info_font, (230, 230, 230))
        draw_text(f"Black wins: {black_wins}", (panel_x, 176), info_font, (230, 230, 230))
        draw_text(f"Draws: {draws}", (panel_x, 214), info_font, (230, 230, 230))
        draw_text(last_result, (panel_x, 280), info_font, (230, 230, 230))
        if last_loss is not None:
            draw_text(f"Last training loss: {last_loss:.3f}", (panel_x, 318), small_font, (180, 205, 180))
        draw_text(
            "Paused" if paused else "Space: pause",
            (panel_x, 370),
            small_font,
            (180, 190, 200),
        )
        draw_text("Esc: quit", (panel_x, 398), small_font, (180, 190, 200))
        draw_text("Saved after every game", (panel_x, 450), small_font, (180, 190, 200))
        draw_text(str(checkpoint), (panel_x, 476), small_font, (180, 190, 200))
        pygame.display.flip()
        clock.tick(120)

    pygame.quit()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play chess, train a policy/value network, or use a saved model."
    )
    commands = parser.add_subparsers(dest="command")

    human_game_parser = commands.add_parser(
        "play",
        help="play a graphical game against the neural-network agent",
    )
    human_game_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("self_play_model.pth"),
    )
    human_game_parser.add_argument("--device", default="cpu")
    human_game_parser.add_argument("--agent-delay", type=float, default=0.35)
    self_play_parser = commands.add_parser(
        "self-play",
        help="watch the neural network play and learn from its own games",
    )
    self_play_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("self_play_model.pth"),
    )
    self_play_parser.add_argument("--device", default="cpu")
    self_play_parser.add_argument("--move-delay", type=float, default=0.0)
    commands.add_parser("demo", help="show a model forward-pass example")

    train_parser = commands.add_parser("train", help="train from a PGN games file")
    train_parser.add_argument("--pgn", type=Path, default=Path("pgn.pgn"))
    train_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("chess_model.pth"),
    )
    train_parser.add_argument("--positions", type=int, default=5000)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=0.0003)
    train_parser.add_argument("--device", default="cpu")

    predict_parser = commands.add_parser(
        "predict",
        help="choose a move using a previously trained checkpoint",
    )
    predict_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("chess_model.pth"),
    )
    predict_parser.add_argument("--fen", help="an optional Forsyth-Edwards position")
    predict_parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    command = args.command or "play"

    if command == "play":
        device = _device_from_argument(getattr(args, "device", "cpu"))
        run_human_game_window(
            getattr(args, "checkpoint", Path("self_play_model.pth")),
            device,
            getattr(args, "agent_delay", 0.35),
        )
    elif command == "self-play":
        device = _device_from_argument(args.device)
        run_self_play_window(args.checkpoint, device, args.move_delay)
    elif command == "demo":
        run_demo()
    elif command == "train":
        device = _device_from_argument(args.device)
        train_model(
            args.pgn,
            args.checkpoint,
            max_positions=args.positions,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
        )
    elif command == "predict":
        device = _device_from_argument(args.device)
        run_prediction(args.checkpoint, args.fen, device)


if __name__ == "__main__":
    main()
