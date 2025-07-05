from dataclasses import dataclass

@dataclass
class Player:
    """Represents the player character."""
    x: float
    speed: float


@dataclass
class Enemy:
    """Represents a single enemy."""
    x: float
    speed: float
    direction: str  # 'left' or 'right'


class JoustGame:
    """Simple Joust-like horizontal movement simulation."""

    def __init__(self, width: int, enemy_speed: float = 5.0, level: int = 1):
        self.width = width
        self.enemy_speed = enemy_speed
        self.level = level
        self.player_speed = 0.0
        self.update_player_speed()
        self.player = Player(x=width / 2, speed=self.player_speed)
        self.enemy = Enemy(x=0.0, speed=self.enemy_speed, direction="right")

    def update_player_speed(self) -> None:
        """Calculate player speed relative to the enemy."""
        diff = 0.30 - 0.05 * (self.level - 1)
        if diff < 0:
            diff = 0
        self.player_speed = self.enemy_speed * (1 + diff)

    def wrap_position(self, x: float) -> float:
        """Wrap a position around the screen width."""
        if x < 0:
            return self.width + x
        if x >= self.width:
            return x - self.width
        return x

    def move_player(self, direction: str) -> None:
        """Move the player left or right with wrap-around."""
        if direction == "left":
            self.player.x -= self.player.speed
        else:
            self.player.x += self.player.speed
        self.player.x = self.wrap_position(self.player.x)

    def move_enemy(self) -> None:
        """Move the enemy in its set direction with wrap-around."""
        if self.enemy.direction == "left":
            self.enemy.x -= self.enemy.speed
            if self.enemy.x < 0:
                self.enemy.x = self.wrap_position(self.enemy.x)
        else:
            self.enemy.x += self.enemy.speed
            if self.enemy.x >= self.width:
                self.enemy.x = self.wrap_position(self.enemy.x)

