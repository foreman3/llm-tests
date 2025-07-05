import unittest
from examples.joust_game.joust import JoustGame


class TestJoustGame(unittest.TestCase):
    def test_speed_difference(self):
        game = JoustGame(width=100, enemy_speed=10, level=1)
        self.assertAlmostEqual(game.player_speed, 13.0)
        game.level = 2
        game.update_player_speed()
        self.assertAlmostEqual(game.player_speed, 12.5)
        game.level = 7
        game.update_player_speed()
        self.assertAlmostEqual(game.player_speed, 10.0)

    def test_wrap_and_direction(self):
        game = JoustGame(width=100, enemy_speed=10, level=1)
        game.player.x = 95
        game.move_player("right")
        self.assertAlmostEqual(game.player.x, 8.0)
        game.enemy.x = 95
        game.enemy.direction = "right"
        game.move_enemy()
        self.assertAlmostEqual(game.enemy.x, 5.0)
        self.assertEqual(game.enemy.direction, "right")


if __name__ == "__main__":
    unittest.main()

