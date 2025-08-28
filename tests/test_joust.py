import unittest
import json
import subprocess


class TestJoustGameHTML(unittest.TestCase):
    """Validate the Joust game logic implemented in JavaScript."""

    def run_node(self, script: str) -> str:
        result = subprocess.run([
            "node",
            "-e",
            script
        ], capture_output=True, text=True, check=True)
        return result.stdout.strip()

    def test_html_logic(self):
        node_script = r"""
const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync('examples/joust_game/joust.html', 'utf8');
const match = /<script>([\s\S]*?)<\/script>/m.exec(html);
if (!match) {
  throw new Error('Script not found');
}
const scriptContent = match[1];
const context = { module: { exports: {} }, exports: {} };
context.global = context;
vm.createContext(context);
vm.runInContext(scriptContent, context);
const { JoustGame } = context.module.exports;
let results = [];
let game = new JoustGame(100, 10, 1);
results.push(game.playerSpeed);
game.level = 2;
game.updatePlayerSpeed();
results.push(game.playerSpeed);
game.level = 7;
game.updatePlayerSpeed();
results.push(game.playerSpeed);

game.player.x = 95;
game.movePlayer('right');
results.push(game.player.x);

game.enemy.x = 95;
game.enemy.direction = 'right';
game.moveEnemy();
results.push(game.enemy.x);
results.push(game.enemy.direction);

console.log(JSON.stringify(results));
"""
        output = self.run_node(node_script)
        results = json.loads(output)
        self.assertAlmostEqual(results[0], 13.0)
        self.assertAlmostEqual(results[1], 12.5)
        self.assertAlmostEqual(results[2], 10.0)
        self.assertAlmostEqual(results[3], 8.0)
        self.assertAlmostEqual(results[4], 5.0)
        self.assertEqual(results[5], 'right')


if __name__ == '__main__':
    unittest.main()
