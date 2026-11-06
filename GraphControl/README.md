# GraphControl

Multi-agent velocity consensus with centralized, distributed, and imitation-trained graph neural network controllers.

From this folder:

```sh
python -m pip install -r requirements.txt
python run_examples.py --epochs 50 --seed 0
```

Runs Parts 2–5 with 100 agents and 100 time steps. Saves figures, logs, and checkpoints to `outputs/latest/`. Use `--parts 2` for only the classical controllers.

One CPU run (November 5, 2026; seed 0; 50 epochs):

| Script | Controller | Cost ↓ |
| --- | --- | ---: |
| `Part2.py` | Centralized | 2.589 |
| `Part2.py` | Distributed | 133.361 |
| `Part3.py` | GNN, fixed graph | 70.928 |
| `Part4.py` | GNN, moving graph | 79.577 |
| `Part5.py` | GNN, collision repulsion | 73.686 |

[Example trajectory](results/example-2026-11-05/part5/figures/positions.png) · [Run details](results/example-2026-11-05/summary.json). Costs combine velocity error and control effort on separate random rollouts; collision safety is not scored.

Includes portable paths and corrected speed plots.
