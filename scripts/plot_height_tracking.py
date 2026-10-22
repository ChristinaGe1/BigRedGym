"""Plot how well a trained Go2Trot policy tracks the commanded base height.

By default, plays the latest checkpoint headless, steps the height command
through low-high-low-high while standing in place, and saves a time-series
plot of commanded vs. actual base height.

With --log, plots a session recorded with scripts/play.py --log_height
instead, i.e. whatever you commanded with the arrow keys.

Usage:
    uv run --frozen python -m scripts.plot_height_tracking
    uv run --frozen python -m scripts.plot_height_tracking --hold 4 \
                           --output height.pdf --load_run <run> --checkpoint 500
    uv run --frozen python -m scripts.plot_height_tracking --log height_log.csv

Arguments other than --hold, --output and --log are passed on to
scripts/play.py.
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from scripts.play import get_play_args, setup  # noqa: E402

# Error statistics ignore this long after each command change (the transition).
SETTLE_S = 1.0


def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hold", type=float, default=3.0, help="seconds per height")
    parser.add_argument("--output", type=str, default="height_tracking.pdf")
    parser.add_argument(
        "--log", type=str, default=None, help="CSV from play.py --log_height"
    )
    own_args, play_argv = parser.parse_known_args()
    if own_args.log is not None:
        return own_args, None
    if not any(arg.startswith("--task") for arg in play_argv):
        play_argv.append("--task=go2trot")
    play_args = get_play_args(play_argv + ["--headless"])
    return own_args, play_args


def rollout(env, runner, hold_s):
    low, high = env.cfg.commands.ranges.height
    steps_per_hold = int(hold_s / env.dt)
    time, commanded, actual = [], [], []
    for target in (low, high, low, high):
        for _ in range(steps_per_hold):
            # Stand in place; only the height command changes.
            env.commands[:] = 0.0
            env.height_command[:] = target
            runner.set_actions(
                runner.actor_cfg["actions"],
                runner.get_inference_actions(),
                runner.actor_cfg["disable_actions"],
            )
            env.step()
            time.append(len(time) * env.dt)
            commanded.append(target)
            actual.append(env.base_height.flatten().cpu().clone())
    return torch.tensor(time), torch.tensor(commanded), torch.stack(actual)


def load_log(path):
    with open(path, newline="") as f:
        rows = [tuple(map(float, row)) for row in list(csv.reader(f))[1:]]
    if not rows:
        raise SystemExit(f"{path} has no data")
    time, commanded, actual = torch.tensor(rows).T
    return time - time[0], commanded, actual.unsqueeze(1)


def command_segments(commanded):
    """[start, end) index ranges over which the command is constant."""
    changes = (commanded[1:] != commanded[:-1]).nonzero().flatten() + 1
    bounds = [0, *changes.tolist(), len(commanded)]
    return list(zip(bounds[:-1], bounds[1:]))


def plot(time, commanded, actual, output):
    mean = actual.mean(dim=1)
    std = actual.std(dim=1) if actual.shape[1] > 1 else torch.zeros_like(mean)
    segments = command_segments(commanded)
    settled = torch.zeros_like(commanded, dtype=torch.bool)
    for start, end in segments:
        settled[start:end] = time[start:end] >= time[start] + SETTLE_S
    error = actual[settled] - commanded[settled].unsqueeze(1)
    rmse_mm = 1000 * error.square().mean().sqrt().item()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(time, commanded, where="post", color="k", linestyle="--", label="commanded")
    if actual.shape[1] > 1:
        label = f"actual (mean of {actual.shape[1]} envs)"
    else:
        label = "actual"
    ax.plot(time, mean, color="C0", label=label)
    if actual.shape[1] > 1:
        ax.fill_between(
            time, mean - std, mean + std, color="C0", alpha=0.25, label="±1 std"
        )
    ax.set_xlabel("time [s]")
    ax.set_ylabel("base height [m]")
    ax.set_title(f"Go2 base-height tracking (settled RMSE {rmse_mm:.1f} mm)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output)
    print(
        f"settled RMSE ({SETTLE_S:.0f} s after each command change): {rmse_mm:.1f} mm"
    )
    for start, end in segments:
        window = settled[start:end]
        if window.any():
            print(
                f"  t={time[start]:5.1f}s  commanded {commanded[start]:.3f} m"
                f"  actual {actual[start:end][window].mean():.3f} m"
            )
    print(f"saved {output}")


if __name__ == "__main__":
    own_args, play_args = get_args()
    if own_args.log is not None:
        time, commanded, actual = load_log(own_args.log)
    else:
        with torch.no_grad():
            env, runner = setup(play_args)
            time, commanded, actual = rollout(env, runner, own_args.hold)
    plot(time, commanded, actual, own_args.output)
