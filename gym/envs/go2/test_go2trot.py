from types import SimpleNamespace

import torch
import pytest

from gym.envs.go2.go2trot import Go2Trot
from gym.envs.go2.go2trot_config import Go2TrotCfg


def test_contact_strength_uses_each_environments_applied_total_mass():
    task = Go2Trot.__new__(Go2Trot)
    task.feet_indices = torch.arange(4)
    task.contact_forces = torch.zeros(2, 4, 3)
    task.contact_forces[:, :, 2] = 98.1
    task._backend = SimpleNamespace(link_mass=torch.tensor([[5.0, 5.0], [10.0, 10.0]]))

    strength = task._foot_contact_strength()

    torch.testing.assert_close(strength[0], torch.ones(4))
    torch.testing.assert_close(strength[1], torch.full((4,), 0.5))


def test_action_rate_penalizes_alternating_targets():
    task = Go2Trot.__new__(Go2Trot)
    task.num_actuators = 12
    # q(t) == q(t-2) must not hide a large change since the previous command.
    task.dof_pos_history = torch.tensor([[0.2] * 12 + [-0.2] * 12 + [0.2] * 12])
    assert task._reward_action_rate().item() == pytest.approx(-0.16)


def test_yaw_tracking_uses_velocity_error_not_its_square():
    task = Go2Trot.__new__(Go2Trot)
    task.cfg = Go2TrotCfg()
    task.commands = torch.zeros(3, 3)
    task.base_ang_vel = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, -0.5]]
    )
    # Keep the existing 2.5 rad/s error scale; square the error only once.
    torch.testing.assert_close(
        task._reward_tracking_ang_vel(),
        torch.tensor(
            [1.0, torch.exp(torch.tensor(-0.16)), torch.exp(torch.tensor(-0.16))]
        ),
    )
