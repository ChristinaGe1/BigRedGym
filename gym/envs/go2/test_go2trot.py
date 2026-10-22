from types import SimpleNamespace

import torch

from gym.envs.go2.go2trot import Go2Trot


def test_contact_strength_uses_each_environments_applied_total_mass():
    task = Go2Trot.__new__(Go2Trot)
    task.feet_indices = torch.arange(4)
    task.contact_forces = torch.zeros(2, 4, 3)
    task.contact_forces[:, :, 2] = 98.1
    task._backend = SimpleNamespace(link_mass=torch.tensor([[5.0, 5.0], [10.0, 10.0]]))

    strength = task._foot_contact_strength()

    torch.testing.assert_close(strength[0], torch.ones(4))
    torch.testing.assert_close(strength[1], torch.full((4,), 0.5))


def test_height_tracking_reward_peaks_at_the_commanded_height():
    task = Go2Trot.__new__(Go2Trot)
    task.cfg = SimpleNamespace(
        reward_settings=SimpleNamespace(tracking_height_sigma=0.01)
    )
    task.height_command = torch.tensor([[0.2], [0.4], [0.3]])
    task.base_height = torch.tensor([[0.2], [0.3], [0.3]])

    reward = task._reward_tracking_height()

    torch.testing.assert_close(
        reward, torch.tensor([1.0, torch.exp(torch.tensor(-1.0)).item(), 1.0])
    )
