'''
--ind, training on a subset of the particle stack

Everything indexed per particle, the images, poses, ctf and group assignments, has to be filtered
with the same indices: the i-th image of the filtered stack is trained with the pose and ctf of
particle ind[i], and z.N.pkl and pose.N.pkl list the particles in that order for dsd analyze.
LazyMRCData used to accept ind and ignore it, which train_multi only noticed when the filtered ctf
no longer matched the size of the stack.
'''
import argparse
import os
import pickle

import numpy as np
import pytest
import torch

from cryodrgn import utils
from cryodrgn.commands import parse_pose_star
from cryodrgn.dataset import LazyMRCData
from cryodrgn.pose import PoseTracker

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
STACK = f'{DATA}/hand.mrcs'  # 100 images of 64x64
N, D = 100, 64
IND = np.array([7, 3, 99, 0, 42])  # out of order on purpose


@pytest.mark.parametrize('kwargs', [dict(), dict(in_mem=False), dict(real_data=False, in_mem=False)],
                         ids=['in_mem', 'lazy', 'hartley'])
def test_lazy_data_keeps_the_indexed_particles_in_order(kwargs):
    full = LazyMRCData(STACK, norm=(0, 1), **kwargs)
    sub = LazyMRCData(STACK, norm=(0, 1), ind=IND, **kwargs)
    assert sub.N == len(sub) == len(IND)
    assert sub.D == full.D
    for i, j in enumerate(IND):
        assert np.array_equal(sub.get(i), full.get(j))
    # the dataloader hands the trainers positions in the filtered stack
    assert sub[2][1] == 2


@pytest.mark.parametrize('ind', [[0, N], [-1, 3]])
def test_indices_outside_the_stack_are_an_error(ind):
    with pytest.raises(ValueError, match=f'only {N} images'):
        LazyMRCData(STACK, ind=np.array(ind))


def write_pkl(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return str(path)


def random_poses(n, bodies=None):
    rng = np.random.default_rng(0)
    euler = np.stack([rng.uniform(-180, 180, n), rng.uniform(0, 180, n), rng.uniform(-180, 180, n)], 1)
    rot = np.asarray([utils.R_from_relion(*e) for e in euler])
    trans = rng.uniform(-0.1, 0.1, (n, 2))
    if bodies is None:
        return rot, trans, euler
    return rot, trans, euler, rng.uniform(-180, 180, (n, bodies, 3)), rng.uniform(-0.1, 0.1, (n, bodies, 2))


def test_poses_follow_the_indices(tmp_path):
    '''The eulers were only filtered when the already filtered translations were still too long'''
    rot, trans, euler = random_poses(N)
    tracker = PoseTracker.load(write_pkl(tmp_path/'pose.pkl', (rot, trans, euler)), len(IND), D, None, IND)
    assert torch.allclose(tracker.rots, torch.tensor(rot[IND]).float())
    assert torch.allclose(tracker.trans, torch.tensor(trans[IND]*D).float())
    assert torch.allclose(tracker.eulers, torch.tensor(euler[IND]).float())


def test_multibody_poses_follow_the_indices(tmp_path):
    rot, trans, euler, body_euler, body_trans = random_poses(N, bodies=2)
    path = write_pkl(tmp_path/'pose.pkl', (rot, trans, euler, body_euler, body_trans))
    tracker = PoseTracker.load(path, len(IND), D, None, IND)
    assert torch.allclose(tracker.eulers, torch.tensor(euler[IND]).float())
    assert torch.allclose(tracker.body_eulers, torch.tensor(body_euler[IND]).float())
    assert torch.allclose(tracker.body_trans, torch.tensor(body_trans[IND]).float())


def test_poses_already_filtered_are_kept(tmp_path):
    '''pose.N.pkl written by a run with --ind lists only the particles of that run'''
    rot, trans, euler = (x[IND] for x in random_poses(N))
    tracker = PoseTracker.load(write_pkl(tmp_path/'pose.pkl', (rot, trans, euler)), len(IND), D, None, IND)
    assert torch.allclose(tracker.eulers, torch.tensor(euler).float())


def test_poses_without_eulers_load(tmp_path):
    '''(rot, trans) is what cryodrgn writes, and what backproject_voxel is usually given'''
    rot, trans, _ = random_poses(N)
    tracker = PoseTracker.load(write_pkl(tmp_path/'pose.pkl', (rot, trans)), len(IND), D, None, IND)
    assert tracker.eulers is None
    assert torch.allclose(tracker.trans, torch.tensor(trans[IND]*D).float())


STAR = '''data_
loop_
_rlnImageName
_rlnAngleRot
_rlnAngleTilt
_rlnAnglePsi
_rlnOriginX
_rlnOriginY
'''


def parse_poses(tmp_path, labels):
    star = tmp_path/'particles.star'
    star.write_text(STAR + ''.join(f'{i+1:06d}@particles.mrcs {10*i} {5*i} {-3*i} 0 0\n' for i in range(5)))
    parser = argparse.ArgumentParser()
    parse_pose_star.add_args(parser)
    args = parser.parse_args([str(star), '-D', '64', '--labels', write_pkl(tmp_path/'labels.pkl', np.array(labels)),
                              '--outdir', str(tmp_path)])
    parse_pose_star.main(args)


def test_labels_split_the_starfile(tmp_path):
    parse_poses(tmp_path, [0, 1, 1, 0, 1])
    assert len((tmp_path/'pre0.star').read_text().split('@')) - 1 == 2
    assert len((tmp_path/'pre1.star').read_text().split('@')) - 1 == 3


def test_labels_of_a_filtered_run_do_not_split_the_full_starfile(tmp_path):
    '''labels.pkl of a run trained with --ind has one label per particle in that run'''
    with pytest.raises(ValueError, match='labels 3 particles but .* has 5'):
        parse_poses(tmp_path, [0, 1, 1])
