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
from cryodrgn.dataset import LazyMRCData, TiltMRCData
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


@pytest.mark.parametrize('ind, message', [([3, 7, 3], 'more than once'),
                                          (np.random.default_rng(1).permutation(N), 'in another order')],
                         ids=['duplicates', 'reordered_full_stack'])
def test_ambiguous_indices_are_an_error(ind, message):
    '''A file of the whole stack and one already filtered would have the same length, so poses could not follow'''
    with pytest.raises(ValueError, match=message):
        LazyMRCData(STACK, ind=np.array(ind))


def test_the_whole_stack_in_order_is_fine():
    data = LazyMRCData(STACK, ind=np.arange(N))
    assert data.N == data.N_full == N


def test_datasets_know_the_size_of_the_unfiltered_stack():
    assert LazyMRCData(STACK, ind=IND).N_full == N
    assert LazyMRCData(STACK).N_full == N
    assert TiltMRCData(STACK, f'{DATA}/hand_tilt.mrcs', ind=IND).N_full == N
    with pytest.raises(ValueError, match='more than once'):
        TiltMRCData(STACK, f'{DATA}/hand_tilt.mrcs', ind=np.array([1, 1]))


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


@pytest.mark.parametrize('rows', ['full', 'filtered'])
def test_files_of_the_whole_stack_are_filtered_and_files_of_the_run_kept(tmp_path, rows):
    '''Given the size of the stack, as the commands pass it, poses and ctf follow the particles of the run'''
    rot, trans, euler = random_poses(N)
    ctf = np.arange(N, dtype=np.float32)[:, None].repeat(9, 1)
    if rows == 'filtered':
        rot, trans, euler, ctf = rot[IND], trans[IND], euler[IND], ctf[IND]
    data = LazyMRCData(STACK, norm=(0, 1), ind=IND)
    tracker = PoseTracker.load(write_pkl(tmp_path/'pose.pkl', (rot, trans, euler)), data.N, D, None, IND,
                               Nfull=data.N_full)
    full_euler = random_poses(N)[2]
    assert torch.allclose(tracker.eulers, torch.tensor(full_euler[IND]).float())
    assert np.array_equal(utils.select_rows(ctf, IND, data.N_full, 'ctf')[:, 0], IND)


def test_indices_saved_as_a_column_select_the_same_particles(tmp_path):
    '''np.argwhere gives indices of shape (K, 1), which the stack flattens; the poses and ctf have to as well'''
    rot, trans, euler = random_poses(N)
    tracker = PoseTracker.load(write_pkl(tmp_path/'pose.pkl', (rot, trans, euler)), len(IND), D, None, IND[:, None],
                               Nfull=N)
    assert torch.allclose(tracker.eulers, torch.tensor(euler[IND]).float())
    assert np.array_equal(utils.select_rows(np.arange(N), IND[:, None], N, 'ctf'), IND)


def test_per_particle_files_of_another_selection_are_an_error(tmp_path):
    '''A pose.N.pkl of a run with a different --ind used to be indexed as if it were the whole stack'''
    rot, trans, euler = (x[:60] for x in random_poses(N))
    with pytest.raises(ValueError, match='the rotations has 60 rows, but the particle stack has 100'):
        PoseTracker.load(write_pkl(tmp_path/'pose.pkl', (rot, trans, euler)), len(IND), D, None, IND, Nfull=N)
    with pytest.raises(ValueError, match='ctf.pkl has 60 rows'):
        utils.select_rows(np.zeros((60, 9)), IND, N, 'ctf.pkl')


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
