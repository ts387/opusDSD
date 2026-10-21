'''
dsd backproject_voxel, the direct fourier inversion of a particle stack with known poses

It inserts the symmetrized hartley transform of every image into the volume, so it has to ask
LazyMRCData for hartley transforms; the real space images LazyMRCData returns by default are one
pixel smaller per side and made it fail on the first image.
'''
import argparse
import contextlib
import io
import os
import pickle

import numpy as np
import pytest

from cryodrgn import mrc
from cryodrgn.commands import backproject_voxel

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
# toy_projections.mrcs are noiseless 30x30 projections of toymodel_small_nocenter.mrc at these poses
PROJECTIONS = f'{DATA}/toy_projections.mrcs'
POSES = f'{DATA}/toy_rot_zerotrans.pkl'
MODEL = f'{DATA}/toymodel_small_nocenter.mrc'


def backproject(*argv):
    parser = argparse.ArgumentParser()
    backproject_voxel.add_args(parser)
    with contextlib.redirect_stdout(io.StringIO()):
        backproject_voxel.main(parser.parse_args([str(a) for a in argv]))
    return mrc.parse_mrc(argv[argv.index('-o') + 1])


def fsc(a, b):
    '''Fourier shell correlation of two cubic volumes, one value per integer shell up to nyquist'''
    D = a.shape[0]
    fa, fb = np.fft.fftshift(np.fft.fftn(a)), np.fft.fftshift(np.fft.fftn(b))
    x = np.arange(-D//2, D//2)
    r = np.sqrt(sum(g**2 for g in np.meshgrid(x, x, x, indexing='ij'))).round().astype(int)
    return np.array([np.real((fa[r == s]*np.conj(fb[r == s])).sum()) /
                     np.sqrt((np.abs(fa[r == s])**2).sum()*(np.abs(fb[r == s])**2).sum()) for s in range(D//2)])


def write_stack(path, imgs):
    mrc.write(str(path), np.asarray(imgs, dtype=np.float32))
    return path


def write_pkl(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return path


def test_runs_on_the_hand_stack(tmp_path):
    vol, header = backproject(f'{DATA}/hand.mrcs', '--poses', f'{DATA}/hand_rot.pkl', '-o', tmp_path/'hand.mrc')
    assert vol.shape == (64, 64, 64)
    assert np.isfinite(vol).all() and vol.std() > 0


def test_recovers_the_volume_the_projections_were_made_from(tmp_path):
    vol, _ = backproject(PROJECTIONS, '--poses', POSES, '--uninvert-data', '-o', tmp_path/'toy.mrc')
    model, _ = mrc.parse_mrc(MODEL)
    assert (fsc(vol, model) > 0.95).all()


def test_inverting_the_data_inverts_the_volume(tmp_path):
    '''The default, since particles are dark in micrographs'''
    vol, _ = backproject(PROJECTIONS, '--poses', POSES, '--first', 200, '-o', tmp_path/'inv.mrc')
    uninv, _ = backproject(PROJECTIONS, '--poses', POSES, '--first', 200, '--uninvert-data', '-o', tmp_path/'uninv.mrc')
    assert np.allclose(vol, -uninv, atol=1e-4*np.abs(uninv).max())


def test_translations_undo_the_shifts_of_the_images(tmp_path):
    '''Translations are stored as a fraction of the box, the box being the image size'''
    imgs, _ = mrc.parse_mrc(PROJECTIONS)
    rot, _ = pickle.load(open(POSES, 'rb'))
    N, L, _ = imgs.shape
    shifts = np.random.default_rng(0).integers(-8, 9, size=(N, 2))  # x, y in pixels
    shifted = [np.roll(img, (sy, sx), axis=(0, 1)) for img, (sx, sy) in zip(imgs, shifts)]
    stack = write_stack(tmp_path/'shifted.mrcs', shifted)
    poses = write_pkl(tmp_path/'poses.pkl', (rot, -shifts/L))

    vol, _ = backproject(stack, '--poses', poses, '--uninvert-data', '-o', tmp_path/'shifted.mrc')
    model, _ = mrc.parse_mrc(MODEL)
    assert (fsc(vol, model) > 0.95).all()


def test_ind_backprojects_the_selected_particles(tmp_path):
    '''--ind has to pick the same particles out of the stack, the poses and the ctf

    Compared against a stack that only holds those particles, in the order --ind lists them.
    '''
    imgs, _ = mrc.parse_mrc(PROJECTIONS)
    rot, trans = pickle.load(open(f'{DATA}/toy_rot_trans.pkl', 'rb'))
    N, L, _ = imgs.shape
    rng = np.random.default_rng(1)
    ctf = np.zeros((N, 9))
    ctf[:, 0], ctf[:, 1] = L, 6.  # box size and apix
    ctf[:, 2] = rng.uniform(5000, 20000, N)  # dfu
    ctf[:, 3] = ctf[:, 2] + 200  # dfv
    ctf[:, 4:8] = 45., 300., 2.7, 0.1  # dfang, kv, cs, amplitude contrast
    ind = rng.permutation(N)[:300]

    full = backproject(PROJECTIONS, '--poses', write_pkl(tmp_path/'poses.pkl', (rot, trans)),
                       '--ctf', write_pkl(tmp_path/'ctf.pkl', ctf), '--ind', write_pkl(tmp_path/'ind.pkl', ind),
                       '-o', tmp_path/'ind.mrc')
    subset = backproject(write_stack(tmp_path/'subset.mrcs', imgs[ind]),
                         '--poses', write_pkl(tmp_path/'subset_poses.pkl', (rot[ind], trans[ind])),
                         '--ctf', write_pkl(tmp_path/'subset_ctf.pkl', ctf[ind]), '-o', tmp_path/'subset.mrc')
    assert np.allclose(full[0], subset[0], atol=1e-5*np.abs(subset[0]).max())
    assert full[1].get_apix() == pytest.approx(6.)


def toy_ctf(N, L, seed=1):
    rng = np.random.default_rng(seed)
    ctf = np.zeros((N, 9))
    ctf[:, 0], ctf[:, 1] = L, 6.  # box size and apix
    ctf[:, 2] = rng.uniform(5000, 20000, N)  # dfu
    ctf[:, 3] = ctf[:, 2] + 200  # dfv
    ctf[:, 4:8] = 45., 300., 2.7, 0.1  # dfang, kv, cs, amplitude contrast
    return ctf


def test_ind_takes_poses_and_ctf_already_cut_to_the_selection(tmp_path):
    '''Poses and ctf may list the whole stack or only the particles --ind selects, e.g. pose.N.pkl of an --ind run'''
    imgs, _ = mrc.parse_mrc(PROJECTIONS)
    rot, trans = pickle.load(open(f'{DATA}/toy_rot_trans.pkl', 'rb'))
    ctf = toy_ctf(*imgs.shape[:2])
    ind = np.sort(np.random.default_rng(2).permutation(len(imgs))[:300])
    args = ('--ind', write_pkl(tmp_path/'ind.pkl', ind))
    whole = backproject(PROJECTIONS, '--poses', write_pkl(tmp_path/'poses.pkl', (rot, trans)),
                        '--ctf', write_pkl(tmp_path/'ctf.pkl', ctf), *args, '-o', tmp_path/'whole.mrc')
    cut = backproject(PROJECTIONS, '--poses', write_pkl(tmp_path/'cut_poses.pkl', (rot[ind], trans[ind])),
                      '--ctf', write_pkl(tmp_path/'cut_ctf.pkl', ctf[ind]), *args, '-o', tmp_path/'cut.mrc')
    assert np.allclose(whole[0], cut[0], atol=1e-5*np.abs(whole[0]).max())
    with pytest.raises(ValueError, match='has 600 rows'):
        backproject(PROJECTIONS, '--poses', write_pkl(tmp_path/'other_poses.pkl', (rot[:600], trans[:600])),
                    *args, '-o', tmp_path/'other.mrc')
    with pytest.raises(ValueError, match='has 600 rows'):
        backproject(PROJECTIONS, '--poses', tmp_path/'poses.pkl', '--ctf', write_pkl(tmp_path/'other_ctf.pkl', ctf[:600]),
                    *args, '-o', tmp_path/'other.mrc')


def test_ctf_for_a_different_stack_is_an_error(tmp_path):
    with pytest.raises(AssertionError, match='ctf params'):
        backproject(f'{DATA}/hand.mrcs', '--poses', f'{DATA}/hand_rot.pkl', '--ctf', f'{DATA}/ctf2.pkl',
                    '-o', tmp_path/'hand.mrc')
