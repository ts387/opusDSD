'''
The interactive filtering notebook that dsd analyze copies into each analyze.N folder

After a run trained with --ind, z and the cluster labels count only the selected particles, while the
notebook loads the poses, particles and starfile of the whole stack. Its cells used to index those
with positions in the run, so cluster_sel.star, the particle views and the saved selections were of
other particles. The cells are run here as the notebook runs them, on a small stand-in for a run.
'''
import json
import os

import numpy as np
import pytest
import torch

from cryodrgn import dataset, starfile, utils

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
TEMPLATE = os.path.join(os.path.dirname(DATA), '..', 'cryodrgn', 'templates', 'cryoDRGN_filtering_template.ipynb')
STACK = f'{DATA}/hand.mrcs'  # 100 images
N = 100
IND = np.array([7, 3, 99, 0, 42])  # out of order on purpose


def cell(marker):
    with open(TEMPLATE, encoding='utf-8') as f:
        cells = json.load(f)['cells']
    hits = [''.join(c['source']) for c in cells if marker in ''.join(c['source'])]
    assert len(hits) == 1, marker
    return hits[0]


def run_notebook(tmp_path, ind, trained_on=None, starfile_rows=N):
    rng = np.random.default_rng(0)
    euler = rng.uniform(0, 180, (N, 3))
    poses = (rng.normal(size=(N, 3, 3)), rng.uniform(-0.1, 0.1, (N, 2)), euler,
             np.zeros((N, 1, 3)), np.zeros((N, 1, 2)))
    utils.save_pkl(poses, tmp_path/'pose.pkl')
    if ind is not None:
        utils.save_pkl(ind, tmp_path/'ind.pkl')
    star = tmp_path/'particles.star'
    star.write_text('data_\nloop_\n_rlnImageName\n_rlnAngleRot\n' +
                    ''.join(f'{i + 1:06d}@hand.mrcs {i}\n' for i in range(starfile_rows)))  # rot names the particle
    (tmp_path/'analyze.0').mkdir()
    n = len(trained_on if trained_on is not None else (np.arange(N) if ind is None else ind))
    ns = dict(np=np, utils=utils, dataset=dataset, WORKDIR=str(tmp_path), EPOCH=0,
              z=torch.zeros(n, 4), config=dict(dataset_args=dict(
                  particles=STACK, poses=str(tmp_path/'pose.pkl'), datadir=None,
                  ind=None if ind is None else str(tmp_path/'ind.pkl'))))
    exec(cell('# Load poses'), ns)
    exec(cell('# define the original starfile').replace("'/work/jpma/luo/spliceosome/consensus_data.star'",
                                                         repr(str(star))), ns)
    exec(cell('# Load input particles'), ns)
    exec(cell("ind_orig = config['dataset_args'].get('ind')"), ns)
    ns['ind_selected'] = np.array([0, 2])
    exec(cell('#save selected ind to starfile'), ns)
    ns['ind_keep'], ns['ind_bad'] = np.array([1, 2]), np.array([0, 3, 4])
    exec(cell('# Save the selections as indices of the whole particle stack'), ns)
    return ns, euler


@pytest.mark.parametrize('ind', [None, IND], ids=['all_particles', 'ind'])
def test_selections_name_particles_of_the_stack(tmp_path, ind):
    ns, euler = run_notebook(tmp_path, ind)
    orig = np.arange(N) if ind is None else ind
    # the poses and particle images shown are those of the particles of the run
    assert np.array_equal(ns['euler'], euler[orig])
    full = dataset.load_particles(STACK, lazy=True)
    assert np.array_equal(ns['particles'][2].get(), full[orig[2]].get())
    # cluster_sel.star holds the starfile rows of the selected particles
    sel = starfile.Starfile.load(str(tmp_path/'analyze.0'/'cluster_sel.star'))
    assert sel.df['_rlnAngleRot'].astype(float).astype(int).tolist() == orig[[0, 2]].tolist()
    # the saved selections are indices of the whole stack
    assert np.array_equal(utils.load_pkl(tmp_path/'analyze.0'/'ind_keep.2_particles.pkl'), orig[[1, 2]])
    assert np.array_equal(utils.load_pkl(tmp_path/'analyze.0'/'ind_bad.3_particles.pkl'), orig[[0, 3, 4]])


def test_latents_of_another_run_are_reported(tmp_path):
    ns, _ = run_notebook(tmp_path, None)
    ns['z'] = torch.zeros(5, 4)
    with pytest.raises(AssertionError, match='5 latent codes, 100 poses and 100 particles'):
        exec(cell("ind_orig = config['dataset_args'].get('ind')"), ns)


def test_a_run_that_recorded_ind_but_trained_on_every_particle(tmp_path):
    '''train_multi_dist before 67c0848 wrote --ind into the config and then trained on the whole stack'''
    ns, euler = run_notebook(tmp_path, IND, trained_on=np.arange(N))
    assert ns['ind_orig'] is None
    assert np.array_equal(ns['euler'], euler)
    assert np.array_equal(utils.load_pkl(tmp_path/'analyze.0'/'ind_keep.2_particles.pkl'), [1, 2])


def test_running_the_cell_again_changes_nothing(tmp_path):
    ns, euler = run_notebook(tmp_path, IND)
    exec(cell("ind_orig = config['dataset_args'].get('ind')"), ns)
    assert np.array_equal(ns['euler'], euler[IND])
    full = dataset.load_particles(STACK, lazy=True)
    assert np.array_equal(ns['particles'][2].get(), full[IND[2]].get())


def test_the_selection_is_saved_without_a_starfile(tmp_path):
    '''Stacks given as .mrcs come without a starfile: only cluster_sel.star needs one'''
    ns, _ = run_notebook(tmp_path, IND)
    ns.pop('starfile_orig')
    exec(cell("ind_orig = config['dataset_args'].get('ind')"), ns)
    ns['ind_keep'], ns['ind_bad'] = np.array([0]), np.array([1])
    exec(cell('# Save the selections as indices of the whole particle stack'), ns)
    assert np.array_equal(utils.load_pkl(tmp_path/'analyze.0'/'ind_keep.1_particles.pkl'), IND[[0]])


def test_a_starfile_of_other_particles_is_an_error(tmp_path):
    with pytest.raises(AssertionError, match='the starfile lists 50 particles and the particle stack 100'):
        run_notebook(tmp_path, IND, starfile_rows=50)
