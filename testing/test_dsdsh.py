'''
dsdsh, the shell wrappers around dsd

It used to discard the exit status of the scripts it runs, and the scripts carried on past a failed
step, so a failure looked like a success to whatever drove dsdsh, a notebook cell or a pipeline.
'''
import os
import subprocess
import sys

from analysis_scripts import wrapper

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fake_dsd(tmp_path, fail_on=None, status=3):
    '''A dsd on PATH that records how it was called and exits with status for the subcommand fail_on'''
    bindir = tmp_path/'bin'
    bindir.mkdir()
    calls = tmp_path/'calls.log'
    script = bindir/'dsd'
    script.write_text('#!/bin/sh\n'
                      f'echo "$1" >> "{calls}"\n'
                      f'[ "$1" = "{fail_on}" ] && exit {status}\n'
                      'exit 0\n')
    script.chmod(0o755)
    return bindir, calls


def dsdsh(bindir, *argv):
    env = dict(os.environ, PATH=f'{bindir}{os.pathsep}{os.environ["PATH"]}', PYTHONPATH=REPO)
    return subprocess.run([sys.executable, '-m', 'analysis_scripts', *map(str, argv)],
                          env=env, cwd=REPO, capture_output=True, text=True)


def test_a_failed_command_fails_dsdsh(tmp_path):
    bindir, calls = fake_dsd(tmp_path, fail_on='eval_vol')
    assert dsdsh(bindir, 'eval_vol', tmp_path/'run', 0, 'kmeans', 4, 0).returncode == 3
    assert calls.read_text().split() == ['eval_vol']


def test_a_successful_command_succeeds(tmp_path):
    bindir, calls = fake_dsd(tmp_path)
    assert dsdsh(bindir, 'eval_vol', tmp_path/'run', 0, 'kmeans', 4, 1.5).returncode == 0
    assert calls.read_text().split() == ['eval_vol']


def test_scripts_stop_at_the_first_failed_step(tmp_path):
    '''prepare parses the ctf from the starfile after the poses, it used to do so even when parsing the poses failed'''
    bindir, calls = fake_dsd(tmp_path, fail_on='parse_pose_star')
    assert dsdsh(bindir, 'prepare', tmp_path/'particles.star', 128, 1.5).returncode == 3
    assert calls.read_text().split() == ['parse_pose_star']


def test_python_scripts_report_their_failure_too(tmp_path):
    bindir, _ = fake_dsd(tmp_path)
    result = dsdsh(bindir, 'create_mask', tmp_path/'missing.mrc', '-o', tmp_path/'mask.mrc')
    assert result.returncode != 0
    assert not (tmp_path/'mask.mrc').exists()


def test_a_command_killed_by_a_signal_reports_it_like_a_shell():
    assert wrapper.run(['sh', '-c', 'kill -9 $$']) == 128 + 9
