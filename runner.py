import itertools
import subprocess
import os

cwd = os.getcwd()

# model = 'no_stn'
# nepoch = 20
# save_interval = 5
# test_interval = 1

angles = [0, 30, 45, 60, 90]
translate = [0, 0.1, 0.2, 0.3]

for angle, trans in itertools.product(angles, translate):
    model = 'no_stn'
    out = subprocess.run(['python', 'experiments.py', '--model', model,
                          '--angle', str(angle), '--trans', str(trans)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd)
    with open('logs/%s.log' % '_'.join(out.args), 'wb') as file:
        file.write(out.stdout)

    model = 'stn_tps'
    # nepoch = 300

    for grid_size in [6, 8, 10]:
        out = subprocess.run(['python', 'experiments.py', '--model', model, '--grid-size', str(grid_size),
                              '--angle', str(angle), '--trans', str(trans)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd)
        with open('logs/%s.log' % '_'.join(out.args), 'wb') as file:
            file.write(out.stdout)

    # out = subprocess.run(['python', 'experiments.py', '--model', model, '--nepoch', str(nepoch),
    #                       '--grid-size', str(10)],
    #                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd)
    # with open('logs/%s.log' % '_'.join(out.args), 'wb') as file:
    #     file.write(out.stdout)

    model = 'stn_affine'
    # nepoch = 300
    out = subprocess.run(['python', 'experiments.py', '--model', model,
                          '--angle', str(angle), '--trans', str(trans)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd)
    with open('logs/%s.log' % '_'.join(out.args), 'wb') as file:
        file.write(out.stdout)
