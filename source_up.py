import numpy as np
import formationplanning.harmonic_parametric_constraints as solver
import formationplanning.plot_paper as plotter
import time
from pathlib import Path

def generate_paths(directory):
    # max number of iterations before we give up on convergence.
    iterations = 200000
    # flux constraint
    alpha = 1000
    # connectivity constraint
    beta = 40
    constraint_w = (alpha, beta)

    # initial positions of the particles. Base of the cubic swarm
    h = 20
    # edge length between connected particles
    d = 5
    # location of the target
    target = np.array([0, 0, 200], dtype=np.float64)

    particles = solver.Particles(d, h, target)

    phi_t, phi_ref_t , d_phi_t, vertices_2_t, followers_2_t, index = solver.solve_constraints(particles, iterations, constraint_w)

    # save the output data
    np.save(directory / 'phi_t', phi_t[:index])
    np.save(directory / 'phi_ref_t', phi_ref_t[:index])
    np.save(directory / 'd_phi_t', d_phi_t[:index])
    np.save(directory / 'vertices_2_t', vertices_2_t[:index])
    np.save(directory / 'followers_2_t', followers_2_t[:index])
    np.save(directory / 'index', index)
    np.save(directory / 'p', target)

if __name__ == '__main__':
    stem = Path(__file__).stem
    directory = Path(stem)
    directory.mkdir(exist_ok=True)
    
    generate_paths(directory)
    plotter.plot(directory, save=True)