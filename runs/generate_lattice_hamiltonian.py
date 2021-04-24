import argparse
import numpy as np

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_x',
        type=int,
        help='Extent of lattice in X direction (default = 5)',
        default=5
    )
    parser.add_argument(
        '--num_y',
        type=int,
        help='Extent of lattice in Y direction (default = 5)',
        default=5
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='File where generated lattice will be saved (default = hamiltonian_<NUM_X>x<NUM_Y>.yaml)',
        default=None
    )
    parser.add_argument(
        '--vertex_weight',
        type=float,
        help='Constant weight assigned to each vertex of the lattice',
        default=5.0
    )
    parser.add_argument(
        '--random_edge_weights',
        help='Whether to used random edge weights. Weights will be drawn from Uniform(-1, 1) distribution',
        action='store_true'
    )
    parser.add_argument(
        '--constant_edge_weight',
        type=float,
        help='If not using random edge weights, the constant weight to assign to each edge (default = 1.0)',
        default=1.0
    )
    args = parser.parse_args()

    if (args.output_file is None):
        args.output_file = 'lattice_{:d}x{:d}_hamiltonian.yaml'.format(
            args.num_x, args.num_y)

    with open(args.output_file, 'w') as f:
        f.write('---\n\n')
        f.write('num_rotor: {:d}\n\n'.format(args.num_x * args.num_y))
        f.write('vertex_weight: {:e}\n\n'.format(args.vertex_weight))
        f.write('edges:\n')
        for y in range(args.num_y):
            for x in range(args.num_x):
                j = x + args.num_x * y

                for (dx, dy) in [(1, 0), (0, 1)]:
                    x_next = x + dx
                    y_next = y + dy

                    if x_next < args.num_x and y_next < args.num_y:
                        k = x_next + args.num_x * y_next

                        if args.random_edge_weights:
                            beta = np.random.uniform(-1, 1)
                        else:
                            beta = args.constant_edge_weight

                        f.write('- j: {:d}\n'.format(j))
                        f.write('  k: {:d}\n'.format(k))
                        f.write('  beta: {:e}\n'.format(beta))
