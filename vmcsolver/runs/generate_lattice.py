import argparse
import numpy as np

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_x', type=int, help='Extent of lattice in X direction (default = 5)', default=5)
    parser.add_argument('--num_y', type=int, help='Extent of lattice in Y direction (default = 5)', default=5)
    parser.add_argument('--output_file', type=str, help='File where generated lattice will be saved (default = lattice_<NUM_X>_<NUM_Y>.yaml)', default='')
    parser.add_argument('--random_weights', help='Whether to used random edge weights. Weights will be drawn from Uniform(-1, 1) distribution', action='store_true')
    parser.add_argument('--constant_weight', type=float, help='If not using random weights, the constant weight to assign to each edge (default = 1.0)', default=1.0)
    args = parser.parse_args()

    if (args.output_file == ''):
        args.output_file = 'lattice_{:d}_{:d}.yaml'.format(args.num_x, args.num_y)

    with open(args.output_file, 'w') as f:
        for y in range(args.num_y):
            for x in range(args.num_x):
                j = x + args.num_x * y
                
                for (dx, dy) in [(1, 0), (0, 1)]:
                    x_next = x + dx
                    y_next = y + dy
                    
                    if x_next < args.num_x and y_next < args.num_y:
                        k = x_next + args.num_x * y_next

                        if args.random_weights:
                            beta = np.random.uniform(-1, 1)
                        else:
                            beta = args.constant_weight

                        f.write('- j: {:d}\n'.format(j))
                        f.write('  k: {:d}\n'.format(k))
                        f.write('  beta: {:e}\n'.format(beta))
