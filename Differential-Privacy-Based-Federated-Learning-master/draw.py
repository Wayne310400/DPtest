import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from draw_option import args_parser

def openfile(filepath):
    file = open(filepath)
    y = []
    while 1:
        line = file.readline()
        if line.rstrip('\n') == '':
            break
        y.append(float(line.rstrip('\n')))
        if not line:
            break
        pass
    file.close()
    return y

if __name__ == '__main__':
    args = args_parser()

    if args.output == 'epsilon':
        plt.figure()
        plt.ylabel('test accuracy')
        plt.xlabel('global round')
        for epsilon in args.epsilons:
            y = openfile('./log/accfile_fed_{}_{}_{}_iid{}_dp_{}{}_epsilon_{}.dat'.
                format(args.dataset, args.model, args.epoch, args.iid, args.dp_mechanism, args.dp_ratio, epsilon))
            plt.plot(range(args.epoch), y, label=r'$\epsilon={}$'.format(epsilon))
        plt.title('{} {}'.format(args.dataset, args.dp_mechanism))
        plt.legend()
        plt.savefig('{}_{}.png'.format(args.dataset, args.dp_mechanism))

    elif args.output == 'dp_ratio':
        plt.figure()
        plt.ylabel('test accuracy')
        plt.xlabel('global round')
        for dp_ratio in args.dp_ratios:
            y = openfile('./log/accfile_fed_{}_{}_{}_iid{}_dp_{}{}_epsilon_{}.dat'.
                format(args.dataset, args.model, args.epoch, args.iid, args.dp_mechanism, dp_ratio, args.epsilon))
            plt.plot(range(args.epoch), y, label='dp_ratio={}'.format(dp_ratio))
        plt.title('{} {}'.format(args.dataset, args.dp_mechanism))
        plt.legend()
        plt.savefig('{}_{}.png'.format(args.dataset, args.dp_mechanism))
    
    elif args.output == 'drop':
        if args.acc:
            plt.figure()
            plt.ylabel('test accuracy')
            plt.xlabel('global round')
            for drop in args.drops:
                y = openfile('./log/acc/fed_{}_{}_{}_iid{}_dp_{}_epsilon_{}_drop_{}.dat'.
                    format(args.dataset, args.model, args.epoch, args.iid, args.dp_mechanism, args.epsilon, drop))
                plt.plot(range(args.epoch), y, label='drop_ratio={}'.format(drop))
            plt.title('{} {} Accuracy'.format(args.dataset, args.dp_mechanism))
            plt.legend()
            plt.savefig('./log/acc/{}_{}.png'.format(args.dataset, args.dp_mechanism))
            plt.close()

        if args.loss:
            plt.figure()
            plt.ylabel('test accuracy')
            plt.xlabel('global round')
            for drop in args.drops:
                y = openfile('./log/loss/fed_{}_{}_{}_iid{}_dp_{}_epsilon_{}_drop_{}.dat'.
                    format(args.dataset, args.model, args.epoch, args.iid, args.dp_mechanism, args.epsilon, drop))
                plt.plot(range(args.epoch), y, label='drop_ratio={}'.format(drop))
            plt.title('{} {} Loss'.format(args.dataset, args.dp_mechanism))
            plt.legend()
            plt.savefig('./log/loss/{}_{}.png'.format(args.dataset, args.dp_mechanism))
            plt.close()

        plt.figure()
        plt.ylabel('test accuracy')
        plt.xlabel('drop ratio')
        partial_acc, gaussian_acc = [], []
        for drop in args.drops:
            y = openfile('./log/acc/fed_{}_{}_{}_iid{}_dp_{}_epsilon_{}_drop_{}.dat'.
                format(args.dataset, args.model, args.epoch, args.iid, 'Partial', args.epsilon, drop))
            partial_acc.append(y[-1])
        for drop in args.drops:
            y = openfile('./log/acc/fed_{}_{}_{}_iid{}_dp_{}_epsilon_{}_drop_{}.dat'.
                format(args.dataset, args.model, args.epoch, args.iid, 'Gaussian', args.epsilon, drop))
            gaussian_acc.append(y[-1])
        plt.plot(args.drops, partial_acc, label='{}'.format("Proposed Scheme"))
        plt.plot(args.drops, gaussian_acc, label='{}'.format("Gaussian"))
        plt.title('{} Accuracy'.format(args.dataset))
        plt.legend()
        plt.savefig('Compare.png')
        plt.close()

    # plt.figure()
    # epsilon_array = ['1.0', '5.0', '10.0', '20.0', '30.0']
    # plt.ylabel('test accuracy')
    # plt.xlabel('global round')
    # for epsilon in epsilon_array:
    #     y = openfile('./log/accfile_fed_mnist_cnn_100_iidFalse_dp_MA_epsilon_{}.dat'.format(epsilon))
    #     plt.plot(range(100), y, label=r'$\epsilon={}$'.format(epsilon))
    # y = openfile('./log/accfile_fed_mnist_cnn_100_iidFalse_dp_no_dp_epsilon_20.dat'.format(epsilon))
    # plt.plot(range(100), y, label=r'$\epsilon=+\infty$')
    # plt.title('Mnist Gaussian Moment Account (q = 0.01)')
    # plt.legend()
    # plt.savefig('mnist_gaussian_MA.png')

    # plt.figure()
    # epsilon_array = ['10.0', '25.0', '50.0', '75.0', '100.0']
    # plt.ylabel('test accuracy')
    # plt.xlabel('global round')
    # for epsilon in epsilon_array:
    #     y = openfile('./log/accfile_fed_mnist_cnn_100_iidFalse_dp_Laplace_epsilon_{}.dat'.format(epsilon))
    #     plt.plot(range(100), y, label=r'$\epsilon={}$'.format(epsilon))
    # y = openfile('./log/accfile_fed_mnist_cnn_100_iidFalse_dp_no_dp_epsilon_20.dat'.format(epsilon))
    # plt.plot(range(100), y, label=r'$\epsilon=+\infty$')
    # plt.title('Mnist Laplace')
    # plt.legend()
    # plt.savefig('mnist_gaussian_laplace.png')

