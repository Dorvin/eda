import csv
import matplotlib.pyplot as plt 
from scipy import interpolate

def get_fig(data_num, epoch):
    # make data structure to store results
    #data_num = input('Dataset: ')
    #epoch = int(input('Epoch: '))
    x = [i+1 for i in range(0, epoch)]
    avg_train_loss = {
        'd0': [0 for i in range(0, epoch)],
        'd1': [0 for i in range(0, epoch)],
        'd5': [0 for i in range(0, epoch)],
    }

    avg_val_loss = {
        'd0': [0 for i in range(0, epoch)],
        'd1': [0 for i in range(0, epoch)],
        'd5': [0 for i in range(0, epoch)],
    }

    avg_train_acc = {
        'd0': [0 for i in range(0, epoch)],
        'd1': [0 for i in range(0, epoch)],
        'd5': [0 for i in range(0, epoch)],
    }

    avg_val_acc = {
        'd0': [0 for i in range(0, epoch)],
        'd1': [0 for i in range(0, epoch)],
        'd5': [0 for i in range(0, epoch)],
    }

    avg_diff_loss = {
        'd0': [0 for i in range(0, epoch)],
        'd1': [0 for i in range(0, epoch)],
        'd5': [0 for i in range(0, epoch)],
    }

    avg_diff_acc = {
        'd0': [0 for i in range(0, epoch)],
        'd1': [0 for i in range(0, epoch)],
        'd5': [0 for i in range(0, epoch)],
    }

    for method in (0, 1, 5):
        # train_loss
        logs = []
        sparse_data = [{'x': [], 'y': []} for i in range(0, 10)]
        f = open(f'./result/{data_num}/train/d{method}_loss.csv', 'r', encoding='utf-8')
        cr = csv.reader(f)
        # skip the first row
        next(cr)
        pre = epoch + 5000
        curr = -1
        for row in cr:
            step = int(row[1])
            value = float(row[2])
            if pre > step:
                curr += 1
                sparse_data[curr]['x'].append(1)
                sparse_data[curr]['y'].append(value)
            sparse_data[curr]['x'].append(step)
            sparse_data[curr]['y'].append(value)
            pre = step
        for i in range(0, 10):
            sparse_data[i]['x'].append(epoch)
            sparse_data[i]['y'].append(sparse_data[i]['y'][len(sparse_data[i]['y'])-1])
        for i in range(0, 10):
            fn = interpolate.interp1d(sparse_data[i]['x'],sparse_data[i]['y'])
            logs.append(fn(x))
        for i in range(0, 10):
            for t in range(0,epoch):
                avg_train_loss[f'd{method}'][t] += logs[i][t]/10
        f.close()

        # val_loss
        logs = []
        sparse_data = [{'x': [], 'y': []} for i in range(0, 10)]
        f = open(f'./result/{data_num}/val/d{method}_loss.csv', 'r', encoding='utf-8')
        cr = csv.reader(f)
        # skip the first row
        next(cr)
        pre = epoch + 1000
        curr = -1
        for row in cr:
            step = int(row[1])
            value = float(row[2])
            if pre > step:
                curr += 1
                sparse_data[curr]['x'].append(1)
                sparse_data[curr]['y'].append(value)
            sparse_data[curr]['x'].append(step)
            sparse_data[curr]['y'].append(value)
            pre = step
        for i in range(0, 10):
            sparse_data[i]['x'].append(epoch)
            sparse_data[i]['y'].append(sparse_data[i]['y'][len(sparse_data[i]['y'])-1])
        for i in range(0, 10):
            fn = interpolate.interp1d(sparse_data[i]['x'],sparse_data[i]['y'])
            logs.append(fn(x))
        for i in range(0, 10):
            for t in range(0,epoch):
                avg_val_loss[f'd{method}'][t] += logs[i][t]/10
        f.close()

        # train_acc
        logs = []
        sparse_data = [{'x': [], 'y': []} for i in range(0, 10)]
        f = open(f'./result/{data_num}/train/d{method}_accuracy.csv', 'r', encoding='utf-8')
        cr = csv.reader(f)
        # skip the first row
        next(cr)
        pre = epoch + 1000
        curr = -1
        for row in cr:
            step = int(row[1])
            value = float(row[2])
            if pre > step:
                curr += 1
                sparse_data[curr]['x'].append(1)
                sparse_data[curr]['y'].append(value)
            sparse_data[curr]['x'].append(step)
            sparse_data[curr]['y'].append(value)
            pre = step
        for i in range(0, 10):
            sparse_data[i]['x'].append(epoch)
            sparse_data[i]['y'].append(sparse_data[i]['y'][len(sparse_data[i]['y'])-1])
        for i in range(0, 10):
            fn = interpolate.interp1d(sparse_data[i]['x'],sparse_data[i]['y'])
            logs.append(fn(x))
        for i in range(0, 10):
            for t in range(0,epoch):
                avg_train_acc[f'd{method}'][t] += logs[i][t]/10
        f.close()

        # val_acc
        logs = []
        sparse_data = [{'x': [], 'y': []} for i in range(0, 10)]
        f = open(f'./result/{data_num}/val/d{method}_accuracy.csv', 'r', encoding='utf-8')
        cr = csv.reader(f)
        # skip the first row
        next(cr)
        pre = epoch + 1000
        curr = -1
        for row in cr:
            step = int(row[1])
            value = float(row[2])
            if pre > step:
                curr += 1
                sparse_data[curr]['x'].append(1)
                sparse_data[curr]['y'].append(value)
            sparse_data[curr]['x'].append(step)
            sparse_data[curr]['y'].append(value)
            pre = step
        for i in range(0, 10):
            sparse_data[i]['x'].append(epoch)
            sparse_data[i]['y'].append(sparse_data[i]['y'][len(sparse_data[i]['y'])-1])
        for i in range(0, 10):
            fn = interpolate.interp1d(sparse_data[i]['x'],sparse_data[i]['y'])
            logs.append(fn(x))
        for i in range(0, 10):
            for t in range(0,epoch):
                avg_val_acc[f'd{method}'][t] += logs[i][t]/10
        f.close()

    for method in (0, 1, 5):
        for t in range(0,epoch):
            avg_diff_loss[f'd{method}'][t] = avg_train_loss[f'd{method}'][t] - avg_val_loss[f'd{method}'][t]
            avg_diff_acc[f'd{method}'][t] = avg_train_acc[f'd{method}'][t] - avg_val_acc[f'd{method}'][t]

    # train_loss(d015)     val_loss(d015)
    # train_acc(d015)      val_acc(d015)

    f, axes = plt.subplots(2, 2, sharex=False, sharey=False)
    #f.set_size_inches((16, 16))
    ####################################################################################################
    axes[0][0].plot(x, avg_train_loss['d0'], x, avg_train_loss['d1'], x, avg_train_loss['d5'])
    axes[0][0].legend(['d0','d1','d5'], loc='best')
    axes[0][0].set_title("train_loss(d015)")

    axes[0][1].plot(x, avg_val_loss['d0'], x, avg_val_loss['d1'], x, avg_val_loss['d5'])
    axes[0][1].legend(['d0','d1','d5'], loc='best')
    axes[0][1].set_title("val_loss(d015)")

    ####################################################################################################

    axes[1][0].plot(x, avg_train_acc['d0'], x, avg_train_acc['d1'], x, avg_train_acc['d5'])
    axes[1][0].legend(['d0','d1','d5'], loc='best')
    axes[1][0].set_title("train_acc(d015)")

    axes[1][1].plot(x, avg_val_acc['d0'], x, avg_val_acc['d1'], x, avg_val_acc['d5'])
    axes[1][1].legend(['d0','d1','d5'], loc='best')
    axes[1][1].set_title("val_acc(d015)")

    f.tight_layout()
    plt.savefig(f'./result/{data_num}_analysis.svg')
    #plt.show()
