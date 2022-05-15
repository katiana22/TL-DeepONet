'''
Authors: Katiana Kontolati, PhD Candidate, Johns Hopkins University
         Somdatta Goswami, Postdoctoral Researcher, Brown University
Tensorflow Version Required: TF1.15     
'''
import numpy as np
from matplotlib import pylab as plt

inputs_test = np.loadtxt('./Output/target/f_test')
pred = np.loadtxt('./Output/target/u_pred')
true = np.loadtxt('./Output/target/u_ref')
num_test = inputs_test.shape[0]

# Load source test loss
test_loss_target = np.loadtxt('./Output/target/loss_test')
epochs_target = np.loadtxt('./Output/target/epochs')
test_loss_source = np.loadtxt('./Output/source/loss_test')
epochs_source = np.loadtxt('./Output/source/epochs')

plt.rcParams.update({'font.size': 17})
fig = plt.figure(constrained_layout=True, figsize=(7, 5))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0])
ax.plot(epochs_source, test_loss_source, color='b', label='Testing Loss')
ax.set_yscale('log')
ax.set_ylabel('Loss')
ax.set_xlabel('Epochs')
ax.legend(loc='upper left')
fig.savefig('./Output/source/loss_test_log.png')

## Plotting both source and target loss
fig = plt.figure(constrained_layout=True, figsize=(7, 5))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0])
ax.plot(epochs_source, test_loss_source, color='b', label='source')
ax.plot(epochs_target[100:], test_loss_target[100:], color='r', label='target')
ax.set_ylabel('Loss')
ax.set_xlabel('Epochs')
ax.legend(loc='upper left')
fig.savefig('./Output/target/loss_test_comparison.png')

'''
plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(7, 6))
epochs = train_loss.shape[0]
x = np.linspace(1, epochs, epochs)
plt.plot(x, train_loss, label='train loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('Output/train_loss.png', dpi=250)

fig = plt.figure(figsize=(7, 6))
plt.plot(x, test_loss, label='test loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('Output/test_loss.png', dpi=250)

nx, ny = 28, 28
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xx, yy = np.meshgrid(x, y)

index = 0
tot = int(nx*ny)

snapshots = 5
time = np.linspace(0,1,snapshots)
print(time)
get = np.linspace(0, pred.shape[1], snapshots+1)
get = [int(x) for x in get]

plt.rcParams.update({'font.size': 12.5})
th = 0
fig, axs = plt.subplots(3, snapshots, figsize=(14.5,6), constrained_layout=True)
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1
for col in range(snapshots):
    for row in range(3):
        ax = axs[row, col]
        if row == 0:
            ss1 = true[index, get[col]:get[col]+tot]
            pcm = ax.contourf(xx, yy, ss1.reshape(nx,ny), levels=np.linspace(np.min(ss1)-th, np.max(ss1)+th, 200), cmap='jet')
            cbar = plt.colorbar(pcm, ax=ax, format='%.1f', ticks=np.linspace(np.min(ss1)-th, np.max(ss1)+th , 5))
            ax.set_title(r'$t={}$'.format(time[col]))
        if row == 1:
            ss2 = pred[index, get[col]:get[col]+tot]
            pcm = ax.contourf(xx, yy, ss2.reshape(nx,ny), levels=np.linspace(np.min(ss1)-th, np.max(ss1)+th, 200), cmap='jet')
            cbar = plt.colorbar(pcm, ax=ax, format='%.1f', ticks=np.linspace(np.min(ss1)-th, np.max(ss1)+th , 5))
        if row == 2:
            errors = np.abs((pred - true)/true)
            ss = errors[index, get[col]:get[col]+tot]
            pcm = ax.contourf(xx, yy, ss.reshape(nx,ny), levels=200, cmap='jet')
            plt.colorbar(pcm, ax=ax, format='%.0e', ticks=np.linspace(np.min(ss), np.max(ss) , 5))

        if row == 2:
            ax.set_xlabel(r'$x$', fontsize=13)
        if col ==0 and row ==0:
            ax.set_ylabel('Reference \n y', fontsize=13)
        if col == 0 and row==1:
            ax.set_ylabel('DeepONet \n y', fontsize=13)
        if col == 0 and row==2:
            ax.set_ylabel('Error \n y', fontsize=13)
        ax.locator_params(axis='y', nbins=3)
        ax.locator_params(axis='x', nbins=3)
plt.savefig('Output/target/time_evolution_comparison.png', bbox_inches='tight', dpi=500)
'''
