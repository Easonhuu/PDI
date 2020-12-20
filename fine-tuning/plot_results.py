import numpy as np 
import pandas as pd
import os 
import csv
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import seaborn as sns
from sklearn.linear_model import LinearRegression
import scipy

def plot_fold_line(plot_RMSE, plot_R, dir_save):
    # plot_RMSE : list, [[epoch, train_rmse, eval_rmse], [epoch, train_rmse, eval_rmse], ...]
    # plot_R : list, [[epoch, train_r, eval_r], [epoch, train_r, eval_r], ...]
    plot_RMSE_arr = np.array(plot_RMSE).T
    plot_R_arr = np.array(plot_R).T

    with open(os.path.join(dir_save, "Dataset-RMSE-R.csv"), 'w', newline='', encoding='utf-8') as pickle_file:
        fwriter = csv.writer(pickle_file)
        fwriter.writerow(["epoch", "train_RMSE", "valid_RMSE", "test_RMSE(core2016)", "test_RMSE(casf2013)", "test_RMSE(astex)",
                                   "train_R", "valid_R", "test_R(core2016)", "test_R(casf2013)", "test_R(astex)"])
        for i in range(len(plot_RMSE)):
            fwriter.writerow(plot_RMSE[i] + plot_R[i][1:])

    plt.figure('1')
    plt.plot(plot_RMSE_arr[0], plot_RMSE_arr[1], label="train")
    plt.plot(plot_RMSE_arr[0], plot_RMSE_arr[2], label="valid")
    plt.plot(plot_RMSE_arr[0], plot_RMSE_arr[3], label="core2016")
    plt.plot(plot_RMSE_arr[0], plot_RMSE_arr[4], label="CASF2013")
    plt.plot(plot_RMSE_arr[0], plot_RMSE_arr[5], label="astex")
    plt.legend()
    plt.title('Dataset RMSE')
    plt.xlabel('epoch')
    plt.ylabel('RMSE')
    plt.savefig(os.path.join(dir_save, 'Dataset-RMSE.jpg'))

    plt.figure('2')
    plt.plot(plot_R_arr[0], plot_R_arr[1], label="train")
    plt.plot(plot_R_arr[0], plot_R_arr[2], label="valid")
    plt.plot(plot_R_arr[0], plot_R_arr[3], label="core2016")
    plt.plot(plot_R_arr[0], plot_R_arr[4], label="CASF2013")
    plt.plot(plot_R_arr[0], plot_R_arr[5], label="astex")
    plt.legend()
    plt.title('Dataset R with finetuning of bert')
    plt.xlabel('epoch')
    plt.ylabel('R')
    plt.savefig(os.path.join(dir_save, 'Dataset-R.jpg'))

def plot_scatter(label_pred, label_true, dir_save, train_eval_test):
    sns.set(context='paper', style='white')
    sns.set_color_codes() 
    set_colors = {'train': 'b', 'valid': 'green', 'core2016': 'purple', 'casf2013': 'darkorange', 'astex': 'r'}

    rmse = ((label_pred - label_true) ** 2).mean() ** 0.5
    mae = (np.abs(label_pred - label_true)).mean()
    corr = scipy.stats.pearsonr(label_pred, label_true)
    lr = LinearRegression()
    lr.fit(np.expand_dims(label_pred, axis=1), label_true)
    y_ = lr.predict(np.expand_dims(label_pred, axis=1))
    sd = (((label_true - y_) ** 2).sum() / (len(label_true) - 1)) ** 0.5
    print("%s set: RMSE=%.3f, MAE=%.3f, R=%.2f (p=%.2e), SD=%.3f" %
        (train_eval_test, rmse, mae, *corr, sd))
    
    table = pd.DataFrame({'real': label_true, 'pred': label_pred})
    grid = sns.jointplot('real', 'pred', data=table, stat_func=None, color=set_colors[train_eval_test],
                        space=0, height=4, ratio=4, s=20, edgecolor='w', ylim=(0, 16), xlim=(0, 16))  # (0.16)
    grid.ax_joint.set_xticks(range(0, 16, 5))
    grid.ax_joint.set_yticks(range(0, 16, 5)) 
    grid.ax_joint.text(1, 14, train_eval_test + ' set', fontsize=16) #调整标题大小
    parm_font_size = 8
    grid.ax_joint.text(16, 19.5, 'RMSE: %.3f' % (rmse), fontsize=parm_font_size)
    grid.ax_joint.text(16, 18.5, 'MAE: %.3f' % (mae), fontsize=parm_font_size)
    grid.ax_joint.text(16, 17.5, 'R: %.2f ' % corr[0], fontsize=parm_font_size)
    grid.ax_joint.text(16, 16.5, 'SD: %.3f ' % sd, fontsize=parm_font_size)
    grid.ax_joint.text(16.5, -1.25, '$\it{(pK_a)}$')
    # grid.ax_joint.text(-2, 17, '(pKa)')
    grid.fig.savefig(dir_save+'/positive_pred_%s.jpg' % train_eval_test,dpi=400)
