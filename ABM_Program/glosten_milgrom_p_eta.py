import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import cm
from matplotlib.colors import ListedColormap
import seaborn as sns
from glosten_milgrom_base import Glosten_Milgrom
import time
import functools
import argparse

class GMchild(Glosten_Milgrom):

    def __init__(self, V_L=100, V_H=200, mu_i=0.1, mu_p=0.2, eta=0.8, sigma=0.5, gamma=0.5, q_0=0.5, t_max=200):
        super().__init__()
        self.V_L = V_L
        self.V_H = V_H
        self.mu_i = mu_i      # proportion of informed traders
        self.mu_p = mu_p      # proportion of partial informed traders
        self.mu = mu_i + mu_p
        self.eta = eta
        self.sigma = sigma    # prob. of V_L
        self.gamma = gamma    # prob. of buying for uninformed traders
        self.q_0 = q_0
        self.t_max = t_max

    def set_env(self):
        super().set_env()
        """
        Put here whatever you wish to overwrite in the set_env() function.
        """
        self.traders = np.random.choice(["I", "P", "U"], size=self.t_max, p=[self.mu_i, self.mu_p, 1-self.mu_i-self.mu_p])
        self.p_b = np.array([self.mu_p*(1-self.eta) + (1-self.mu_i-self.mu_p)*(self.gamma), self.mu_i + self.mu_p*self.eta + (1-self.mu_i-self.mu_p)*(self.gamma)])
        self.p_s = 1 - self.p_b

    def gm_model(self):
        # set up the environment for the model
        self.set_env()

        ask, bid = None, None   # starting without any previous quote
        for t in range(self.t_max):
            # market maker's belief of V being V_L or V_H until now
            belief = np.array([self.q[t], 1-self.q[t]])

            #
            # what is the order coming in (buy or sell)
            #
            order = None
            if self.traders[t] == "U":
                order = np.random.choice(["B", "S"], p=[self.gamma, 1-self.gamma])
            elif self.traders[t] in["I", "P"]:
                if self.V == self.V_L:
                    order = "S"
                elif self.V == self.V_H:
                    order = "B"
                # partial informed traders make mistake.
                if self.traders[t] == "P" and not np.random.choice([True, False], p=[self.eta, 1-self.eta]):
                    if order == "B":
                        order = "S"
                    elif order == "S":
                        order = "B"

            #
            # market maker's doing upon receiving an order
            #
            if order == "B":
                # update market maker's belief of V is V_L
                support = self.p_b/np.sum(self.p_b*belief)
                post_belief = support * belief

                # provide the ask (expectation of V according to belief)
                ask = np.sum(np.array([self.V_L, self.V_H]) * post_belief)

                # logging
                self.prices.append(ask)
                self.profits.append(ask - self.V)
                profit = self.V - ask

            elif order == "S":
                # update market maker's belief of V is V_L
                support = self.p_s/np.sum(self.p_s*belief)
                post_belief = support * belief

                # provide the bid (expectation of V according to belief)
                bid = np.sum(np.array([self.V_L, self.V_H]) * post_belief)

                # logging
                self.prices.append(bid)
                self.profits.append(self.V - bid)
                profit = bid - self.V
            self.save_analytics(trader_profit=(self.traders[t], profit))

            # more logging of the process
            if bid: self.bids.append(bid)
            if ask: self.asks.append(ask)
            self.q.append(post_belief[0])

            # test of convergence, terminate if converges
            if t > self.s and self.check_conv():
                # prompt the convergence status
                self.save_analytics(converged_time=t)
                # print(f'With {t} orders received, the market maker is now {round(self.q[-1]*100, 4)}% confident that V is low.')
                break

    def plot_profit(self, star=False):
        super().plot_profit(star=star)
        plt.title(f"Welfare($\sigma={self.sigma}, \mu_{{i}}={self.mu_i}, \mu_{{p}}={self.mu_p}, \gamma={self.gamma}, \eta={self.eta}$)")


class analytics:

    def __init__(self, path="./", variant=3):
        self.path = path
        self.variant = variant

    def timer(func):
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            start_time = time.perf_counter()    # 1
            value = func(*args, **kwargs)
            end_time = time.perf_counter()      # 2
            run_time = end_time - start_time    # 3
            # print(f"-- execute {func.__name__!r} in {run_time:.2f} secs")
            return value
        return wrapper_timer

    @staticmethod
    @timer
    def plot_heatmap(x_list, y_list, fixed_param, fixed_v, xlabel=None, ylabel=None, title=None, ax=None):
        X, Y = np.meshgrid(x_list, y_list)
        time_record = []
        for x, y in zip(X.ravel(), Y.ravel()):
            if fixed_param == 'mu_i':
                if fixed_v + y >= 1:
                    time_record.append(0)
                    continue
                model = GMchild(mu_i=fixed_v, mu_p=y, eta=x)
            elif fixed_param == 'mu_p':
                if fixed_v + y >= 1:
                    time_record.append(0)
                    continue
                model = GMchild(mu_i=y, mu_p=fixed_v, eta=x)
            elif fixed_param == 'eta':
                if x+ y >= 1:
                    time_record.append(0)
                    continue
                model = GMchild(mu_i=x, mu_p=y, eta=fixed_v)

            model.gm_model()
            time_record.append(model.analytics['converged_time'])

        Z = np.array(time_record).reshape(Y.shape)

        # design ploting
        newcolors = cm.get_cmap('Blues', 256)(np.linspace(0, 1, 256))
        newcolors[-1:, :] = np.array([256/256, 0/256, 0/256, 0.5])
        newcmp = ListedColormap(newcolors)
        plt.tight_layout()
        cbar_kws = {"orientation": "horizontal", "pad":0.2, "label": "orders", "shrink": 0.7}
        plot = sns.heatmap(Z, cmap=newcmp, ax=ax, square=True, cbar=True, cbar_kws=cbar_kws)
        xtick = [round(i, 1) for i in x_list]
        ytick = [round(i, 1) for i in y_list]
        plot.set_xticks(range(0, len(xtick), 10))
        plot.set_xticklabels(xtick[::10], rotation=90)
        plot.set_yticks(range(0, len(ytick), 10))
        plot.set_yticklabels(ytick[::10])
        plot.set_xlabel(xlabel)
        plot.set_ylabel(ylabel)
        plot.set_title(f'Fixed value {title}')

    @staticmethod
    def plot_hist(etas=[0.5, 0.7, 0.9], mu_i=0.1, mu_p=0.3, iter=3000, bins=50, std_threshold=2):
        # sampling
        def sampling(iter=1000, eta=0.5):
            distribute_ = {agent : list() for agent in agents}
            for i in range(iter):
                model = GMchild(mu_i=mu_i, mu_p=mu_p, eta=eta)
                model.gm_model()
                for agent in agents:
                    # calculate cumulative sum
                    distribute_[agent].append(np.cumsum(model.analytics['profit'][agent[0]])[-1])

            # remove outliers
            dis = np.array(list(distribute_.values())).ravel()
            std = [np.mean(dis) - std_threshold * np.std(dis), np.mean(dis) + std_threshold * np.std(dis)]
            for key, value in distribute_.items():
                distribute_[key] = [v for v in value if v >= std[0] and v <= std[1]]
            return distribute_

        # plot subplots
        agents = ['Informed', 'Partially informed', 'Uninformed', 'Market maker']
        fig, axes = plt.subplots(ncols=1, nrows=len(agents), figsize=(15, 15), sharex=True, sharey=True, constrained_layout=True)
        fig.suptitle(f'Distribution of welfare regarding $\eta$ at $\mu_i={mu_i}, \mu_p={mu_p}$', fontsize=20)
        fig.supxlabel('Profit', size=20)
        fig.supylabel('Trader', size=20)
        for i, eta in enumerate(etas):
            distribute = sampling(iter=iter, eta=eta)
            for ax, agent in zip(axes, agents):
                ax.hist(distribute[agent], bins=bins, histtype='step', label=f'$\eta$={eta} (mean={round(np.mean(distribute[agent]), 2)})', alpha=0.45, linewidth=3)
                ax.xaxis.set_tick_params(labelbottom=True)
                ax.set_ylabel(agent, rotation=90, size='large')
                if i == 0:
                    ax.axvline(x=0, color='red', linestyle='dashed', linewidth=1.5, alpha=0.4, label='zero profit')
                ax.legend(loc='upper right', prop={'size': 12})

    # save image to default folder
    def saveimg(self, filename, plt):
        filepath = f'{self.path}/v{self.variant}_{filename}'
        plt.savefig(filepath)
        plt.clf()
        print(f'- {filename} is saved!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to save images", type=str, default=".")
    parser.add_argument("-a", "--all", help="path to save images", action="store_true")
    parser.add_argument("--heatmap", help="execute heatmap", action="store_true")
    parser.add_argument("--distribution", help="execute distribution", nargs="*", type=float)
    parser.add_argument("--profit", help="execute profit in a simulation", action="store_true")
    args = parser.parse_args()

    analytic = analytics(path=args.path, variant=3)

    ## single welfare
    if args.all or args.profit:
        model = GMchild(mu_i=0.1, mu_p=0.3)
        model.gm_model()
        model.plot_profit(star=True)
        analytic.saveimg('fig5_profit.png', plt)

    ## the distribution of welfare
    if args.all or args.distribution:
        if args.distribution:
            etas = args.distribution
        else:
            etas = [0.6, 0.77, 0.95]
        analytics.plot_hist(etas=etas, iter=3000)
        analytic.saveimg('fig6_distribution.png', plt)

    ## the distribution of welfare when eta is small
    if args.all:
        etas = [0.01]
        analytics.plot_hist(etas=etas, iter=3000)
        analytic.saveimg('fig7_distribution_small_eta.png', plt)


    ## Learning efficiency about \mu_p when \eta > 0.5
    if args.all:
        mu_i = 0.0
        mu_p_lst = np.arange(0.01, 1.0, 0.01)
        eta_lst = [0.5, 0.65, 0.8, 0.95]

        fig, axes = plt.subplots(nrows=1, ncols=len(eta_lst), figsize=(20, 5), sharey=True, constrained_layout=True)
        fig.suptitle('Learning efficiency regarding $\mu_p$ with varying $\eta$', fontsize=20)
        for ax, eta in zip(axes, eta_lst):
            orders = []
            for mu_p in mu_p_lst:
                model = GMchild(mu_i=mu_i, mu_p=mu_p, eta=eta)
                model.gm_model()
                orders.append(model.analytics['converged_time'])
            ax.plot(mu_p_lst, orders)
            ax.set_xlabel('$\mu_{p}$')
            ax.set_ylabel('number of orders')
            ax.set_title(f'$\eta = {eta}$')
        analytic.saveimg('fig1_large_eta.png', plt)

    ## Learning efficiency regarding \mu_p when \eta is < 0.5
    if args.all:
        mu_i = 0.0
        mu_p_lst = np.arange(0.01, 1.0, 0.01)
        eta = 0.01

        orders = []
        for mu_p in mu_p_lst:
            model = GMchild(mu_i=mu_i, mu_p=mu_p, eta=eta)
            model.gm_model()
            orders.append(model.analytics['converged_time'])
        plt.figure(figsize=(5, 5))
        plt.plot(mu_p_lst, orders)
        plt.xlabel('$\mu_{p}$')
        plt.ylabel('number of orders')
        plt.title(f'Learning efficiency regarding $\mu_p$ when $\eta = {eta}$')
        analytic.saveimg('fig2_small_eta.png', plt)


    ## Discover learning efficiency with different eta
    if args.all or args.heatmap:
        mu_i_lst = [0.1, 0.25, 0.4]
        mu_p_lst = np.arange(0.01, 1.0, 0.01)
        eta_lst = np.arange(0.01, 1.0, 0.01)

        fig, axes = plt.subplots(ncols=len(mu_i_lst), nrows=1, figsize=(18, 6))
        fig.suptitle('Learning efficiency between $\mu_p$, $\eta$ with a set of $\mu_i$', fontsize=20)
        for ax, mu_i in zip(axes, mu_i_lst):
            analytics.plot_heatmap(x_list=eta_lst, y_list=mu_p_lst, fixed_param='mu_i', fixed_v=mu_i, xlabel='$\eta$', ylabel='$\mu_{p}$', title=f'$\mu_{{i}}={mu_i}$', ax=ax)

        analytic.saveimg('fig3_heatmap_fixed_mu_i.png', plt)

    ## Discover learning efficiency with close ratio of \mu_i and \mu_p
    if args.all or args.heatmap:
        mu_i_lst = np.arange(0.01, 1.0, 0.01)
        mu_p_lst = np.arange(0.01, 1.0, 0.01)
        eta_lst = [0.1, 0.3, 0.5, 0.75, 0.98]

        fig, axes = plt.subplots(ncols=len(eta_lst), nrows=1, figsize=(18, 6))
        fig.suptitle('Learning efficiency between $\mu_i$, $\mu_p$ with a set of $\eta$', fontsize=20)
        for ax, eta in zip(axes, eta_lst):
            analytics.plot_heatmap(x_list=mu_i_lst, y_list=mu_p_lst, fixed_param='eta', fixed_v=eta, xlabel='$\mu_{i}$', ylabel='$\mu_{p}$', title=f'$\eta={eta}$', ax=ax)

        analytic.saveimg('fig4_heatmap_fixed_eta.png', plt)