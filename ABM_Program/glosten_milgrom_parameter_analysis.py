import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
from glosten_milgrom_base import Glosten_Milgrom
import argparse


# Create child class for parameter analysis of base agent base model
class ParameterAnalysis(Glosten_Milgrom):
    def __init__(self, path='.', filename=None):
        # Inherit all attributes from parent class
        super().__init__(self)
        self.path = path
        self.filename = filename
        self.variant = 1

     # ----
    # Analysing information parameter sigma
    # ----
    
    # Analyse market makers initial belief of sigma in a bullis/bearish market environment
    def mm_market_trend(self,view,conditions,iterations):
        eff = dict()
        # For each environment compute learning efficiency
        for behav in conditions.keys():
            num_orders = list()
            for i in range(iterations):
                initial_belief, market_sigma = float(conditions[behav][0]),float(conditions[behav][1])
                model = Glosten_Milgrom(q_0=initial_belief,sigma=market_sigma)
                model.gm_model()
                num_order = len(model.prices)
                num_orders.append(num_order)
            eff[behav] = np.mean(num_orders)

        # Create datframe for efficiencies and belief values
        df1 = pd.DataFrame(conditions.values())
        df2 = pd.DataFrame(eff.items())
        
        # Concatenate dataframes
        df = pd.concat([df1,df2],axis=1)
        df.columns = ['Intial belief','Sigma','Market type','Efficiency']
        df.index = df['Market type']
        df.pop('Market type')
        # Print dataframe 
        #print("\nMarket maker belief is that market is %s\n"%view)
        #print(df)
        plt.figure(figsize=(10, 3))
        ax = plt.subplot(111, frame_on=False) # no visible frame
        ax.axis('off')
        ax.set_title(f'Market maker belief is that market is {view}', size=18)
        dftable = table(ax, df, loc='center')
        dftable.scale(1, 2)
        dftable.set_fontsize(16)

        
    # ----
    # Analysing information parameter mu
    # ----

    # analyse affect of varying mu
    def mu_analysis(self,mu_list):
        mu_results = list()
        # Iterate through each mu in list
        for i in range(len(mu_list)):
            model = Glosten_Milgrom(mu=mu_list[i],t_max=500)
            model.gm_model()
            result = [model.bids,model.asks,model.prices]
            mu_results.append(result)

        # Plot results
        fig, axs = plt.subplots(nrows=1, ncols=len(mu_list), figsize=(25,5))
        axs = axs.ravel()
        for i in range(len(mu_results)):
            for j in range(len(mu_results[i])):
                axs[i].plot(mu_results[i][j])
                axs[i].set_title("Mu = %f"%mu_list[i])
                axs[i].set_xlabel("Orders received")
                axs[i].set_ylabel("Prices")
        fig.text(0.5, 1, "Bids, asks and transaction prices with varying mu", ha="center", va="top", fontsize=15)
        #plt.show()
        self.saveimg(plt)

    # analyse market maker learning efficiency of varying mu
    def mu_learning(self,mu_list,t_max=200):
        # List to store when market maker learns and does not
        mu_convergence = list()
        mu_non_convergence = list()

        # Iterate through each mu in list
        for i in range(len(mu_list)):
            model = Glosten_Milgrom(mu=mu_list[i],t_max=t_max)
            model.gm_model()
            # get number of orders required to learn / not learn
            result = len(model.prices)
            if (result == t_max):
                mu_non_convergence.append(result)
            else:
                mu_convergence.append(result)

        # Plot efficiency of market makers learning under different mu
        idx = len(mu_non_convergence)
        plt.plot(mu_list[0:idx],mu_non_convergence,'r--',label='MM does not learn')
        plt.plot(mu_list[idx-1:idx+1],[mu_non_convergence[-1],mu_convergence[0]],'b')
        plt.plot(mu_list[idx:],mu_convergence,'b',label = 'MM learns')
        plt.title('Learning efficiency under different levels of informed traders')
        plt.xlabel('Value of $\mu$')
        plt.ylabel('Orders received by market maker ')
        plt.legend()
        #plt.show()
        self.saveimg(plt)

    # Observe market maker abs expected profits 
    def mu_profits(self,mu_list):
        mm_profits = list()
        # Iterate through each mu in list
        for i in range(len(mu_list)):
            model = Glosten_Milgrom(mu=mu_list[i])
            model.gm_model()
            exp_pnl = [x/(i+1) for i, x in enumerate(np.cumsum(model.profits))]
            mm_profits.append(exp_pnl)

        # Plot results
        fig, axs = plt.subplots(nrows=1, ncols=len(mu_list), figsize=(25,5))
        axs = axs.ravel()
        for i in range(len(mm_profits)):
            axs[i].plot(np.abs(mm_profits[i]))
            axs[i].set_xlabel("Orders received")
            axs[i].set_ylabel("Abs Expected profits")
            axs[i].set_title("Mu = %f"%mu_list[i])
        fig.text(0.5, 1, "Market maker absolute expected profits with varying mu", ha="center", va="top", fontsize=15)
        #plt.show()
        self.saveimg(plt)

    # Analyse welfare of agents
    def mu_welfare(self,mus):
        def sampling(iter=1000, mu=0.5):
            distribute_ = {
                'Informed': [],
                'Uninformed': [],
                'Market maker': []
            }
            for i in range(iter):
                your_model = Glosten_Milgrom(mu=mu)
                your_model.gm_model()
                for agent in agents:
                    distribute_[agent].append(np.cumsum(your_model.analytics['profit'][agent[0]])[-1])

            # remove outliers
            threshold = 2
            dis = np.array(list(distribute_.values())).ravel()
            std3 = [np.mean(dis) - threshold * np.std(dis), np.mean(dis) + threshold * np.std(dis)]
            for key, value in distribute_.items():
                distribute_[key] = [v for v in value if v >= std3[0] and v <= std3[1]]
            return distribute_

        agents = ['Informed', 'Uninformed', 'Market maker']
        fig, axs = plt.subplots(ncols=1, nrows=len(agents), figsize=(15, 12), sharex=True, sharey=True)

        for i in range(len(mus)):
            distribute = sampling(iter=1000, mu=mus[i])
            axs[i].hist(distribute[agents[0]], bins=50, histtype='step', label=f'agent= Informed (mean={round(np.mean(distribute[agents[0]]), 2)})', alpha=0.45, linewidth=2)
            axs[i].hist(distribute[agents[1]], bins=50, histtype='step', label=f'agent= Uninformed (mean={round(np.mean(distribute[agents[1]]), 2)})', alpha=0.45, linewidth=2)
            axs[i].hist(distribute[agents[2]], bins=50, histtype='step', label=f'agent= Market marker (mean={round(np.mean(distribute[agents[2]]), 2)})', alpha=0.45, linewidth=2)
            axs[i].axvline(x=0, color='red', linestyle='dashed', linewidth=1.5, alpha=0.4, label='zero profit')
            axs[i].xaxis.set_tick_params(labelbottom=True)
            axs[i].set_title(f'$\mu$ = {mus[i]}', size='large')
            axs[i].legend(loc='upper right', prop={'size': 12})

        plt.tight_layout()
        plt.subplots_adjust(left=0.08, bottom=0.08, wspace=0.1)
        fig.text(0.2, 1, "Agents welfare distribution with mu varying", ha="center", va="top", fontsize=15)
        fig.text(0.53, 0.01, "Profits", ha="center", va="bottom", fontsize=15)
        #plt.show()
        self.saveimg(plt)

    def saveimg(self, plt):
        filepath = f'{self.path}/v{self.variant}_{self.filename}'
        plt.savefig(filepath)
        plt.clf()
        print(f'- {self.filename} is saved!')

# Controller
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to save images", type=str, default=".")
    parser.add_argument("-a", "--all", help="path to save images", action="store_true")
    parser.add_argument("--trend", help="sigma market trend", action="store_true")
    parser.add_argument("--analysis", help="mu analysis", action="store_true")
    parser.add_argument("--learning", help="mu learning", action="store_true")
    parser.add_argument("--profit", help="mu profit", action="store_true")
    parser.add_argument("--welfare", help="mu welfare", action="store_true")
    args = parser.parse_args()
    
    # # Select analysis action
    # action = "mu welfare"
    
    # Sigma analytics 
    # Analyse the initial assumptions under different market conditions
    # Output: dataframe
    if args.all or args.trend:
        # Market maker believes market is bullish
        market_maker_view  = "Bullish"
        if market_maker_view == "Bullish":
            initial_belief = 0.4
        # Market maker believes market is bearish
        else:
            initial_belief = 0.6

        conditions = {"Normal":[0.5,0.5],"Bullish":[initial_belief, 0.1],"Bearish":[initial_belief, 0.9]} # Different environments and market maker assumptions
        iterations = 100 # Number of iterations
        my_model = ParameterAnalysis(path=args.path, filename='trend_mu')
        my_model.mm_market_trend(market_maker_view,conditions,iterations)
        my_model.saveimg(plt)

    # mu analytics 
    # Analyse efficiency under varied mu multiple plots
    # Output: graph
    if args.all or args.analysis:
        # Analyse dynamics by altering parameter mu
        mu_list = np.arange(0.1,1.0,0.1)
        my_model = ParameterAnalysis(path=args.path, filename='analysis_mu')
        my_model.mu_analysis(mu_list)

    # Comapre efficiency under varied mu 
    # Output: graph
    if args.all or args.learning:
        # Analyse dynamics by altering parameter mu
        mu_list = np.arange(0.01,1,0.02)
        my_model = ParameterAnalysis(path=args.path, filename='learning_mu')
        my_model.mu_learning(mu_list)

    # Observe mm profits under varied mus
    # Output: graph
    if args.all or args.profit:
        # Analyse expected profits at different parameter mu 
        mu_list = np.arange(0.1,1,0.1)
        my_model = ParameterAnalysis(path=args.path, filename='profit_mu')
        my_model.mu_profits(mu_list)

    # Welfare of all agents
    # Output: graph
    if args.all or args.welfare:
        mu_list = [0.2,0.5,0.8]
        my_model = ParameterAnalysis(path=args.path, filename='welfare_mu')
        my_model.mu_welfare(mu_list)