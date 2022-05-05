import pandas as pd
from scipy.stats import ttest_ind


def main():
    cpd = pd.read_csv("../data/cpd_oos_results.csv")
    nn = pd.read_csv("../data/nn_oos_results.csv")
    rf = pd.read_csv("../data/rf_oos_results.csv")

    print("CSID")
    print(cpd.log_loss.mean())
    print(cpd.log_loss.std())
    print()
    print("NN")
    print(nn.log_loss.mean())
    print(nn.log_loss.std())
    print()
    print("RF")
    print(rf.log_loss.mean())
    print(rf.log_loss.std())
    print()
    print("CSID vs NN")
    print(ttest_ind(cpd.log_loss, nn.log_loss, equal_var=False, alternative="less"))
    print()
    print("CSID vs RF")
    print(ttest_ind(cpd.log_loss, rf.log_loss, equal_var=False, alternative="less"))


if __name__ == "__main__":
    main()
