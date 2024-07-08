# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:40:09 2024


"""


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scipy.stats as st
from itertools import product
import networkx as nx
import matplotlib.pyplot as plt
from vinecopulas.bivariate import *

# %% Copulas

copulas = {
    1: "Gaussian",
    2: "Gumbel0",
    3: "Gumbel90",
    4: "Gumbel180",
    5: "Gumbel270",
    6: "Clayton0",
    7: "Clayton90",
    8: "Clayton180",
    9: "Clayton270",
    10: "Frank",
    11: "Joe0",
    12: "Joe90",
    13: "Joe180",
    14: "Joe270",
    15: "Student",
}


# %% fitting vinecopula


def fit_vinecop(u1, copsi, vine="R", printing=True):
    """
    Fit a regular vine copula to data.

    Arguments:
        *u1* :  the data, provided as a numpy array where each column contains a separate variable (eg. u1,u2,...,un), which have already been transferred to standard uniform margins (0<= u <= 1)

        *copsi* : A list of integers referring to the copulae of interest for which the fit has to be evaluated in the vine copula. eg. a list of [1, 10] refers to the Gaussian and Frank copula  (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).

        *vine* : The type of vine copula that needs to be fit, either 'R', 'D', or 'C'

        *printing*: True if the fitted copula should be printed and False if not


    Returns:
     *a* : The vine tree structure provided as a triangular matrix, composed of integers. The integer refers to different variables depending on which column the variable was in u1, where the first column is 0 and the second column is 1, etc.

     *p* : Parameters of the bivariate copulae provided as a triangular matrix.

     *c* : The types of the bivariate copulae provided as a triangular matrix, composed of integers referring to the copulae with the best fit. eg. a 1 refers to the gaussian copula  (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).

    """
    # Reference: Dißmann et al. 2013

    v1 = []  # list for variable 1
    v2 = []  # list for  variable 2
    tauabs = []  # list for the absolute kendal tau between v1 and v2
    dimen = u1.shape[1]  # number of variables (number of columns)
    for i in range(dimen - 1):
        for j in range(i + 1, dimen):
            v1.append(int(i))  # add variable to v1
            v2.append(int(j))  # add variable to v2
            tauabs.append(
                abs(st.kendalltau(u1[:, i], u1[:, j])[0])
            )  # calculate the absolute kendall tau between v1 and v2 and add it to the taubs list

    order1 = pd.DataFrame(
        {"v1": v1, "v2": v2, "tauabs": tauabs}
    )  # put the v1, v2, and tauabs list into a dataframe for the first tree
    order1 = order1.sort_values(by="tauabs", ascending=False).reset_index(
        drop=True
    )  # sort this dataframe from highest to lowest tauabs

    # R vine
    if vine == "R":
        inde = []  # list to put used rows of order1 in
        for i in range(
            len(order1)
        ):  # loop through all the rows to include those pairs with the highest ktau first
            if i == 0:
                order2 = order1.head(
                    1
                )  # create a new dataframe where the row with the highest ktau is the first row
            else:
                if (
                    order1.v1[i] in list(order1.v2[:i])
                    or order1.v1[i] in list(order1.v1[:i])
                ) and (
                    order1.v2[i] in list(order1.v2[:i])
                    or order1.v2[i] in list(order1.v1[:i])
                ):  # see if both v1i and v2i are already in order2, first rows with unique variables need to be added to ensure all variables are included
                    continue
                else:
                    inde.append(i)  # add used rows to inde
                    order2 = pd.concat(
                        [order2, order1.loc[i].to_frame().T], ignore_index=True
                    )  # add row to dataframe order2
        if len(order2) < (
            dimen - 1
        ):  # the first tree will have the number of variables - 1 number of rows.
            for i in range(
                1, len(order1)
            ):  # loop through all the combinations again and add based on the highest ktauabs
                if i in inde:  # check if row has not been added before
                    continue
                lst = (
                    list(order2.v2[order2.v1 == order1.v2[i]])
                    + list(order2.v1[order2.v2 == order1.v2[i]])
                    + list(order2.v2[order2.v1 == order1.v1[i]])
                    + list(order2.v1[order2.v2 == order1.v1[i]])
                )
                l1 = list(order2.v2[order2.v1 == order1.v2[i]]) + list(
                    order2.v1[order2.v2 == order1.v2[i]]
                )
                lk = l1.copy()
                while len(lk) > 0:
                    lk2 = lk.copy()
                    lk = []
                    for j in lk2:
                        ln = list(order2.v2[order2.v1 == j]) + list(
                            order2.v1[order2.v2 == j]
                        )
                        try:
                            ln.remove(order1.v2[i])
                        except:
                            pass

                        for s in l1:
                            try:
                                ln.remove(s)
                            except:
                                pass
                        l1 = l1 + ln
                        lk = lk + ln

                l2 = list(order2.v2[order2.v1 == order1.v1[i]]) + list(
                    order2.v1[order2.v2 == order1.v1[i]]
                )
                lk = l2.copy()
                while len(lk) > 0:
                    lk2 = lk.copy()
                    lk = []
                    for j in lk2:
                        ln = list(order2.v2[order2.v1 == j]) + list(
                            order2.v1[order2.v2 == j]
                        )
                        try:
                            ln.remove(order1.v1[i])
                        except:
                            pass

                        for s in l2:
                            try:
                                ln.remove(s)
                            except:
                                pass
                        l2 = l2 + ln
                        lk = lk + ln
                skip = False
                for val in l1:
                    if val in l2:
                        skip = True
                        break
                if skip == True:
                    continue
                # if (len(l1) > 1 and len(l2) >0) or (len(l1) > 0 and len(l2) >1): # ensure that there are no groups of variables that all have a connection with one another
                #  continue
                if len(lst) == len(
                    set(lst)
                ):  # check that there are no 'triangles', e.g. three variables that are alll connected with eachother
                    order2 = pd.concat(
                        [order2, order1.loc[i].to_frame().T], ignore_index=True
                    )  # add row if conditions are met
                if (
                    len(order2) == dimen - 1
                ):  # end loop when desired number of rows (number of variables - 1) has been met
                    break

    # D vine
    elif vine == "D":
        inde = []  # lsit to put used variables in
        for j in range(dimen - 1):
            if j == 0:
                order2 = order1.head(
                    1
                )  # loop through all the rows to include those pairs with the highest ktau first
                vi = order2.v1[0]  # select the first variable included in this row
                vj = order2.v2[0]  # select the second variable included in this row
            else:
                for i in range(len(order1)):
                    if (
                        order1.v1[i] == vj
                        or order1.v2[i] == vi
                        or order1.v1[i] == vi
                        or order1.v2[i] == vj
                    ):  # find the rows which include at least one of the two variables from the previous edge
                        if (
                            order1.v1[i] in inde or order1.v2[i] in inde
                        ):  # check if variables have already been connected to other variables twice
                            continue
                        if (order1.v1[i] == vi and order1.v2[i] == vj) or (
                            order1.v1[i] == vj and order1.v2[i] == vi
                        ):  # skip rows that have already been used
                            continue
                        else:
                            order2 = pd.concat(
                                [order2, order1.loc[i].to_frame().T], ignore_index=True
                            )  # if conditions are met, add this to the dataframe of the first tree
                            if (
                                order1.v1[i] == vj or order1.v2[i] == vj
                            ):  # check which value has been used twice already
                                inde.append(vj)  # add this value to inde
                                if order1.v1[i] == vj:
                                    vj = order1.v2[
                                        i
                                    ]  # select the new edge to connect to
                                else:
                                    vj = order1.v1[
                                        i
                                    ]  # select the new edge to connect to
                            elif (
                                order1.v1[i] == vi or order1.v2[i] == vi
                            ):  # check which value has been used twice already
                                inde.append(vi)  # add this value to inde
                                if order1.v1[i] == vi:
                                    vi = order1.v2[
                                        i
                                    ]  # select the new edge to connect to
                                else:
                                    vi = order1.v1[
                                        i
                                    ]  # select the new edge to connect to
    elif vine == "C":
        taus = []  # list to put the sum of all tauabs in for a specific variable
        for i in range(dimen):
            taus.append(
                sum(order1[(order1.v1 == i) | (order1.v2 == i)].tauabs)
            )  # calculate the sum of all taus for a specific variable
        i = np.where(np.array(taus) == max(taus))[0][
            0
        ]  # find where the sum of the taus is the highest to find the vairable to place in the center of the first tree
        order2 = order1[(order1.v1 == i) | (order1.v2 == i)].reset_index(
            drop=True
        )  # select the rows that include this variable

    order1 = order2  # make order1 == the first tree
    del order2
    rhos = []  # list for the rhos
    node = []  # list for the nodes
    cops = []  # list for the copulas
    v1_1 = []  # list for the 1st nodes
    v2_1 = []  # list for the 2nd nodes
    aics = []  # list for the AIC's
    for i in range(len(order1)):
        v1i = int(order1.v1[i])  # first node
        v2i = int(order1.v2[i])  # second node
        u3 = np.vstack(
            (u1[:, v1i], u1[:, v2i])
        ).T  # stacking the combination to fit copula to
        cop, rho, aic = bestcop(copsi, u3)  # fit the best copula
        aics.append(aic)  # add AIC to aics
        rhos.append(rho)  # add parameters to rhos
        cops.append(cop)  # add copula to cops
        node.append([v1i, v2i])  # create final node
        v1_1.append(u1[:, v1i])  # add array of first node
        v2_1.append(u1[:, v2i])  # add array of second node
    v1_1 = np.array(v1_1).T  # v1
    v2_1 = np.array(v2_1).T  # v2

    # add information to dataframe of the first tree
    order1["rhos"] = rhos
    order1["node"] = node
    order1["tree"] = 0
    order1["cop"] = cops
    order1["AIC"] = aics

    # set up variables for the second tree
    v1 = []
    v2 = []
    ktau = []
    rhos = []
    node = []
    cops = []
    v1_k = []
    v2_k = []
    aics = []
    for i in range(
        len(order1) - 1
    ):  # loop through the first tree to identify all possible combination of nodes
        v1i = int(order1.v1[i])  # parent node 1 from nodei in tree
        v2i = int(order1.v2[i])  # parent node 2 from nodei in tree
        copi = int(order1.cop[i])  # copula of nodei
        pari = order1.rhos[i]  # parameters of nodei
        for j in (
            np.where(
                np.array([item == v1i for item in list(order1.v1[i + 1 :])])
                | np.array([item == v1i for item in list(order1.v2[i + 1 :])])
                | np.array([item == v2i for item in list(order1.v1[i + 1 :])])
                | np.array([item == v2i for item in list(order1.v2[i + 1 :])])
            )[0]
            + i
            + 1
        ):  # see if possible connection between nodei and nodej
            v1j = int(order1.v1[j])  # parent node 1 from nodej in tree
            v2j = int(order1.v2[j])  # parent node 2 from nodej in tree
            copj = int(order1.cop[j])  # copula of nodei
            parj = order1.rhos[j]  # parameters of nodei
            v1.append(order1.node[i])  # parent node for next tree
            v2.append(order1.node[j])  # parent node for next tree
            lst = order1.node[i] + order1.node[j]  # list of all variables in node
            s = max(
                set(lst), key=lst.count
            )  # variable that is common in both parent nodes
            # parent node values
            ui1 = v1_1[:, i]
            ui2 = v2_1[:, i]
            uj1 = v1_1[:, j]
            uj2 = v2_1[:, j]
            # defining which parent node the conditional CDF needs to be based on
            if v1i == s:
                uni = 1
                vi1 = v2i
            else:
                uni = 2
                vi1 = v1i
            if v1j == s:
                unj = 1
                vj1 = v2j
            else:
                unj = 2
                vj1 = v1j
            # calculate the conditional CDF
            v1igs = hfunc(copi, ui1, ui2, pari, un=uni)
            v2jgs = hfunc(copj, uj1, uj2, parj, un=unj)
            ktau.append(
                abs(st.kendalltau(v1igs, v2jgs)[0])
            )  # calculate the absolute kendall tau between v1igs and v2jgs, add it to the ktau list
            # add the parent node values
            v1_k.append(v1igs)
            v2_k.append(v2jgs)
            # format the final node
            node.append([vi1, vj1, "g", s])

    k = 2  # second tree

    orderk = pd.DataFrame(
        {"v1": v1, "v2": v2, "tauabs": ktau, "node": node}
    )  # put the v1, v2, tauabs, and the node in list into a dataframe for tree k

    if vine == "R" or vine == "D":
        if (
            len(orderk) > dimen - k
        ):  # the tree will have the number of variables - k number of rows, if this is exceeded in orderk, a selection has to be made based on kendal tau
            orderk = orderk.sort_values(
                by="tauabs", ascending=False
            )  # sort this dataframe from highest to lowest tauabs
            indexes = list(
                orderk.index
            )  # create a list the original order of the dataframe prior to sorting
            orderk = orderk.reset_index(drop=True)  # reset the index
            inde = (
                []
            )  #  list to put used rows of orderk in based on their oroginal position in the dataframe
            inde2 = []  # list to put used rows of orderk i
            for i in range(len(orderk)):
                if i == 0:
                    order = orderk.head(
                        1
                    )  # loop through all the rows to include those pairs with the highest ktau first
                    inde.append(
                        indexes[i]
                    )  # add used rows to inde (original row value)

                else:
                    if (
                        orderk.v1[i] in list(orderk.v2[:i])
                        or orderk.v1[i] in list(orderk.v1[:i])
                    ) and (
                        orderk.v2[i] in list(orderk.v2[:i])
                        or orderk.v2[i] in list(orderk.v1[:i])
                    ):
                        continue
                    else:
                        inde.append(
                            indexes[i]
                        )  # add used rows to inde (original row value)
                        inde2.append(i)  # add used rows to inde
                        order = pd.concat(
                            [order, orderk.loc[i].to_frame().T], ignore_index=True
                        )  # if conditions are met, add this to the dataframe of tree k

            if len(order) < (dimen - k):
                for i in range(1, len(orderk)):
                    if i in inde2:
                        continue
                    lst = (
                        list(
                            order.v2.astype(str)[
                                order.v1.astype(str) == str(orderk.v1[i])
                            ]
                        )
                        + list(
                            order.v1.astype(str)[
                                order.v2.astype(str) == str(orderk.v1[i])
                            ]
                        )
                        + list(
                            order.v2.astype(str)[
                                order.v1.astype(str) == str(orderk.v2[i])
                            ]
                        )
                        + list(
                            order.v1.astype(str)[
                                order.v2.astype(str) == str(orderk.v2[i])
                            ]
                        )
                    )
                    l1 = list(
                        order.v2.astype(str)[order.v1.astype(str) == str(orderk.v1[i])]
                    ) + list(
                        order.v1.astype(str)[order.v2.astype(str) == str(orderk.v1[i])]
                    )
                    lk = l1.copy()
                    while len(lk) > 0:
                        lk2 = lk.copy()
                        lk = []
                        for j in lk2:
                            ln = list(
                                order.v2.astype(str)[order.v1.astype(str) == j]
                            ) + list(order.v1.astype(str)[order.v2.astype(str) == j])
                            try:
                                ln.remove(str(orderk.v1[i]))
                            except:
                                pass

                            for s in l1:
                                try:
                                    ln.remove(s)
                                except:
                                    pass
                            l1 = l1 + ln
                            lk = lk + ln

                    l2 = list(
                        order.v2.astype(str)[order.v1.astype(str) == str(orderk.v2[i])]
                    ) + list(
                        order.v1.astype(str)[order.v2.astype(str) == str(orderk.v2[i])]
                    )
                    lk = l2.copy()
                    while len(lk) > 0:
                        lk2 = lk.copy()
                        lk = []
                        for j in lk2:
                            ln = list(
                                order.v2.astype(str)[order.v1.astype(str) == j]
                            ) + list(order.v1.astype(str)[order.v2.astype(str) == j])
                            try:
                                ln.remove(str(orderk.v2[i]))
                            except:
                                pass

                            for s in l2:
                                try:
                                    ln.remove(s)
                                except:
                                    pass
                            l2 = l2 + ln
                            lk = lk + ln
                    skip = False
                    for val in l1:
                        if val in l2:
                            skip = True
                            break
                    if skip == True:
                        continue
                    # if (len(l1) > 1 and len(l2) >0) or (len(l1) > 0 and len(l2) >1):  # ensure that there are no groups of variables that all have a connection with one another
                    #   continue
                    if len(lst) == len(set(lst)):
                        inde.append(
                            indexes[i]
                        )  # add used rows to inde (original row value)
                        order = pd.concat(
                            [order, orderk.loc[i].to_frame().T], ignore_index=True
                        )  # if conditions are met, add this to the dataframe of tree k
                    if (
                        len(order) == dimen - k
                    ):  # end loop when desired number of rows (number of variables - k) has been met
                        break
            orderk = order  # maker order = to tree k
            orderk = orderk.sort_values(
                by="tauabs", ascending=False
            )  # sort this dataframe from highest to lowest tauabs
            v1_k = np.array([v1_k[ind] for ind in inde]).T  # sort array of first node
            v2_k = np.array([v2_k[ind] for ind in inde]).T  # sort array of second node
            orderk = orderk.reset_index(drop=True)
            orderk["tree"] = k - 1
            v1_2 = v1_k.copy()
            v2_2 = v2_k.copy()
            order2 = orderk.copy()

        else:
            orderk = orderk.sort_values(
                by="tauabs", ascending=False
            )  # sort this dataframe from highest to lowest tauabs
            v1_k = np.array(
                [v1_k[ind] for ind in orderk.index]
            ).T  # sort array of first node
            v2_k = np.array(
                [v2_k[ind] for ind in orderk.index]
            ).T  # sort array of second node
            orderk = orderk.reset_index(drop=True)
            orderk["tree"] = k - 1
            v1_2 = v1_k.copy()
            v2_2 = v2_k.copy()
            order2 = orderk.copy()
    if vine == "C":
        if len(orderk) > dimen - k:
            subnodes = np.unique(
                np.stack((np.array(orderk.v1), np.array(orderk.v2)))
            )  # see all the unique parent nodes
            taus = []  # list to put the sum of all tauabs in for a specific nodes
            for i in range(len(subnodes)):
                orderksub = orderk[
                    (orderk["v1"].apply(lambda x: x == subnodes[i]))
                    | (orderk["v2"].apply(lambda x: x == subnodes[i]))
                ]
                taus.append(
                    sum(orderksub.tauabs)
                )  # calculate the sum of all taus for a specific variable
            i = np.where(np.array(taus) == max(taus))[0][
                0
            ]  # find where the sum of the taus is the highest to find the vairable to place in the center of the first tree
            orderk = orderk[
                (orderk["v1"].apply(lambda x: x == subnodes[i]))
                | (orderk["v2"].apply(lambda x: x == subnodes[i]))
            ]  # select rows where this node is included
            orderk = orderk.sort_values(
                by="tauabs", ascending=False
            )  # sort this dataframe from highest to lowest tauabs
            inde = list(orderk.index)
            v1_k = np.array([v1_k[ind] for ind in inde]).T  # sort array of first node
            v2_k = np.array([v2_k[ind] for ind in inde]).T  # sort array of second node
            orderk = orderk.reset_index(drop=True)
            orderk["tree"] = k - 1
            v1_2 = v1_k.copy()
            v2_2 = v2_k.copy()
            order2 = orderk.copy()
        else:
            orderk = orderk.sort_values(by="tauabs", ascending=False)
            v1_k = np.array([v1_k[ind] for ind in orderk.index]).T
            v2_k = np.array([v2_k[ind] for ind in orderk.index]).T
            orderk = orderk.reset_index(drop=True)
            orderk["tree"] = k - 1
            v1_2 = v1_k.copy()
            v2_2 = v2_k.copy()
            order2 = orderk.copy()

    for i in range(len(order2)):
        u3 = np.vstack(
            (v1_2[:, i], v2_2[:, i])
        ).T  # stacking the combination to fit copula to
        cop, rho, aic = bestcop(copsi, u3)  # fit the best copula
        aics.append(aic)  # add AIC to aics
        rhos.append(rho)  # add parameters to rhos
        cops.append(cop)  # add copula to cops

    # add information to dataframe of tree k
    order2["rhos"] = rhos
    order2["cop"] = cops
    order2["AIC"] = aics

    # loop through remaining trees if there are more than 3 variables
    if dimen > 3:
        for k in range(3, dimen):
            order = locals()[
                "order" + str(k - 1)
            ].copy()  # select the struture of the previous tree
            v1s = locals()[
                "v1_" + str(k - 1)
            ].copy()  # select the first nodes of the previous tree
            v2s = locals()[
                "v2_" + str(k - 1)
            ].copy()  # select the second nodes of the previous tree
            # create lists for the variables
            v1_k = []
            v2_k = []
            v1 = []
            v2 = []
            ktau = []
            rhos = []
            cops = []
            node = []
            lk = []
            aics = []
            rk = []
            for i in range(len(order) - 1):
                v1i = order.v1[i].copy()  # parent node 1 from nodei in tree
                v2i = order.v2[i].copy()  # parent node 2 from nodei in tree
                copi = int(order.cop[i])  # copula of nodei
                pari = order.rhos[i]  # parameters of nodei
                for j in (
                    np.where(
                        np.array([item == v1i for item in list(order.v1[i + 1 :])])
                        | np.array([item == v1i for item in list(order.v2[i + 1 :])])
                        | np.array([item == v2i for item in list(order.v1[i + 1 :])])
                        | np.array([item == v2i for item in list(order.v2[i + 1 :])])
                    )[0]
                    + i
                    + 1
                ):  # see if possible connection between nodei and nodej
                    v1i = order.v1[i].copy()  # parent node 1 from nodei in tree
                    v2i = order.v2[i].copy()  # parent node 2 from nodei in tree
                    copj = int(order.cop[j])  # copula of nodei
                    parj = order.rhos[j]  # parameters of nodei
                    nodei = order.node[i]  # nodei
                    nodej = order.node[j]  # nodej
                    v1j = order.v1[j].copy()  # parent node 1 from nodej in tree
                    v2j = order.v2[j].copy()  # parent node 2 from nodej in tree
                    v1.append(nodei)  # new parent nodes 1
                    v2.append(nodej)  # new parent nodes 2
                    n = 2
                    ri = nodei[n + 1 :]  # select the values on the right side of node i
                    rj = nodej[n + 1 :]  # select the values on the right side of node i

                    if "g" in v1j:
                        v1j.remove("g")
                        v2j.remove("g")
                        v1i.remove("g")
                        v2i.remove("g")

                    # define left and right side of the node in tree k
                    if rj == ri:
                        if len(v1j) == 2:
                            lst = nodei[:n] + nodej[:n]
                            r3 = [max(set(lst), key=lst.count)]
                            li = [value for value in nodei[:n] if value not in r3]
                            lj = [value for value in nodej[:n] if value not in r3]
                            r = list(np.unique(ri + r3))
                        else:
                            lst = v1i[:n] + v2i[:n] + v1j[:n] + v2j[:n]
                            for s in ri:
                                lst = [x for x in lst if x != s]
                            r3 = [max(set(lst), key=lst.count)]
                            li = [value for value in nodei[:n] if value not in r3]
                            lj = [value for value in nodej[:n] if value not in r3]
                            r = list(np.unique(ri + r3))
                        l = list(np.unique(li + lj))
                    else:
                        r = list(np.unique(ri + rj))
                        li = [value for value in nodei[:n] if value not in rj]
                        lj = [value for value in nodej[:n] if value not in ri]
                        if li == lj:
                            lst = v1i[:n] + v2i[:n] + v1j[:n] + v2j[:n]
                            lst = [x for x in lst if x != li[0]]
                            l3 = [min(set(lst), key=lst.count)]
                            l = list(np.unique(li + l3))
                        else:
                            l = list(np.unique(li + lj))
                    # select the parent node values
                    ui1 = v1s[:, i]
                    ui2 = v2s[:, i]
                    uj1 = v1s[:, j]
                    uj2 = v2s[:, j]
                    if set(r).issubset(set(v1i)):
                        uni = 1
                    elif set(r).issubset(set(v2i)):
                        uni = 2
                    elif set(rj).issubset(set(v2i[1:])):
                        uni = 2
                    elif set(rj).issubset(set(v1i[1:])):
                        uni = 1
                    if set(r).issubset(set(v1j)):
                        unj = 1
                    elif set(r).issubset(set(v2j)):
                        unj = 2
                    elif set(ri).issubset(set(v2j[1:])):
                        unj = 2
                    elif set(ri).issubset(set(v1j[1:])):
                        unj = 1

                    # calculate the conditional CDF
                    v1igs = hfunc(copi, ui1, ui2, pari, un=uni)
                    v2jgs = hfunc(copj, uj1, uj2, parj, un=unj)
                    ktau.append(
                        abs(st.kendalltau(v1igs, v2jgs)[0])
                    )  # calculate the absolute kendall tau between v1igs and v2jgs, add it to the ktau list
                    # add the parent node values
                    v1_k.append(v1igs)
                    v2_k.append(v2jgs)

                    del uj1, ui1, ui2, uj2

                    node.append(l + ["g"] + r)  # set node name
                    lk.append(l)  # add left side of node
                    rk.append(r)  # add rigth side of node
            orderk = pd.DataFrame(
                {"v1": v1, "v2": v2, "tauabs": ktau, "node": node, "l": lk, "r": rk}
            )  # put information in dataframe for tree k

            if vine == "R" or vine == "D":
                if (
                    len(orderk) > dimen - k
                ):  # the tree will have the number of variables - k number of rows, if this is exceeded in orderk, a selection has to be made based on kendal tau
                    orderk = orderk.sort_values(
                        by="tauabs", ascending=False
                    )  # sort this dataframe from highest to lowest tauabs
                    indexes = list(
                        orderk.index
                    )  # create a list the original order of the dataframe prior to sorting
                    orderk = orderk.reset_index(drop=True)  # reset the index
                    inde = (
                        []
                    )  #  list to put used rows of orderk in based on their oroginal position in the dataframe
                    inde2 = []  # list to put used rows of orderk i
                    for i in range(len(orderk)):
                        if i == 0:
                            order = orderk.head(
                                1
                            )  # loop through all the rows to include those pairs with the highest ktau first
                            inde.append(
                                indexes[i]
                            )  # add used rows to inde (original row value)
                            l = orderk.l[i]

                        else:
                            if (
                                orderk.v1[i] in list(orderk.v2[:i])
                                or orderk.v1[i] in list(orderk.v1[:i])
                            ) and (
                                orderk.v2[i] in list(orderk.v2[:i])
                                or orderk.v2[i] in list(orderk.v1[:i])
                            ):
                                continue
                            else:
                                inde.append(
                                    indexes[i]
                                )  # add used rows to inde (original row value)
                                inde2.append(i)  # add used rows to inde
                                order = pd.concat(
                                    [order, orderk.loc[i].to_frame().T],
                                    ignore_index=True,
                                )  # if conditions are met, add this to the dataframe of tree k
                                l = l + orderk.l[i]
                    if len(order) < (dimen - k):
                        for i in range(1, len(orderk)):
                            if i in inde2:
                                continue
                            lst = (
                                list(
                                    order.v2.astype(str)[
                                        order.v1.astype(str) == str(orderk.v1[i])
                                    ]
                                )
                                + list(
                                    order.v1.astype(str)[
                                        order.v2.astype(str) == str(orderk.v1[i])
                                    ]
                                )
                                + list(
                                    order.v2.astype(str)[
                                        order.v1.astype(str) == str(orderk.v2[i])
                                    ]
                                )
                                + list(
                                    order.v1.astype(str)[
                                        order.v2.astype(str) == str(orderk.v2[i])
                                    ]
                                )
                            )
                            l1 = list(
                                order.v2.astype(str)[
                                    order.v1.astype(str) == str(orderk.v1[i])
                                ]
                            ) + list(
                                order.v1.astype(str)[
                                    order.v2.astype(str) == str(orderk.v1[i])
                                ]
                            )
                            lk = l1.copy()
                            while len(lk) > 0:
                                lk2 = lk.copy()
                                lk = []
                                for j in lk2:
                                    ln = list(
                                        order.v2.astype(str)[order.v1.astype(str) == j]
                                    ) + list(
                                        order.v1.astype(str)[order.v2.astype(str) == j]
                                    )
                                    try:
                                        ln.remove(str(orderk.v1[i]))
                                    except:
                                        pass

                                    for s in l1:
                                        try:
                                            ln.remove(s)
                                        except:
                                            pass
                                    l1 = l1 + ln
                                    lk = lk + ln

                            l2 = list(
                                order.v2.astype(str)[
                                    order.v1.astype(str) == str(orderk.v2[i])
                                ]
                            ) + list(
                                order.v1.astype(str)[
                                    order.v2.astype(str) == str(orderk.v2[i])
                                ]
                            )
                            lk = l2.copy()
                            while len(lk) > 0:
                                lk2 = lk.copy()
                                lk = []
                                for j in lk2:
                                    ln = list(
                                        order.v2.astype(str)[order.v1.astype(str) == j]
                                    ) + list(
                                        order.v1.astype(str)[order.v2.astype(str) == j]
                                    )
                                    try:
                                        ln.remove(str(orderk.v2[i]))
                                    except:
                                        pass

                                    for s in l2:
                                        try:
                                            ln.remove(s)
                                        except:
                                            pass
                                    l2 = l2 + ln
                                    lk = lk + ln
                            skip = False
                            for val in l1:
                                if val in l2:
                                    skip = True
                                    break
                            if skip == True:
                                continue
                            # if (len(l1) > 1 and len(l2) >0) or (len(l1) > 0 and len(l2) >1):
                            # continue
                            if len(lst) == len(set(lst)):
                                order = pd.concat(
                                    [order, orderk.loc[i].to_frame().T],
                                    ignore_index=True,
                                )  # if conditions are met, add this to the dataframe of tree k
                                inde.append(
                                    indexes[i]
                                )  # add used rows to inde (original row value)
                            if (
                                len(order) == dimen - k
                            ):  # end loop when desired number of rows (number of variables - k) has been met
                                break
                    orderk = order
                    orderk = orderk.sort_values(
                        by="tauabs", ascending=False
                    )  # sort this dataframe from highest to lowest tauabs

                    v1_k = np.array(
                        [v1_k[ind] for ind in inde]
                    ).T  # sort array of first node
                    v2_k = np.array(
                        [v2_k[ind] for ind in inde]
                    ).T  # sort array of second node
                    orderk = orderk.reset_index(drop=True)
                    orderk["tree"] = k - 1
                    for j in range(len(orderk)):
                        u3 = np.vstack(
                            (v1_k[:, j], v2_k[:, j])
                        ).T  # stacking the combination to fit copula to
                        cop, rho, aic = bestcop(copsi, u3)  # fit the best copula
                        aics.append(aic)  # add AIC to aics
                        rhos.append(rho)  # add parameters to rhos
                        cops.append(cop)  # add copula to cops

                    # add information to dataframe of tree k
                    orderk["rhos"] = rhos
                    orderk["cop"] = cops
                    orderk["AIC"] = aics
                    locals()["v1_" + str(k)] = v1_k
                    locals()["v2_" + str(k)] = v2_k
                    locals()["order" + str(k)] = orderk

                else:
                    orderk = orderk.sort_values(
                        by="tauabs", ascending=False
                    )  # sort this dataframe from highest to lowest tauabs
                    v1_k = np.array(
                        [v1_k[ind] for ind in orderk.index]
                    ).T  # sort array of first node
                    v2_k = np.array(
                        [v2_k[ind] for ind in orderk.index]
                    ).T  # sort array of second node
                    orderk = orderk.reset_index(drop=True)
                    orderk["tree"] = k - 1

                    for j in range(len(orderk)):
                        u3 = np.vstack(
                            (v1_k[:, j], v2_k[:, j])
                        ).T  # stacking the combination to fit copula to
                        cop, rho, aic = bestcop(copsi, u3)  # fit the best copula
                        aics.append(aic)  # add AIC to aics
                        rhos.append(rho)  # add parameters to rhos
                        cops.append(cop)  # add copula to cops

                    # add information to dataframe of tree k
                    orderk["rhos"] = rhos
                    orderk["cop"] = cops
                    orderk["AIC"] = aics
                    locals()["v1_" + str(k)] = v1_k
                    locals()["v2_" + str(k)] = v2_k
                    locals()["order" + str(k)] = orderk

            if vine == "C":
                if len(orderk) > dimen - k:
                    subnodes = np.unique(
                        np.stack((np.array(orderk.v1), np.array(orderk.v2)))
                    )  # see all the unique parent nodes
                    taus = (
                        []
                    )  # list to put the sum of all tauabs in for a specific nodes
                    for i in range(len(subnodes)):
                        orderksub = orderk[
                            (orderk["v1"].apply(lambda x: x == subnodes[i]))
                            | (orderk["v2"].apply(lambda x: x == subnodes[i]))
                        ]
                        taus.append(
                            sum(orderksub.tauabs)
                        )  # calculate the sum of all taus for a specific variable
                    i = np.where(np.array(taus) == max(taus))[0][
                        0
                    ]  # find where the sum of the taus is the highest to find the vairable to place in the center of the first tree
                    orderk = orderk[
                        (orderk["v1"].apply(lambda x: x == subnodes[i]))
                        | (orderk["v2"].apply(lambda x: x == subnodes[i]))
                    ]  # select rows where this node is included
                    orderk = orderk.sort_values(
                        by="tauabs", ascending=False
                    )  # sort this dataframe from highest to lowest tauabs
                    inde = list(orderk.index)
                    v1_k = np.array(
                        [v1_k[ind] for ind in inde]
                    ).T  # sort array of first node
                    v2_k = np.array(
                        [v2_k[ind] for ind in inde]
                    ).T  # sort array of second node
                    orderk = orderk.reset_index(drop=True)
                    orderk["tree"] = k - 1
                    v1_k = v1_k.copy()
                    v2_k = v2_k.copy()
                    for j in range(len(orderk)):
                        u3 = np.vstack(
                            (v1_k[:, j], v2_k[:, j])
                        ).T  # stacking the combination to fit copula to
                        cop, rho, aic = bestcop(copsi, u3)  # fit the best copula
                        aics.append(aic)  # add AIC to aics
                        rhos.append(rho)  # add parameters to rhos
                        cops.append(cop)  # add copula to cops

                    # add information to dataframe of tree k
                    orderk["rhos"] = rhos
                    orderk["cop"] = cops
                    orderk["AIC"] = aics
                    locals()["v1_" + str(k)] = v1_k
                    locals()["v2_" + str(k)] = v2_k
                    locals()["order" + str(k)] = orderk

                else:
                    orderk = orderk.sort_values(by="tauabs", ascending=False)
                    v1_k = np.array([v1_k[ind] for ind in orderk.index]).T
                    v2_k = np.array([v2_k[ind] for ind in orderk.index]).T
                    orderk = orderk.reset_index(drop=True)
                    orderk["tree"] = k - 1

                    for j in range(len(orderk)):
                        u3 = np.vstack(
                            (v1_k[:, j], v2_k[:, j])
                        ).T  # stacking the combination to fit copula to
                        cop, rho, aic = bestcop(copsi, u3)  # fit the best copula
                        aics.append(aic)  # add AIC to aics
                        rhos.append(rho)  # add parameters to rhos
                        cops.append(cop)  # add copula to cops

                    # add information to dataframe of tree k
                    orderk["rhos"] = rhos
                    orderk["cop"] = cops
                    orderk["AIC"] = aics
                    locals()["v1_" + str(k)] = v1_k
                    locals()["v2_" + str(k)] = v2_k
                    locals()["order" + str(k)] = orderk

    order = pd.DataFrame(columns=order1.columns)  # dataframe to add all trees to
    for i in range(1, dimen):
        order = pd.concat([order, locals()["order" + str(i)]]).reset_index(
            drop=True
        )  # adding trees in dataframe

    # create array for the tree structure and copula matrices
    a = np.empty((dimen, dimen))
    c = np.empty((dimen, dimen))
    a[:] = np.nan
    c[:] = np.nan

    order["used"] = 0  # set used to 0 for nodes that have not been used
    for i in list(range(dimen - 1))[
        ::-1
    ]:  # create the matix for the tree structure starting with the last tree
        k1 = sorted(
            np.array(
                order[(order.tree == i) & (order["used"] == 0)].node.iloc[0][:2]
            ).astype(int)
        )[::-1]
        order.loc[(order["tree"] == i) & (order["used"] == 0), "used"] = 1
        t1 = i - 1
        ii = dimen - 2 - i
        a[i : dimen - ii, ii] = k1
        s = k1[-1]
        for j in list(range(0, i))[::-1]:  # continueing from tree i to the first tree
            orde = order[(order.tree == j) & (order["used"] == 0)]
            for k in range(len(orde)):
                arr = np.array(orde.node.iloc[k][:2]).astype(int)
                if np.isin(s, arr) == True:
                    inde = orde.iloc[k].name
                    a[j, ii] = arr[arr != s][0]
                    order["used"][inde] = 1

    a[0, dimen - 1] = a[0, dimen - 2]  # set first sample in sampling order
    orderk = pd.DataFrame(columns=order.columns)
    # create array for the copula parameter matrix
    p = np.empty((dimen, dimen))
    p[:] = np.nan
    p = p.astype(object)
    # fill array p with the parameters and c with the copulas, corresponding to the structure in c
    for i in list(range(dimen - 1)):
        orde = order[order.tree == i]
        for k in list(range(dimen - 1 - i)):
            ak = a[:, k]
            akn = np.array([ak[-1 - k], ak[i]]).astype(int)
            for j in range(len(orde)):
                arr = np.array(orde.node.iloc[j][:2]).astype(int)
                if sum(np.isin(akn, arr)) == 2:
                    orderj = order.loc[[orde.index[j]]]
                    p[i, k] = orderj.rhos.iloc[0]
                    c[i, k] = orderj.cop.iloc[0]
                    if i == 0:
                        orderj.node.iloc[0] = list(akn)
                    else:
                        orderj.node.iloc[0] = (
                            list(akn) + ["|"] + list((ak.astype(int)[:i])[::-1])
                        )
                    orderk = pd.concat([orderk, orderj]).reset_index(drop=True)

    # print the copula structure if print == True
    if printing == True:
        for i in list(range(0, dimen - 1)):
            orde = orderk[orderk.tree == i].reset_index(drop=True)
            print("** Tree: ", i + 1)
            for j in range(len(orde)):
                if i != 0:
                    nodej = ",".join(map(str, orde.node[j])).replace(",|,", "|")
                else:
                    nodej = ",".join(map(str, orde.node[j]))
                print(
                    nodej,
                    " ---> ",
                    copulas[int(orde.cop[j])],
                    ": parameters = ",
                    orde.rhos[j],
                )

    return a, p, c

def density_vinecop(u, M, P, C):
    """
    Computes the density function of a vine copula.

    Arguments:
        *u* :  A 2-d numpy array containing the samples for which the PDF will be calculated.

        *M* : The vine tree structure provided as a triangular matrix, composed of integers. The integer refers to different variables depending on which column the variable was in u1, where the first column is 0 and the second column is 1, etc.

        *P* : Parameters of the bivariate copulae provided as a triangular matrix.

        *C* : The types of the bivariate copulae provided as a triangular matrix, composed of integers referring to the copulae with the best fit. eg. a 1 refers to the gaussian copula  (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).




    Returns:
     *F* :  A 1-d numpy array containing the probability density function of the vine copula

    """
    U = u[:,list(np.diag(M[::-1])[::-1].astype(int))]

    a = M.copy()
    p = P.copy()
    c = C.copy()
    s = len(u)
    Ms = np.flipud(a)  # flip structure matrix
    P = np.flipud(p)  # flip parameter matrix
    C = np.flipud(c)  # flip copula matrix

    replace = {}  # dictionary for relabeling martix
    for i in range(int(max(np.unique(Ms)) + 1)):
        val = max(np.unique(Ms)) - i
        replace.update({Ms[i, i]: val})  # relabel

    Ms = np.nan_to_num(Ms, nan=int(max(np.unique(Ms)) + 1))
    replace_func = np.vectorize(
        lambda x: replace.get(x, x)
    )  # Create a vectorized function for replacement
    M = replace_func(Ms)  # relabel
    M[M == np.max(Ms)] = np.nan
    Mm = M.copy()
    # max matrix
    for i in range(M.shape[0]):
        for k in range(M.shape[0]):
            if k == i:
                continue
            if i == 0:
                continue
            Mm[i, k] = max(Mm[i:, k])

    # Vdirect
    Vdir = np.empty((s, M.shape[0], M.shape[0]))
    Vdir[:] = np.nan
    # Vindirect
    Vindir = np.empty((s, M.shape[0], M.shape[0]))
    Vindir[:] = np.nan
    # Z2
    Z2 = np.empty((s, M.shape[0], M.shape[0]))
    Z2[:] = np.nan
    # Z1
    Z1 = np.empty((s, M.shape[0], M.shape[0]))
    Z2[:] = np.nan
    Vdir[:, -1, :] =  np.flip(U.copy(), 1)
    X = np.flip(U.copy(), 1)
    n = M.shape[0] - 1
    F = 1

    for k in list(reversed((range(0,n)))):
        for i in range(k+1, n+1)[::-1]:

            Z1[:, i, k] = Vdir[:, i, k]
            if M[i, k] == Mm[i, k]:
                Z2[:, i, k] = Vdir[:, i, int(n - Mm[i, k])]
            else:
                Z2[:, i, k] = Vindir[:, i, int(n - Mm[i, k])]
            

            F = F * PDF(int(C[i, k]),np.vstack((Z1[:, i, k], Z2[:, i, k])).T,P[i, k])
            Vdir[:, int(i - 1), k] = hfunc(
                int(C[i, k]), Z1[:, i, k], Z2[:, i, k], P[i, k], un=2
            )
            Vindir[:, int(i - 1), k] = hfunc(
                int(C[i, k]), Z1[:, i, k], Z2[:, i, k], P[i, k], un=1
            )
    return F

    

# %% fitting vine copula with sspecific structure
def fit_vinecopstructure(u1, copsi, a):
    """
    Fit a regular vine copula to data based on a known vine structure matrix.

    Arguments:
        *u1* :  the data, provided as a numpy array where each column contains a seperate variable (eg. u1,u2,...,un), which have already been transferred to standard uniform margins (0<= u <= 1)

        *copsi* : A list of integers referring to the copulae of interest for which the fit has to be evauluated in the vine copula. eg. a list of [1, 10] refers to the Gaussian and Frank copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).

        *a* : The vine tree structure provided as a triangular matrix, composed of integers. The integer refers to different variables depending on which column the variable was in u1, where the first column is 0 and the second column is 1, etc.

    Returns:
     *p* : Parameters of the bivariate copulae provided as a triangular matrix.

     *c* : The types of the bivariate copulae provided as a triangular matrix, composed of integers referring to the copulae with the best fit. eg. a 1 refers to the gaussian copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).

    """

    dimen = a.shape[0]  # number of variables (number of columns)
    order = pd.DataFrame(
        columns=["node", "l", "r", "tree"]
    )  # dataframe for vinecopula information
    s = 0
    for i in list(range(dimen - 1)):
        for k in list(range(dimen - 1 - i)):
            ak = a[:, k]  # Edge
            akn = np.array([ak[-1 - k], ak[i]]).astype(int)  # Edge
            if i == 0:
                single_row_values = {
                    "node": list(akn),
                    "l": akn[0],
                    "r": akn[1],
                    "tree": i,
                }
            else:
                single_row_values = {
                    "node": list(akn) + ["|"] + list((ak.astype(int)[:i])[::-1]),
                    "l": list(akn),
                    "r": list((ak.astype(int)[:i])[::-1]),
                    "tree": i,
                }

            order.loc[s] = single_row_values
            s = s + 1

    for t in list(range(dimen - 1)):  # loop through trees
        orderk = order[order.tree == t].reset_index(drop=True)  # select tre
        if t == 0:  # first tree
            orderk["v1"] = orderk.l
            orderk["v2"] = orderk.r
            rhos = []
            v1_1 = []
            v2_1 = []
            cops = []
            tauabs = []

            for j in range(len(orderk)):
                v1i = int(orderk.v1[j])  # first node
                v2i = int(orderk.v2[j])  # second node
                tauabs.append(
                abs(st.kendalltau(u1[:, v1i], u1[:, v2i])[0])
            ) 
                u3 = np.vstack(
                    (u1[:, v1i], u1[:, v2i])
                ).T  # stacking the combination to fit copula to
                cop, rho, aic = bestcop(copsi, u3)  # fit the best copula
                rhos.append(rho)  # add parameters to rhos
                cops.append(cop)  # add copula to cops
                v1_1.append(u1[:, v1i])  # add array of first node
                v2_1.append(u1[:, v2i])  # add array of second node
            v1_1 = np.array(v1_1).T  # v1
            v2_1 = np.array(v2_1).T  # v2

            # add information to dataframe of the first tree
            orderk["rhos"] = rhos
            orderk["cop"] = cops
            orderk["tauabs"] = tauabs
            orderk = orderk.sort_values(by="tauabs", ascending=False)
            v1_1 = v1_1[:, orderk.index]  # sort array of first node
            v2_1 =  v2_1[:, orderk.index] # sort array of second node
            orderk = orderk.reset_index(
        drop=True
    ) 
            
            
            
            
            

        else:

            v1k = []
            v2k = []
            tauabs = []
            for j in range(len(orderk)):
                orderk2 = locals()["order" + str(t)] # Define possible nodes of edge
                l = orderk.l[j] + orderk.r[j]
                subnodes = []
                for k in range(len(orderk2)):
                    subnodes.append(sum(1 for item in orderk2.node[k] if item in l))
                subnodes = np.array(subnodes) == len(l) - 1
                orderk2 = orderk2[subnodes].reset_index(drop=True)
                v1k.append(orderk2.node[0])  # node 1
                v2k.append(orderk2.node[1])  # node 2
            orderk["v1"] = v1k
            orderk["v2"] = v2k
            orderk2 = locals()["order" + str(t)].reset_index(drop=True)
            v1s = locals()["v1_" + str(t)].copy()
            v2s = locals()["v2_" + str(t)].copy()
            v1_k = []
            v2_k = []
            rhos = []
            cops = []

            for k in range(len(orderk)):  # fitting copulas
                r = orderk.r[k]
                nodei = orderk.v1[k]  # nodei
                nodej = orderk.v2[k]  # nodej
                i = orderk2[orderk2["node"].apply(lambda x: x == nodei)].index[0]
                j = orderk2[orderk2["node"].apply(lambda x: x == nodej)].index[0]
                copj = int(orderk2.cop[j])  # copula of nodej
                parj = orderk2.rhos[j]  # parameters of nodej
                copi = int(orderk2.cop[i])  # copula of nodei
                pari = orderk2.rhos[i]  # parameters of nodei
                ri = orderk2.r[i]
                rj = orderk2.r[j]
                v1i = orderk2.v1[i]  # parent node 1 from nodei in tree
                v2i = orderk2.v2[i]
                v1j = orderk2.v1[j]  # parent node 1 from nodej in tree
                v2j = orderk2.v2[j]  # parent node 2 from nodej in tree
                ui1 = v1s[:, i]
                ui2 = v2s[:, i]
                uj1 = v1s[:, j]
                uj2 = v2s[:, j]
                if t > 1:
                    if "g" in v1j:
                        v1j.remove("g")
                        v2j.remove("g")
                        v1i.remove("g")
                        v2i.remove("g")
                    if set(r).issubset(set(v1i)):
                        uni = 1
                    elif set(r).issubset(set(v2i)):
                        uni = 2
                    elif set(rj).issubset(set(v2i[1:])):
                        uni = 2
                    elif set(rj).issubset(set(v1i[1:])):
                        uni = 1
                    if set(r).issubset(set(v1j)):
                        unj = 1
                    elif set(r).issubset(set(v2j)):
                        unj = 2
                    elif set(ri).issubset(set(v2j[1:])):
                        unj = 2
                    elif set(ri).issubset(set(v1j[1:])):
                        unj = 1

                else:
                    lst = orderk2.node[i] + orderk2.node[j]
                    s = max(set(lst), key=lst.count)
                    ui1 = v1_1[:, i]
                    ui2 = v2_1[:, i]
                    uj1 = v1_1[:, j]
                    uj2 = v2_1[:, j]
                    if v1i == s:
                        uni = 1
                        vi1 = v2i
                    else:
                        uni = 2
                        vi1 = v1i
                    if v1j == s:
                        unj = 1
                        vj1 = v2j
                    else:
                        unj = 2
                        vj1 = v1j
                # calculate the conditional CDF
                v1igs = hfunc(copi, ui1, ui2, pari, un=uni)
                v2jgs = hfunc(copj, uj1, uj2, parj, un=unj)
                u3 = np.vstack(
                    (v1igs, v2jgs)
                ).T  # stacking the combination to fit copula to
                cop, rho, aic = bestcop(copsi, u3)  # fit the best copula

                rhos.append(rho)  # add parameters to rhos
                cops.append(cop)  # add copula to cops
                v1_k.append(v1igs)
                v2_k.append(v2jgs)
            orderk["rhos"] = rhos
            orderk["cop"] = cops

            locals()["v1_" + str(t + 1)] = np.array(v1_k).T
            locals()["v2_" + str(t + 1)] = np.array(v2_k).T

        locals()["order" + str(t + 1)] = orderk
    order = pd.DataFrame(columns=orderk.columns)
    for i in range(1, dimen):
        order = pd.concat([order, locals()["order" + str(i)]]).reset_index(drop=True)

    p = np.empty((dimen, dimen))  # parameter array
    p[:] = np.nan
    p = p.astype(object)
    c = np.empty((dimen, dimen))  # copula array
    c[:] = np.nan
    for i in list(range(dimen - 1)):  # fil the copula and parameter array
        orde = order[order.tree == i]
        for k in list(range(dimen - 1 - i)):
            ak = a[:, k]
            akn = np.array([ak[-1 - k], ak[i]]).astype(int)
            for j in range(len(orde)):
                arr = np.array(orde.node.iloc[j][:2]).astype(int)
                if sum(np.isin(akn, arr)) == 2:
                    orderj = order.loc[[orde.index[j]]]
                    p[i, k] = orderj.rhos.iloc[0]
                    c[i, k] = orderj.cop.iloc[0]
    return p, c


# %% Sampling vine copula
def sample_vinecop(a, p, c, s):
    """
    Generate random samples from an R-vine.

    Arguments:
        *a* : The vine tree structure provided as a triangular matrix, composed of integers. The integer refers to different variables depending on which column the variable was in u1, where the first column is 0 and the second column is 1, etc.

        *p* : Parameters of the bivariate copulae provided as a triangular matrix.

        *c* : The types of the bivariate copulae provided as a triangular matrix, composed of integers referring to the copulae with the best fit. eg. a 1 refers to the gaussian copula  (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).

        *s* : number of samples to generate, provided as a positive scalar integer.



    Returns:
     *X2* :  the randomly sampled data data, provided as a numpy array where each column contains samples of a seperate variable (eg. u1,u2,...,un).

    """
    # Reference: Dißmann et al. 2013
    Ms = np.flipud(a)  # flip structure matrix
    P = np.flipud(p)  # flip parameter matrix
    C = np.flipud(c)  # flip copula matrix
    replace = {}  # dictionary for relabeling martix
    for i in range(int(max(np.unique(Ms)) + 1)):
        val = max(np.unique(Ms)) - i
        # Ms[k,k]
        replace.update({Ms[i, i]: val})  # relabel

    Ms = np.nan_to_num(Ms, nan=int(max(np.unique(Ms)) + 1))
    replace_func = np.vectorize(
        lambda x: replace.get(x, x)
    )  # Create a vectorized function for replacement
    M = replace_func(Ms)  # relabel
    M[M == np.max(Ms)] = np.nan
    Mm = M.copy()
    # max matrix
    for i in range(M.shape[0]):
        for k in range(M.shape[0]):
            if k == i:
                continue
            if i == 0:
                continue
            Mm[i, k] = max(Mm[i:, k])

    # Vdirect
    Vdir = np.empty((s, M.shape[0], M.shape[0]))
    Vdir[:] = np.nan
    # Vindirect
    Vindir = np.empty((s, M.shape[0], M.shape[0]))
    Vindir[:] = np.nan
    # Z2
    Z2 = np.empty((s, M.shape[0], M.shape[0]))
    Z2[:] = np.nan
    # Z1
    Z1 = np.empty((s, M.shape[0], M.shape[0]))
    Z2[:] = np.nan
    U = np.random.uniform(0, 1, (s, M.shape[0]))  # random uniform
    Vdir[:, -1, :] = U.copy()
    X = np.flip(U.copy(), 1)
    n = M.shape[0] - 1
    # sampling algorithm
    for k in range(n)[::-1]:
        for i in range(k + 1, n + 1):
            if M[i, k] == Mm[i, k]:
                Z2[:, i, k] = Vdir[:, i, int(n - Mm[i, k])]
            else:
                Z2[:, i, k] = Vindir[:, i, int(n - Mm[i, k])]
            Vdir[:, n, k] = hfuncinverse(
                int(C[i, k]), Z2[:, i, k], Vdir[:, n, k], P[i, k], un=2
            )
        X[:, int(n - k)] = Vdir[:, n, k]
        for i in range(k + 1, n + 1)[::-1]:
            Z1[:, i, k] = Vdir[:, i, k]
            Vdir[:, int(i - 1), k] = hfunc(
                int(C[i, k]), Z1[:, i, k], Z2[:, i, k], P[i, k], un=2
            )
            Vindir[:, int(i - 1), k] = hfunc(
                int(C[i, k]), Z1[:, i, k], Z2[:, i, k], P[i, k], un=1
            )

    # Put X in the original order of the data
    replacedf = pd.DataFrame(list(replace.items()), columns=["Original", "Replacement"])
    replacedf = replacedf.sort_values(by="Original")
    X2 = np.array([])
    for i in replacedf.Replacement:
        if len(X2) == 0:
            X2 = X[:, int(i)].reshape(len(X), 1)
        else:
            X2 = np.hstack((X2, X[:, int(i)].reshape(len(X), 1)))
    return X2


# %% Sampling conditonal vine copula
def sample_vinecopconditional(a, p, c, s, Xc):
    """
    Generate conditional samples from an R-vine based on a provided sampling order

    Arguments:
         *a* : The vine tree structure provided as a triangular matrix, composed of integers. The integer refers to different variables depending on which column the variable was in u1, where the first column is 0 and the second column is 1, etc.

         *p* : Parameters of the bivariate copulae provided as a triangular matrix.

         *c* : The types of the bivariate copulae provided as a triangular matrix, composed of integers referring to the copulae with the best fit. eg. a 1 refers to the gaussian copula  (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).

         *s* : number of samples to generate, provided as a positive scalar integer.

         *XC*: the values of the variables on which the conditional sample has to generated, provided as a 1d array that contains the the values ordered in terms of the sampling order.

    Returns:
     *X2* :  the randomly sampled data data, provided as a numpy array where each column contains samples of a seperate variable (eg. u1,u2,...,un).

    """

    # Reference: Dißmann et al. 2013
    Ms = np.flipud(a)  # flip structure matrix
    P = np.flipud(p)  # flip parameter matrix
    C = np.flipud(c)  # flip copula matrix
    replace = {}  # dictionary for relabeling martix
    for i in range(int(max(np.unique(Ms)) + 1)):
        val = max(np.unique(Ms)) - i
        # Ms[k,k]
        replace.update({Ms[i, i]: val})  # relabel

    Ms = np.nan_to_num(Ms, nan=int(max(np.unique(Ms)) + 1))
    replace_func = np.vectorize(
        lambda x: replace.get(x, x)
    )  # Create a vectorized function for replacement
    M = replace_func(Ms)  # relabel
    M[M == np.max(Ms)] = np.nan
    Mm = M.copy()
    # max matrix
    for i in range(M.shape[0]):
        for k in range(M.shape[0]):
            if k == i:
                continue
            if i == 0:
                continue
            Mm[i, k] = max(Mm[i:, k])

    # Vdirect
    Vdir = np.empty((s, M.shape[0], M.shape[0]))
    Vdir[:] = np.nan
    # Vindirect
    Vindir = np.empty((s, M.shape[0], M.shape[0]))
    Vindir[:] = np.nan
    # Z2
    Z2 = np.empty((s, M.shape[0], M.shape[0]))
    Z2[:] = np.nan
    # Z1
    Z1 = np.empty((s, M.shape[0], M.shape[0]))
    Z2[:] = np.nan
    # combine random uniform data with conditional input
    U = np.hstack(
        (
            np.random.uniform(0, 1, (s, M.shape[0] - len(Xc))),
            np.flip(np.tile(Xc, (s, 1)).copy(), 1),
        )
    )
    Vdir[:, -1, :] = U.copy()
    X = np.flip(U.copy(), 1)
    n = M.shape[0] - 1

    # sampling algorithm
    for k in range(n)[::-1]:
        for i in range(k + 1, n + 1):
            if M[i, k] == Mm[i, k]:
                Z2[:, i, k] = Vdir[:, i, int(n - Mm[i, k])]
            else:
                Z2[:, i, k] = Vindir[:, i, int(n - Mm[i, k])]
            if k <= n - len(Xc):
                Vdir[:, n, k] = hfuncinverse(
                    int(C[i, k]), Z2[:, i, k], Vdir[:, n, k], P[i, k], un=2
                )
        X[:, int(n - k)] = Vdir[:, n, k]
        for i in range(k + 1, n + 1)[::-1]:
            Z1[:, i, k] = Vdir[:, i, k]
            Vdir[:, int(i - 1), k] = hfunc(
                int(C[i, k]), Z1[:, i, k], Z2[:, i, k], P[i, k], un=2
            )
            Vindir[:, int(i - 1), k] = hfunc(
                int(C[i, k]), Z1[:, i, k], Z2[:, i, k], P[i, k], un=1
            )

    # Put X in the original order of the data
    replacedf = pd.DataFrame(list(replace.items()), columns=["Original", "Replacement"])
    replacedf = replacedf.sort_values(by="Original")
    X2 = np.array([])
    for i in replacedf.Replacement:
        if len(X2) == 0:
            X2 = X[:, int(i)].reshape(len(X), 1)
        else:
            X2 = np.hstack((X2, X[:, int(i)].reshape(len(X), 1)))
    return X2


# %%sampling orders


def samplingorder(a):
    """
    Provides all the different sampling orders that are possible for the fitted vine-copula.

    Arguments:
        *a* : The vine tree structure provided as a triangular matrix, composed of integers. The integer refers to different variables depending on which column the variable was in u1, where the first column is 0 and the second column is 1, etc.

        *p* : Parameters of the bivariate copulae provided as a triangular matrix.

        *c* : The types of the bivariate copulae provided as a triangular matrix, composed of integers referring to the copulae with the best fit. eg. a 1 refers to the gaussian copula (see...refer to where this information would be)



    Returns:
     *sortingorder* :  A list of the different sampling orders available for the fitted vine-copula

    """

    dimen = a.shape[0]  # dimension of data.
    order = pd.DataFrame(
        columns=["node", "l", "r", "tree"]
    )  # dataframe for vine copula structure.
    s = 0
    for i in list(range(dimen - 1)):
        for k in list(range(dimen - 1 - i)):
            ak = a[:, k]  # edges
            akn = np.array([ak[-1 - k], ak[i]]).astype(int)
            if i == 0:
                single_row_values = {
                    "node": list(akn),
                    "l": akn[0],
                    "r": akn[1],
                    "tree": i,
                }
            else:
                single_row_values = {
                    "node": list(akn) + ["|"] + list((ak.astype(int)[:i])[::-1]),
                    "l": list(akn),
                    "r": list((ak.astype(int)[:i])[::-1]),
                    "tree": i,
                }

            order.loc[s] = single_row_values
            s = s + 1
    combinations = list(
        product([True, False], repeat=dimen - 1)
    )  # all diffferent sampling routes.
    sortingorder = []
    for q in combinations:
        a = np.empty((dimen, dimen))
        a[:] = np.nan
        order["used"] = 0
        for i in list(range(dimen - 1))[::-1]:
            k1 = sorted(
                np.array(
                    order[(order.tree == i) & (order["used"] == 0)].node.iloc[0][:2]
                ).astype(int),
                reverse=q[i],
            )
            order.loc[(order["tree"] == i) & (order["used"] == 0), "used"] = 1
            ii = dimen - 2 - i
            a[i : dimen - ii, ii] = k1
            s = k1[-1]
            for j in list(range(0, i))[::-1]:
                orde = order[(order.tree == j) & (order["used"] == 0)]
                for k in range(len(orde)):
                    arr = np.array(orde.node.iloc[k][:2]).astype(int)
                    if np.isin(s, arr) == True:
                        inde = orde.iloc[k].name
                        a[j, ii] = arr[arr != s][0]
                        order["used"][inde] = 1
        a[0, dimen - 1] = a[0, dimen - 2]
        sortingorder.append(list(np.diag(a[::-1])[::-1]))  # add unique sampling orders
    return sortingorder


# %%matrices of specific sampling order


def samplingmatrix(a, c, p, sorder):
    """
    Provides the triangular matrices for which the samples can be generated based on the specific sampling order.

    Arguments:
        *a* : The vine tree structure provided as a triangular matrix, composed of integers. The integer rffers to different variables depending on which column the variable was in u1, where the first column is 0 and the second column is 1, etc.

        *p* : Parameters of the bivariate copulae provided as a triangular matrix.

        *c* : The types of the bivariate copulae provided as a triangular matrix, composed of integers referring to the copulae with the best fit. eg. a 1 refers to the gaussian copula  (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).

        *sorder* :  A list containing the specific sampling order of interest

    Returns:
     *ai* : The vine tree structure, based on the sampling order, provided as a triangular matrix, composed of integers. The integer refers to different variables depending on which column the variable was in u1, where the first column is 0 and the second column is 1, etc.

     *pi* : Parameters of the bivariate copulae, based on the sampling order,  provided as a triangular matrix.

     *ci* : The types of the bivariate copulae, based on the sampling order,  provided as a triangular matrix, composed of integers referring to the copulae with the best fit. eg. a 1 refers to the gaussian copula  (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).

    """

    dimen = a.shape[0]  # dimension of data
    order = pd.DataFrame(
        columns=["node", "par", "cop", "l", "r", "tree"]
    )  # dataframe for vine copula structure
    s = 0
    for i in list(range(dimen - 1)):
        for k in list(range(dimen - 1 - i)):
            ak = a[:, k]
            akn = np.array([ak[-1 - k], ak[i]]).astype(int)
            if i == 0:
                single_row_values = {
                    "node": list(akn),
                    "par": p[i, k],
                    "cop": int(c[i, k]),
                    "l": akn[0],
                    "r": akn[1],
                    "tree": i,
                }
            else:
                single_row_values = {
                    "node": list(akn) + ["|"] + list((ak.astype(int)[:i])[::-1]),
                    "par": p[i, k],
                    "cop": int(c[i, k]),
                    "l": list(akn),
                    "r": list((ak.astype(int)[:i])[::-1]),
                    "tree": i,
                }

            order.loc[s] = single_row_values
            s = s + 1
    ai = np.empty((dimen, dimen))  # array for vine copula structure
    ai[:] = np.nan
    order["used"] = 0
    ci = np.empty((dimen, dimen))  # array for vine copula copulas
    ci[:] = np.nan
    pi = np.empty((dimen, dimen))  # array for parameters
    pi[:] = np.nan
    pi = pi.astype(object)
    for i in list(range(dimen - 1))[::-1]:
        if i == 0:
            k1 = order[(order.tree == i) & (order["used"] == 0)].node.iloc[0]
        else:
            k1 = order[(order.tree == i) & (order["used"] == 0)].l.iloc[0]

        if sorder[i + 1] == max(k1):
            k1 = sorted(k1, reverse=False)
        else:
            k1 = sorted(k1, reverse=True)
        ii = dimen - 2 - i
        ai[i : dimen - ii, ii] = k1
        ci[i, ii] = order[(order.tree == i) & (order["used"] == 0)].cop.iloc[0]
        pi[i, ii] = order[(order.tree == i) & (order["used"] == 0)].par.iloc[0]
        order.loc[(order["tree"] == i) & (order["used"] == 0), "used"] = 1
        s = k1[-1]
        for j in list(range(0, i))[::-1]:
            orde = order[(order.tree == j) & (order["used"] == 0)]
            for k in range(len(orde)):
                arr = np.array(orde.node.iloc[k][:2]).astype(int)
                if np.isin(s, arr) == True:
                    inde = orde.iloc[k].name
                    ai[j, ii] = arr[arr != s][0]
                    ci[j, ii] = order["cop"][inde]
                    pi[j, ii] = order["par"][inde]
                    order["used"][inde] = 1
    ai[0, dimen - 1] = ai[0, dimen - 2]
    return ai, pi, ci


# %%


def fit_conditionalvine(u1, vint, copsi, vine="R", condition=1, printing=True):
    """
    Fit a regular vine copula which allows for a conditional sample of a variable of interest.

    Arguments:
        *u1* :  the data, provided as a numpy array where each column contains a seperate variable (eg. u1,u2,...,un), which have already been transferred to standard uniform margins (0<= u <= 1)

        *vint* : the variables of interest, provided as an integere or list that refers to the variables column numbers in u1, where the first column is 0 and the second column is 1, etc.

        *copsi* : A list of integers referring to the copulae of interest for which the fit has to be evauluated in the vine copula. eg. a list of [1, 10] refers to the Gaussian and Frank copula  (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).

        *vine* : The type of vine copula that needs to be fit, either 'R', 'D', or 'C'

        *condition* : condition = 1 indicates that vint needs to be the conditionalized variables (at the end of the sampling order), condition = 2 indicates that vint need to be the conditioning variables (at the start of the sampling order)

        *printing*: True if the fitted copula should be printed and False if not
     Returns:
      *a* : The vine tree structure provided as a triangular matrix, composed of integers. The integer refers to different variables depending on which column the variable was in u1, where the first column is 0 and the second column is 1, etc.

      *p* : Parameters of the bivariate copulae provided as a triangular matrix.

      *c* : The types of the bivariate copulae provided as a triangular matrix, composed of integers referring to the copulae with the best fit. eg. a 1 refers to the gaussian copula  (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).

    """

    # Reference: Dißmann et al. 2013 adapted
    dimen = u1.shape[1]  # number of variables (number of columns)
    if condition == 1:  # if vint is the conditionalized variable
        X = vint
        try:  # if X is an integer put it in a list
            len(X)

        except:
            X = [X]
        Y = list(range(dimen))
        for i in X:
            Y.remove(i)  # Define Y
    elif condition == 2:  # if vint is the conditioning variable
        Y = vint
        try:  # if Y is an integer put it in a list
            len(Y)

        except:
            Y = [Y]
        X = list(range(dimen))
        for i in Y:
            X.remove(i)  # Define X
    v1 = []  # list for variable 1
    v2 = []  # list for  variable 2
    tauabs = []  # list for the absolute kendal tau between v1 and v2

    for i in range(dimen - 1):
        for j in range(i + 1, dimen):
            v1.append(int(i))  # add variable to v1
            v2.append(int(j))  # add variable to v2
            tauabs.append(
                abs(st.kendalltau(u1[:, i], u1[:, j])[0])
            )  # calculate the absolute kendall tau between v1 and v2 and add it to the taubs list

    order1 = pd.DataFrame(
        {"v1": v1, "v2": v2, "tauabs": tauabs}
    )  # put the v1, v2, and tauabs list into a dataframe for the first tree
    order1 = order1.sort_values(by="tauabs", ascending=False).reset_index(
        drop=True
    )  # sort this dataframe from highest to lowest tauabs
    Y2 = Y.copy()  # create copies of Y
    X2 = X.copy()  # create copies of X
    if vine == "R":
        # first find the pairs with the highest kendall tau using variables in Y
        if len(Y) > 1:
            for j in range(len(Y) - 1):
                # define all rows in order1 where only variables in Y are included
                for z in range(j + 1, len(Y)):
                    if z == 1:
                        l = ((order1.v1 == Y[j]) & (order1.v2 == Y[z])) | (
                            (order1.v1 == Y[z]) & (order1.v2 == Y[j])
                        )
                    else:
                        l = (
                            l
                            | ((order1.v1 == Y[j]) & (order1.v2 == Y[z]))
                            | ((order1.v1 == Y[z]) & (order1.v2 == Y[j]))
                        )
            inde = []  # list to put used rows of order1 in
            l = np.where(l == True)[0]
            # loop through the rows with variables in Y
            for k in l:
                if k == l[0]:  # add first one to the tree
                    order2 = order1.loc[k].to_frame().T
                    inde.append(k)  # add used rows to inde
                if k in inde:  # check if row has not been added before
                    continue

                lst = (
                    list(order2.v2[order2.v1 == order1.v2[k]])
                    + list(order2.v1[order2.v2 == order1.v2[k]])
                    + list(order2.v2[order2.v1 == order1.v1[k]])
                    + list(order2.v1[order2.v2 == order1.v1[k]])
                )
                l1 = list(order2.v2[order2.v1 == order1.v2[k]]) + list(
                    order2.v1[order2.v2 == order1.v2[k]]
                )
                l2 = list(order2.v2[order2.v1 == order1.v1[k]]) + list(
                    order2.v1[order2.v2 == order1.v1[k]]
                )
                lk = l1.copy()
                while len(lk) > 0:
                    lk2 = lk.copy()
                    lk = []
                    for j in lk2:
                        ln = list(order2.v2[order2.v1 == j]) + list(
                            order2.v1[order2.v2 == j]
                        )
                        try:
                            ln.remove(order1.v2[k])
                        except:
                            pass

                        for s in l1:
                            try:
                                ln.remove(s)
                            except:
                                pass
                        l1 = l1 + ln
                        lk = lk + ln

                lk = l2.copy()
                while len(lk) > 0:
                    lk2 = lk.copy()
                    lk = []
                    for j in lk2:
                        ln = list(order2.v2[order2.v1 == j]) + list(
                            order2.v1[order2.v2 == j]
                        )
                        try:
                            ln.remove(order1.v1[k])
                        except:
                            pass

                        for s in l2:
                            try:
                                ln.remove(s)
                            except:
                                pass
                        l2 = l2 + ln
                        lk = lk + ln
                skip = False
                for val in l1:
                    if val in l2:
                        skip = True
                        break
                if skip == True:
                    continue
                if len(lst) == len(set(lst)):
                    order2 = pd.concat(
                        [order2, order1.loc[k].to_frame().T], ignore_index=True
                    )
                    inde.append(k)
                    if (
                        len(order2) == len(Y) - 1
                    ):  # end loop when desired number of rows (number of variables in Y - 1) has been met
                        break

        else:  # if there is only one value in Y
            for z in X:
                # define all rows in order1 where variables in X are connected to variabels in Y
                if z == X[0]:
                    l = ((order1.v1 == Y[0]) & (order1.v2 == z)) | (
                        (order1.v1 == z) & (order1.v2 == Y[0])
                    )
                else:
                    l = (
                        l
                        | ((order1.v1 == Y[0]) & (order1.v2 == z))
                        | ((order1.v1 == z) & (order1.v2 == Y[0]))
                    )
            inde = []  # list to put used rows of order1 in
            l = np.where(l == True)[0]
            order2 = order1.loc[l[0]].to_frame().T  # add the first row to order 2
            inde.append(l[0])  # add used rows to inde
            # remove used variables in X from X and add them to Y
            if order2.v1.iloc[-1] in X2:
                Y2.append(int(order2.v1.iloc[-1]))
                X2.remove(int(order2.v1.iloc[-1]))
            else:
                Y2.append(int(order2.v2.iloc[-1]))
                X2.remove(int(order2.v2.iloc[-1]))

        while len(order2) < (dimen - 1):
            # Here we ensure that the last remaining X variable will be connected either to another original X variable, or a variable that is connected to an original X value
            if len(order2) == dimen - 2 and len(X) > 1:
                X3 = X.copy()
                X3.remove(X2[0])
                Y3 = []
                for i in X3:
                    Y3 = (
                        Y3
                        + list(order2.v1[order2.v2 == i])
                        + list(order2.v2[order2.v1 == i])
                        + [i]
                    )

                Y3 = list(np.unique(Y3))
                for j in Y3:
                    for z in X2:
                        if j == Y3[0] and z == X2[0]:
                            l = ((order1.v1 == j) & (order1.v2 == z)) | (
                                (order1.v1 == z) & (order1.v2 == j)
                            )
                        else:
                            l = (
                                l
                                | ((order1.v1 == j) & (order1.v2 == z))
                                | ((order1.v1 == z) & (order1.v2 == j))
                            )

                l = np.where(l == True)[0]

            else:
                # define all rows in order1 where variables in X are connected to variabels in Y
                for j in Y2:
                    for z in X2:
                        if j == Y2[0] and z == X2[0]:
                            l = ((order1.v1 == j) & (order1.v2 == z)) | (
                                (order1.v1 == z) & (order1.v2 == j)
                            )
                        else:
                            l = (
                                l
                                | ((order1.v1 == j) & (order1.v2 == z))
                                | ((order1.v1 == z) & (order1.v2 == j))
                            )

                l = np.where(l == True)[0]
            for k in l:
                lst = (
                    list(order2.v2[order2.v1 == order1.v2[k]])
                    + list(order2.v1[order2.v2 == order1.v2[k]])
                    + list(order2.v2[order2.v1 == order1.v1[k]])
                    + list(order2.v1[order2.v2 == order1.v1[k]])
                )
                l1 = list(order2.v2[order2.v1 == order1.v2[k]]) + list(
                    order2.v1[order2.v2 == order1.v2[k]]
                )
                l2 = list(order2.v2[order2.v1 == order1.v1[k]]) + list(
                    order2.v1[order2.v2 == order1.v1[k]]
                )
                lk = l1.copy()
                while len(lk) > 0:
                    lk2 = lk.copy()
                    lk = []
                    for j in lk2:
                        ln = list(order2.v2[order2.v1 == j]) + list(
                            order2.v1[order2.v2 == j]
                        )
                        try:
                            ln.remove(order1.v2[k])
                        except:
                            pass

                        for s in l1:
                            try:
                                ln.remove(s)
                            except:
                                pass
                        l1 = l1 + ln
                        lk = lk + ln

                lk = l2.copy()
                while len(lk) > 0:
                    lk2 = lk.copy()
                    lk = []
                    for j in lk2:
                        ln = list(order2.v2[order2.v1 == j]) + list(
                            order2.v1[order2.v2 == j]
                        )
                        try:
                            ln.remove(order1.v1[k])
                        except:
                            pass

                        for s in l2:
                            try:
                                ln.remove(s)
                            except:
                                pass
                        l2 = l2 + ln
                        lk = lk + ln
                skip = False
                for val in l1:
                    if val in l2:
                        skip = True
                        break
                if skip == True:
                    continue
                # if (len(l1) > 1 and len(l2) >0) or (len(l1) > 0 and len(l2) >1):
                # continue
                if order1.v1[k] in X and order1.v2[k] in X:
                    skip = False
                    if order1.v2[k] in Y2:

                        numapear = sum(
                            (
                                (order2["v1"].apply(lambda x: order1.v2[k] == x))
                                | (order2["v2"].apply(lambda x: order1.v2[k] == x))
                            )
                        )
                        for i in Y:
                            if numapear == sum(
                                (
                                    (order2["v1"].apply(lambda x: i == x))
                                    | (order2["v2"].apply(lambda x: i == x))
                                )
                            ):
                                skip = True
                                break
                    elif order1.v1[k] in Y2:

                        numapear = sum(
                            (
                                (order2["v1"].apply(lambda x: order1.v1[k] == x))
                                | (order2["v2"].apply(lambda x: order1.v1[k] == x))
                            )
                        )
                        for i in Y:
                            if numapear == sum(
                                (
                                    (order2["v1"].apply(lambda x: i == x))
                                    | (order2["v2"].apply(lambda x: i == x))
                                )
                            ):
                                skip = True
                                break
                    if skip == True:
                        continue
                if len(lst) == len(set(lst)):
                    order2 = pd.concat(
                        [order2, order1.loc[k].to_frame().T], ignore_index=True
                    )
                    inde.append(k)  # add used rows to inde
                    # remove used variables in X from X and add them to Y
                    if order2.v1.iloc[-1] in X2:
                        Y2.append(int(order2.v1.iloc[-1]))
                        X2.remove(int(order2.v1.iloc[-1]))
                    else:
                        Y2.append(int(order2.v2.iloc[-1]))
                        X2.remove(int(order2.v2.iloc[-1]))

                    break
    elif vine == "D":
        if len(Y) > 1:
            inde = []  # list to put used rows of order1 in
            for j in range(len(Y) - 1):
                # define all rows in order1 where only variables in Y are included
                for z in range(j + 1, len(Y)):
                    if z == 1:
                        l = ((order1.v1 == Y[j]) & (order1.v2 == Y[z])) | (
                            (order1.v1 == Y[z]) & (order1.v2 == Y[j])
                        )
                    else:
                        l = (
                            l
                            | ((order1.v1 == Y[j]) & (order1.v2 == Y[z]))
                            | ((order1.v1 == Y[z]) & (order1.v2 == Y[j]))
                        )
            l2 = list(order1[~l].index)
            l = np.where(l == True)[0]
            order2 = order1.loc[l[0]].to_frame().T.reset_index(drop=True)
            vi = order2.v1[0]  # select the first variable included in this row
            vj = order2.v2[0]  # select the second variable included in this row
            while len(order2) < len(Y) - 1:
                for i in l:
                    if (
                        order1.v1[i] == vj
                        or order1.v2[i] == vi
                        or order1.v1[i] == vi
                        or order1.v2[i] == vj
                    ):  # find the rows which include at least one of the two variables from the previous edge
                        if (
                            order1.v1[i] in inde or order1.v2[i] in inde
                        ):  # check if variables have already been connected to other variables twice
                            continue
                        if (order1.v1[i] == vi and order1.v2[i] == vj) or (
                            order1.v1[i] == vj and order1.v2[i] == vi
                        ):  # skip rows that have already been used
                            continue

                        else:
                            order2 = pd.concat(
                                [order2, order1.loc[i].to_frame().T], ignore_index=True
                            )  # if conditions are met, add this to the dataframe of the first tree
                            if (
                                order1.v1[i] == vj or order1.v2[i] == vj
                            ):  # check which value has been used twice already
                                inde.append(vj)  # add this value to inde
                                if order1.v1[i] == vj:
                                    vj = order1.v2[i]  # select the new edge to connect to
                                else:
                                    vj = order1.v1[i]  # select the new edge to connect to
                                break
                            elif (
                                order1.v1[i] == vi or order1.v2[i] == vi
                            ):  # check which value has been used twice already
                                inde.append(vi)  # add this value to inde
                                if order1.v1[i] == vi:
                                    vi = order1.v2[i]  # select the new edge to connect to
                                else:
                                    vi = order1.v1[i]  # select the new edge to connect to
                                break
            for i in l2:
                if (
                    order1.v1[i] == vj
                    or order1.v2[i] == vi
                    or order1.v1[i] == vi
                    or order1.v2[i] == vj
                ):  # find the rows which include at least one of the two variables from the previous edge
                    if (
                        order1.v1[i] in inde or order1.v2[i] in inde
                    ):  # check if variables have already been connected to other variables twice
                        continue
                    if (order1.v1[i] == vi and order1.v2[i] == vj) or (
                        order1.v1[i] == vj and order1.v2[i] == vi
                    ):  # skip rows that have already been used
                        continue

                    else:
                        order2 = pd.concat(
                            [order2, order1.loc[i].to_frame().T], ignore_index=True
                        )  # if conditions are met, add this to the dataframe of the first tree
                        if (
                            order1.v1[i] == vj or order1.v2[i] == vj
                        ):  # check which value has been used twice already
                            inde.append(vj)  # add this value to inde
                            if order1.v1[i] == vj:
                                vj = order1.v2[i]  # select the new edge to connect to
                            else:
                                vj = order1.v1[i]  # select the new edge to connect to
                        elif (
                            order1.v1[i] == vi or order1.v2[i] == vi
                        ):  # check which value has been used twice already
                            inde.append(vi)  # add this value to inde
                            if order1.v1[i] == vi:
                                vi = order1.v2[i]  # select the new edge to connect to
                            else:
                                vi = order1.v1[i]  # select the new edge to connect to
        else:
            inde = []  # lsit to put used variables in
            for j in range(dimen - 1):
                if j == 0:
                    order2 = order1.head(
                        1
                    )  # loop through all the rows to include those pairs with the highest ktau first
                    vi = order2.v1[0]  # select the first variable included in this row
                    vj = order2.v2[0]  # select the second variable included in this row
                else:
                    for i in range(len(order1)):
                        if (
                            order1.v1[i] == vj
                            or order1.v2[i] == vi
                            or order1.v1[i] == vi
                            or order1.v2[i] == vj
                        ):  # find the rows which include at least one of the two variables from the previous edge
                            if (
                                order1.v1[i] in inde or order1.v2[i] in inde
                            ):  # check if variables have already been connected to other variables twice
                                continue
                            if (order1.v1[i] == vi and order1.v2[i] == vj) or (
                                order1.v1[i] == vj and order1.v2[i] == vi
                            ):  # skip rows that have already been used
                                continue
                            else:
                                order2 = pd.concat(
                                    [order2, order1.loc[i].to_frame().T],
                                    ignore_index=True,
                                )  # if conditions are met, add this to the dataframe of the first tree
                                if (
                                    order1.v1[i] == vj or order1.v2[i] == vj
                                ):  # check which value has been used twice already
                                    inde.append(vj)  # add this value to inde
                                    if order1.v1[i] == vj:
                                        vj = order1.v2[
                                            i
                                        ]  # select the new edge to connect to
                                    else:
                                        vj = order1.v1[
                                            i
                                        ]  # select the new edge to connect to
                                elif (
                                    order1.v1[i] == vi or order1.v2[i] == vi
                                ):  # check which value has been used twice already
                                    inde.append(vi)  # add this value to inde
                                    if order1.v1[i] == vi:
                                        vi = order1.v2[
                                            i
                                        ]  # select the new edge to connect to
                                    else:
                                        vi = order1.v1[
                                            i
                                        ]  # select the new edge to connect to
    elif vine == "C":
        taus = []  # list to put the sum of all tauabs in for a specific variable
        for i in Y:
            taus.append(
                sum(order1[(order1.v1 == i) | (order1.v2 == i)].tauabs)
            )  # calculate the sum of all taus for a specific variable
        i = Y[
            np.where(np.array(taus) == max(taus))[0][0]
        ]  # find where the sum of the taus is the highest to find the vairable to place in the center of the first tree
        order2 = order1[(order1.v1 == i) | (order1.v2 == i)].reset_index(
            drop=True
        )  # select the rows that include this variable
        order2["both_in_Y"] = order2.apply(
            lambda row: (row["v1"] in Y) and (row["v2"] in Y), axis=1
        )  # find the columns with values in Y
        order2 = order2.sort_values(by="both_in_Y", ascending=False).reset_index(
            drop=True
        )  # Sort the DataFrame by the 'both_in_Y' column
        order2 = order2.drop(columns=["both_in_Y"])  # delete this column

    order1 = order2  # make order1 == the first tree
    del order2
    rhos = []  # list for the rhos
    node = []  # list for the nodes
    cops = []  # list for the copulas
    v1_1 = []  # list for the 1st nodes
    v2_1 = []  # list for the 2nd nodes
    aics = []  # list for the AIC's
    for i in range(len(order1)):
        v1i = int(order1.v1[i])  # first node
        v2i = int(order1.v2[i])  # second node
        u3 = np.vstack(
            (u1[:, v1i], u1[:, v2i])
        ).T  # stacking the combination to fit copula to
        cop, rho, aic = bestcop(copsi, u3)  # fit the best copula
        aics.append(aic)  # add AIC to aics
        rhos.append(rho)  # add parameters to rhos
        cops.append(cop)  # add copula to cops
        node.append([v1i, v2i])  # create final node
        v1_1.append(u1[:, v1i])  # add array of first node
        v2_1.append(u1[:, v2i])  # add array of second node
    v1_1 = np.array(v1_1).T  # v1
    v2_1 = np.array(v2_1).T  # v2

    # add information to dataframe of the first tree
    order1["rhos"] = rhos
    order1["node"] = node
    order1["tree"] = 0
    order1["cop"] = cops
    order1["AIC"] = aics

    # set up variables for the second tree
    v1 = []
    v2 = []
    ktau = []
    rhos = []
    node = []
    cops = []
    v1_k = []
    v2_k = []
    aics = []
    for i in range(
        len(order1) - 1
    ):  # loop through the first tree to identify all possible combination of nodes
        v1i = int(order1.v1[i])  # parent node 1 from nodei in tree
        v2i = int(order1.v2[i])  # parent node 2 from nodei in tree
        copi = int(order1.cop[i])  # copula of nodei
        pari = order1.rhos[i]  # parameters of nodei
        for j in (
            np.where(
                np.array([item == v1i for item in list(order1.v1[i + 1 :])])
                | np.array([item == v1i for item in list(order1.v2[i + 1 :])])
                | np.array([item == v2i for item in list(order1.v1[i + 1 :])])
                | np.array([item == v2i for item in list(order1.v2[i + 1 :])])
            )[0]
            + i
            + 1
        ):  # see if possible connection between nodei and nodej
            v1j = int(order1.v1[j])  # parent node 1 from nodej in tree
            v2j = int(order1.v2[j])  # parent node 2 from nodej in tree
            copj = int(order1.cop[j])  # copula of nodei
            parj = order1.rhos[j]  # parameters of nodei
            v1.append(order1.node[i])  # parent node for next tree
            v2.append(order1.node[j])  # parent node for next tree
            lst = order1.node[i] + order1.node[j]  # list of all variables in node
            s = max(
                set(lst), key=lst.count
            )  # variable that is common in both parent nodes
            # parent node values
            ui1 = v1_1[:, i]
            ui2 = v2_1[:, i]
            uj1 = v1_1[:, j]
            uj2 = v2_1[:, j]
            # defining which parent node the conditional CDF needs to be based on
            if v1i == s:
                uni = 1
                vi1 = v2i
            else:
                uni = 2
                vi1 = v1i
            if v1j == s:
                unj = 1
                vj1 = v2j
            else:
                unj = 2
                vj1 = v1j
            # calculate the conditional CDF
            v1igs = hfunc(copi, ui1, ui2, pari, un=uni)
            v2jgs = hfunc(copj, uj1, uj2, parj, un=unj)
            ktau.append(
                abs(st.kendalltau(v1igs, v2jgs)[0])
            )  # calculate the absolute kendall tau between v1igs and v2jgs, add it to the ktau list
            # add the parent node values
            v1_k.append(v1igs)
            v2_k.append(v2jgs)
            # format the final node
            node.append([vi1, vj1, "g", s])

    k = 2  # second tree

    orderk = pd.DataFrame(
        {"v1": v1, "v2": v2, "tauabs": ktau, "node": node}
    )  # put the v1, v2, tauabs, and the node in list into a dataframe for tree k

    if vine == "R":
        if len(orderk) > dimen - k:
            sorder2 = order1.node  # get all the unique parent nodes
            # divide the unique nodes in Y and X
            if len(Y) - 1 > 0:
                Y2 = list(sorder2[: len(Y) - 1].reset_index(drop=True))
                X2 = list(sorder2[len(Y) - 1 :].reset_index(drop=True))
            else:
                Y2 = list(sorder2[:1].reset_index(drop=True))
                X2 = list(sorder2[1:].reset_index(drop=True))
            orderk = orderk.sort_values(
                by="tauabs", ascending=False
            )  # sort this dataframe from highest to lowest tauabs
            indexi = list(
                orderk.index
            )  # create a list the original order of the dataframe prior to sorting
            orderk = orderk.reset_index(drop=True)  # reset the index
            inde = []  # list to put used rows of orderk i
            inde2 = (
                []
            )  #  list to put used rows of orderk in based on their oroginal position in the dataframe
            Y3 = Y2.copy()
            X3 = X2.copy()
            if len(Y2) == 1:  # if there is only one node in Y
                # define all rows in orderk where the nodes in X are connected to nodes in Y
                for j in Y2:
                    for z in X2:
                        if j == Y2[0] and z == X2[0]:
                            l = (
                                (orderk["v1"].apply(lambda x: j == x))
                                & (orderk["v2"].apply(lambda x: z == x))
                            ) | (
                                (orderk["v1"].apply(lambda x: z == x))
                                & (orderk["v2"].apply(lambda x: j == x))
                            )
                        else:
                            l = (
                                l
                                | (
                                    (orderk["v1"].apply(lambda x: j == x))
                                    & (orderk["v2"].apply(lambda x: z == x))
                                )
                                | (
                                    (orderk["v1"].apply(lambda x: z == x))
                                    & (orderk["v2"].apply(lambda x: j == x))
                                )
                            )

                l = np.where(l == True)[0]
                order = orderk.loc[l[0]].to_frame().T
                inde.append(l[0])
                inde2.append(indexi[l[0]])  # add used rows to inde (original row value)
                # remove used nodes in X from X and add them to Y
                if order.v1.iloc[-1] in X3:

                    X3.remove(order.v1.iloc[-1])
                    Y3.append(order.v1.iloc[-1])
                else:
                    X3.remove(order.v2.iloc[-1])
                    Y3.append(order.v2.iloc[-1])
            else:
                # first find the pairs with the highest kendall tau using nodes in Y
                for j in range(len(Y2) - 1):
                    # define all rows in orderk where only nodes in Y are included
                    for z in range(j + 1, len(Y2)):
                        if z == 1:
                            l = (
                                (orderk["v1"].apply(lambda x: Y2[j] == x))
                                & (orderk["v2"].apply(lambda x: Y2[z] == x))
                            ) | (
                                (orderk["v1"].apply(lambda x: Y2[z] == x))
                                & (orderk["v2"].apply(lambda x: Y2[j] == x))
                            )
                        else:
                            l = (
                                l
                                | (
                                    (orderk["v1"].apply(lambda x: Y2[j] == x))
                                    & (orderk["v2"].apply(lambda x: Y2[z] == x))
                                )
                                | (
                                    (orderk["v1"].apply(lambda x: Y2[z] == x))
                                    & (orderk["v2"].apply(lambda x: Y2[j] == x))
                                )
                            )

                l = np.where(l == True)[0]
                # loop through the rows with nodes in Y
                for k in l:
                    if k == l[0]:  # add first one to the tree
                        order = orderk.loc[k].to_frame().T
                        inde.append(k)
                        inde2.append(
                            indexi[k]
                        )  # add used rows to inde (original row value)
                    if k in inde:  # check if row has not been added before
                        continue
                    lst = (
                        list(
                            order.v2.astype(str)[
                                order.v1.astype(str) == str(orderk.v1[k])
                            ]
                        )
                        + list(
                            order.v1.astype(str)[
                                order.v2.astype(str) == str(orderk.v1[k])
                            ]
                        )
                        + list(
                            order.v2.astype(str)[
                                order.v1.astype(str) == str(orderk.v2[k])
                            ]
                        )
                        + list(
                            order.v1.astype(str)[
                                order.v2.astype(str) == str(orderk.v2[k])
                            ]
                        )
                    )
                    l1 = list(
                        order.v2.astype(str)[order.v1.astype(str) == str(orderk.v1[k])]
                    ) + list(
                        order.v1.astype(str)[order.v2.astype(str) == str(orderk.v1[k])]
                    )
                    lk = l1.copy()
                    while len(lk) > 0:
                        lk2 = lk.copy()
                        lk = []
                        for j in lk2:
                            ln = list(
                                order.v2.astype(str)[order.v1.astype(str) == j]
                            ) + list(order.v1.astype(str)[order.v2.astype(str) == j])
                            try:
                                ln.remove(str(orderk.v1[k]))
                            except:
                                pass

                            for s in l1:
                                try:
                                    ln.remove(s)
                                except:
                                    pass
                            l1 = l1 + ln
                            lk = lk + ln

                    l2 = list(
                        order.v2.astype(str)[order.v1.astype(str) == str(orderk.v2[k])]
                    ) + list(
                        order.v1.astype(str)[order.v2.astype(str) == str(orderk.v2[k])]
                    )
                    lk = l2.copy()
                    while len(lk) > 0:
                        lk2 = lk.copy()
                        lk = []
                        for j in lk2:
                            ln = list(
                                order.v2.astype(str)[order.v1.astype(str) == j]
                            ) + list(order.v1.astype(str)[order.v2.astype(str) == j])
                            try:
                                ln.remove(str(orderk.v2[k]))
                            except:
                                pass

                            for s in l2:
                                try:
                                    ln.remove(s)
                                except:
                                    pass
                            l2 = l2 + ln
                            lk = lk + ln
                    skip = False
                    for val in l1:
                        if val in l2:
                            skip = True
                            break
                    if skip == True:
                        continue
                    # if (len(l1) > 1 and len(l2) >0) or (len(l1) > 0 and len(l2) >1):  # ensure that there are no groups of variables that all have a connection with one another
                    # continue
                    if len(lst) == len(set(lst)):
                        order = pd.concat(
                            [order, orderk.loc[k].to_frame().T], ignore_index=True
                        )
                        inde.append(k)
                        inde2.append(indexi[k])
                        if (
                            len(order) == len(Y2) - 1
                        ):  # end loop when desired number of rows (number of variables in Y - 1) has been met
                            break

            while len(order) < (dimen - 2):
                # define all rows in orderk where nodes in X are connected to nodes in Y
                for j in Y3:
                    for z in X3:
                        if j == Y3[0] and z == X3[0]:
                            l = (
                                (orderk["v1"].apply(lambda x: j == x))
                                & (orderk["v2"].apply(lambda x: z == x))
                            ) | (
                                (orderk["v1"].apply(lambda x: z == x))
                                & (orderk["v2"].apply(lambda x: j == x))
                            )
                        else:
                            l = (
                                l
                                | (
                                    (orderk["v1"].apply(lambda x: j == x))
                                    & (orderk["v2"].apply(lambda x: z == x))
                                )
                                | (
                                    (orderk["v1"].apply(lambda x: z == x))
                                    & (orderk["v2"].apply(lambda x: j == x))
                                )
                            )

                l = np.where(l == True)[0]
                for k in l:
                    if k in inde:
                        continue
                    lst = (
                        list(
                            order.v2.astype(str)[
                                order.v1.astype(str) == str(orderk.v1[k])
                            ]
                        )
                        + list(
                            order.v1.astype(str)[
                                order.v2.astype(str) == str(orderk.v1[k])
                            ]
                        )
                        + list(
                            order.v2.astype(str)[
                                order.v1.astype(str) == str(orderk.v2[k])
                            ]
                        )
                        + list(
                            order.v1.astype(str)[
                                order.v2.astype(str) == str(orderk.v2[k])
                            ]
                        )
                    )
                    l1 = list(
                        order.v2.astype(str)[order.v1.astype(str) == str(orderk.v1[k])]
                    ) + list(
                        order.v1.astype(str)[order.v2.astype(str) == str(orderk.v1[k])]
                    )
                    lk = l1.copy()
                    while len(lk) > 0:
                        lk2 = lk.copy()
                        lk = []
                        for j in lk2:
                            ln = list(
                                order.v2.astype(str)[order.v1.astype(str) == j]
                            ) + list(order.v1.astype(str)[order.v2.astype(str) == j])
                            try:
                                ln.remove(str(orderk.v1[k]))
                            except:
                                pass

                            for s in l1:
                                try:
                                    ln.remove(s)
                                except:
                                    pass
                            l1 = l1 + ln
                            lk = lk + ln

                    l2 = list(
                        order.v2.astype(str)[order.v1.astype(str) == str(orderk.v2[k])]
                    ) + list(
                        order.v1.astype(str)[order.v2.astype(str) == str(orderk.v2[k])]
                    )
                    lk = l2.copy()
                    while len(lk) > 0:
                        lk2 = lk.copy()
                        lk = []
                        for j in lk2:
                            ln = list(
                                order.v2.astype(str)[order.v1.astype(str) == j]
                            ) + list(order.v1.astype(str)[order.v2.astype(str) == j])
                            try:
                                ln.remove(str(orderk.v2[k]))
                            except:
                                pass

                            for s in l2:
                                try:
                                    ln.remove(s)
                                except:
                                    pass
                            l2 = l2 + ln
                            lk = lk + ln
                    skip = False
                    for val in l1:
                        if val in l2:
                            skip = True
                            break
                    if skip == True:
                        continue
                    # if (len(l1) > 1 and len(l2) >0) or (len(l1) > 0 and len(l2) >1):
                    # continue
                    if len(lst) == len(set(lst)):
                        order = pd.concat(
                            [order, orderk.loc[k].to_frame().T], ignore_index=True
                        )
                        inde.append(k)
                        inde2.append(indexi[k])
                        # remove used variables in X from X and add them to Y
                        if order.v1.iloc[-1] in X3:

                            X3.remove(order.v1.iloc[-1])
                            Y3.append(order.v1.iloc[-1])
                            # X2.remove(int(order2.v1.iloc[-1]))
                        else:
                            X3.remove(order.v2.iloc[-1])
                            Y3.append(order.v2.iloc[-1])

                        break
            orderk = order
            k = 2
            # orderk = orderk.sort_values(by='tauabs', ascending=False)
            v1_k = np.array([v1_k[ind] for ind in inde]).T  # sort array of first node
            v2_k = np.array([v2_k[ind] for ind in inde]).T  # sort array of second node
            orderk = orderk.reset_index(drop=True)
            orderk["tree"] = k - 1
            v1_2 = v1_k.copy()
            v2_2 = v2_k.copy()
            order2 = orderk.copy()
        else:
            orderk = orderk.sort_values(
                by="tauabs", ascending=False
            )  # sort this dataframe from highest to lowest tauabs
            v1_k = np.array(
                [v1_k[ind] for ind in orderk.index]
            ).T  # sort array of first node
            v2_k = np.array(
                [v2_k[ind] for ind in orderk.index]
            ).T  # sort array of second node
            orderk = orderk.reset_index(drop=True)
            orderk["tree"] = k - 1
            v1_2 = v1_k.copy()
            v2_2 = v2_k.copy()
            order2 = orderk.copy()

    elif vine == "D":
        orderk = orderk.sort_values(
            by="tauabs", ascending=False
        )  # sort this dataframe from highest to lowest tauabs
        v1_k = np.array(
            [v1_k[ind] for ind in orderk.index]
        ).T  # sort array of first node
        v2_k = np.array(
            [v2_k[ind] for ind in orderk.index]
        ).T  # sort array of second node
        orderk = orderk.reset_index(drop=True)
        orderk["tree"] = k - 1
        v1_2 = v1_k.copy()
        v2_2 = v2_k.copy()
        order2 = orderk.copy()

    elif vine == "C":
        if len(orderk) > dimen - k:
            subnodes = order1[
                ["v1", "v2"]
            ].values.tolist()  # see all the unique parent nodes
            taus = []  # list to put the sum of all tauabs in for a specific nodes
            for i in range(len(Y) - 1):
                orderksub = orderk[
                    (orderk["v1"].apply(lambda x: x == subnodes[i]))
                    | (orderk["v2"].apply(lambda x: x == subnodes[i]))
                ]
                taus.append(
                    sum(orderksub.tauabs)
                )  # calculate the sum of all taus for a specific variable
            i = np.where(np.array(taus) == max(taus))[0][
                0
            ]  # find where the sum of the taus is the highest to find the vairable to place in the center of the first tree
            orderk = orderk[
                (orderk["v1"].apply(lambda x: x == subnodes[i]))
                | (orderk["v2"].apply(lambda x: x == subnodes[i]))
            ]  # select rows where this node is included
            inde = list(orderk.index)
            v1_k = np.array([v1_k[ind] for ind in inde]).T  # sort array of first node
            v2_k = np.array([v2_k[ind] for ind in inde]).T  # sort array of second node
            orderk = orderk.reset_index(drop=True)
            orderk["tree"] = k - 1
            v1_2 = v1_k.copy()
            v2_2 = v2_k.copy()
            order2 = orderk.copy()
        else:
            orderk = orderk.sort_values(by="tauabs", ascending=False)
            v1_k = np.array([v1_k[ind] for ind in orderk.index]).T
            v2_k = np.array([v2_k[ind] for ind in orderk.index]).T
            orderk = orderk.reset_index(drop=True)
            orderk["tree"] = k - 1
            v1_2 = v1_k.copy()
            v2_2 = v2_k.copy()
            order2 = orderk.copy()

    for i in range(len(order2)):
        u3 = np.vstack(
            (v1_2[:, i], v2_2[:, i])
        ).T  # stacking the combination to fit copula to
        cop, rho, aic = bestcop(copsi, u3)  # fit the best copula
        aics.append(aic)  # add AIC to aics
        rhos.append(rho)  # add parameters to rhos
        cops.append(cop)  # add copula to cops

    # add information to dataframe of tree k
    order2["rhos"] = rhos
    order2["cop"] = cops
    order2["AIC"] = aics

    # loop through remaining trees if there are more than 3 variables
    if dimen > 3:
        for k in range(3, dimen):
            order = locals()[
                "order" + str(k - 1)
            ].copy()  # select the struture of the previous tree
            v1s = locals()[
                "v1_" + str(k - 1)
            ].copy()  # select the first nodes of the previous tree
            v2s = locals()[
                "v2_" + str(k - 1)
            ].copy()  # select the second nodes of the previous tree
            # create lists for the variables
            v1_k = []
            v2_k = []
            v1 = []
            v2 = []
            ktau = []
            rhos = []
            cops = []
            node = []
            lk = []
            aics = []
            rk = []
            for i in range(len(order) - 1):
                v1i = order.v1[i].copy()  # parent node 1 from nodei in tree
                v2i = order.v2[i].copy()  # parent node 2 from nodei in tree
                copi = int(order.cop[i])  # copula of nodei
                pari = order.rhos[i]  # parameters of nodei
                for j in (
                    np.where(
                        np.array([item == v1i for item in list(order.v1[i + 1 :])])
                        | np.array([item == v1i for item in list(order.v2[i + 1 :])])
                        | np.array([item == v2i for item in list(order.v1[i + 1 :])])
                        | np.array([item == v2i for item in list(order.v2[i + 1 :])])
                    )[0]
                    + i
                    + 1
                ):  # see if possible connection between nodei and nodej
                    v1i = order.v1[i].copy()  # parent node 1 from nodei in tree
                    v2i = order.v2[i].copy()  # parent node 2 from nodei in tree
                    copj = int(order.cop[j])  # copula of nodei
                    parj = order.rhos[j]  # parameters of nodei
                    nodei = order.node[i]  # nodei
                    nodej = order.node[j]  # nodej
                    v1j = order.v1[j].copy()  # parent node 1 from nodej in tree
                    v2j = order.v2[j].copy()  # parent node 2 from nodej in tree
                    v1.append(nodei)  # new parent nodes 1
                    v2.append(nodej)  # new parent nodes 2
                    n = 2
                    ri = nodei[n + 1 :]  # select the values on the right side of node i
                    rj = nodej[n + 1 :]  # select the values on the right side of node i

                    if "g" in v1j:
                        v1j.remove("g")
                        v2j.remove("g")
                        v1i.remove("g")
                        v2i.remove("g")

                    # define left and right side of the node in tree k
                    if rj == ri:
                        if len(v1j) == 2:
                            lst = nodei[:n] + nodej[:n]
                            r3 = [max(set(lst), key=lst.count)]
                            li = [value for value in nodei[:n] if value not in r3]
                            lj = [value for value in nodej[:n] if value not in r3]
                            r = list(np.unique(ri + r3))
                        else:
                            lst = v1i[:n] + v2i[:n] + v1j[:n] + v2j[:n]
                            for s in ri:
                                lst = [x for x in lst if x != s]
                            r3 = [max(set(lst), key=lst.count)]
                            li = [value for value in nodei[:n] if value not in r3]
                            lj = [value for value in nodej[:n] if value not in r3]
                            r = list(np.unique(ri + r3))
                        l = list(np.unique(li + lj))
                    else:
                        r = list(np.unique(ri + rj))
                        li = [value for value in nodei[:n] if value not in rj]
                        lj = [value for value in nodej[:n] if value not in ri]
                        if li == lj:
                            lst = v1i[:n] + v2i[:n] + v1j[:n] + v2j[:n]
                            lst = [x for x in lst if x != li[0]]
                            l3 = [min(set(lst), key=lst.count)]
                            l = list(np.unique(li + l3))
                        else:
                            l = list(np.unique(li + lj))
                    # select the parent node values
                    ui1 = v1s[:, i]
                    ui2 = v2s[:, i]
                    uj1 = v1s[:, j]
                    uj2 = v2s[:, j]
                    if set(r).issubset(set(v1i)):
                        uni = 1
                    elif set(r).issubset(set(v2i)):
                        uni = 2
                    elif set(rj).issubset(set(v2i[1:])):
                        uni = 2
                    elif set(rj).issubset(set(v1i[1:])):
                        uni = 1
                    if set(r).issubset(set(v1j)):
                        unj = 1
                    elif set(r).issubset(set(v2j)):
                        unj = 2
                    elif set(ri).issubset(set(v2j[1:])):
                        unj = 2
                    elif set(ri).issubset(set(v1j[1:])):
                        unj = 1

                    # calculate the conditional CDF
                    v1igs = hfunc(copi, ui1, ui2, pari, un=uni)
                    v2jgs = hfunc(copj, uj1, uj2, parj, un=unj)
                    ktau.append(
                        abs(st.kendalltau(v1igs, v2jgs)[0])
                    )  # calculate the absolute kendall tau between v1igs and v2jgs, add it to the ktau list
                    # add the parent node values
                    v1_k.append(v1igs)
                    v2_k.append(v2jgs)

                    del uj1, ui1, ui2, uj2

                    node.append(l + ["g"] + r)  # set node name
                    lk.append(l)  # add left side of node
                    rk.append(r)  # add rigth side of node
            orderk = pd.DataFrame(
                {"v1": v1, "v2": v2, "tauabs": ktau, "node": node, "l": lk, "r": rk}
            )  # put information in dataframe for tree k

            if vine == "R":

                if len(orderk) > dimen - k:
                    sorder2 = order.node  # get all the unique parent nodes
                    # divide the unique nodes in Y and X
                    if (len(Y) + 1 - k) > 1:
                        Y2 = list(sorder2[: len(Y) - k].reset_index(drop=True))
                        X2 = list(sorder2[len(Y) - k :].reset_index(drop=True))
                    else:
                        Y2 = list(sorder2[:1].reset_index(drop=True))
                        X2 = list(sorder2[1:].reset_index(drop=True))
                    Y3 = Y2.copy()
                    X3 = X2.copy()

                    orderk = orderk.sort_values(by="tauabs", ascending=False)
                    indexi = list(orderk.index)
                    orderk = orderk.reset_index(drop=True)
                    inde = []
                    inde2 = []

                    if len(Y2) > 1:
                        for j in range(len(Y2) - 1):
                            for z in range(j + 1, len(Y2)):
                                if z == 1:
                                    l = (
                                        (orderk["v1"].apply(lambda x: Y2[j] == x))
                                        & (orderk["v2"].apply(lambda x: Y2[z] == x))
                                    ) | (
                                        (orderk["v1"].apply(lambda x: Y2[z] == x))
                                        & (orderk["v2"].apply(lambda x: Y2[j] == x))
                                    )
                                else:
                                    l = (
                                        l
                                        | (
                                            (orderk["v1"].apply(lambda x: Y2[j] == x))
                                            & (orderk["v2"].apply(lambda x: Y2[z] == x))
                                        )
                                        | (
                                            (orderk["v1"].apply(lambda x: Y2[z] == x))
                                            & (orderk["v2"].apply(lambda x: Y2[j] == x))
                                        )
                                    )

                        l = np.where(l == True)[0]
                        for z in l:
                            if z == l[0]:
                                order = orderk.loc[z].to_frame().T
                                inde.append(z)
                                inde2.append(indexi[z])
                            if z in inde:
                                continue
                            lst = (
                                list(
                                    order.v2.astype(str)[
                                        order.v1.astype(str) == str(orderk.v1[z])
                                    ]
                                )
                                + list(
                                    order.v1.astype(str)[
                                        order.v2.astype(str) == str(orderk.v1[z])
                                    ]
                                )
                                + list(
                                    order.v2.astype(str)[
                                        order.v1.astype(str) == str(orderk.v2[z])
                                    ]
                                )
                                + list(
                                    order.v1.astype(str)[
                                        order.v2.astype(str) == str(orderk.v2[z])
                                    ]
                                )
                            )
                            l1 = list(
                                order.v2.astype(str)[
                                    order.v1.astype(str) == str(orderk.v1[z])
                                ]
                            ) + list(
                                order.v1.astype(str)[
                                    order.v2.astype(str) == str(orderk.v1[z])
                                ]
                            )
                            lk = l1.copy()
                            while len(lk) > 0:
                                lk2 = lk.copy()
                                lk = []
                                for j in lk2:
                                    ln = list(
                                        order.v2.astype(str)[order.v1.astype(str) == j]
                                    ) + list(
                                        order.v1.astype(str)[order.v2.astype(str) == j]
                                    )
                                    try:
                                        ln.remove(str(orderk.v1[z]))
                                    except:
                                        pass

                                    for s in l1:
                                        try:
                                            ln.remove(s)
                                        except:
                                            pass
                                    l1 = l1 + ln
                                    lk = lk + ln

                            l2 = list(
                                order.v2.astype(str)[
                                    order.v1.astype(str) == str(orderk.v2[z])
                                ]
                            ) + list(
                                order.v1.astype(str)[
                                    order.v2.astype(str) == str(orderk.v2[z])
                                ]
                            )
                            lk = l2.copy()
                            while len(lk) > 0:
                                lk2 = lk.copy()
                                lk = []
                                for j in lk2:
                                    ln = list(
                                        order.v2.astype(str)[order.v1.astype(str) == j]
                                    ) + list(
                                        order.v1.astype(str)[order.v2.astype(str) == j]
                                    )
                                    try:
                                        ln.remove(str(orderk.v2[z]))
                                    except:
                                        pass

                                    for s in l2:
                                        try:
                                            ln.remove(s)
                                        except:
                                            pass
                                    l2 = l2 + ln
                                    lk = lk + ln
                            skip = False
                            for val in l1:
                                if val in l2:
                                    skip = True
                                    break
                            if skip == True:
                                continue
                            # if (len(l1) > 1 and len(l2) >0) or (len(l1) > 0 and len(l2) >1):
                            # continue
                            if len(lst) == len(set(lst)):
                                order = pd.concat(
                                    [order, orderk.loc[z].to_frame().T],
                                    ignore_index=True,
                                )
                                inde.append(z)
                                inde2.append(indexi[z])
                                break
                    else:
                        for j in Y2:
                            for z in X2:
                                if j == Y2[0] and z == X2[0]:
                                    l = (
                                        (orderk["v1"].apply(lambda x: j == x))
                                        & (orderk["v2"].apply(lambda x: z == x))
                                    ) | (
                                        (orderk["v1"].apply(lambda x: z == x))
                                        & (orderk["v2"].apply(lambda x: j == x))
                                    )
                                else:
                                    l = (
                                        l
                                        | (
                                            (orderk["v1"].apply(lambda x: j == x))
                                            & (orderk["v2"].apply(lambda x: z == x))
                                        )
                                        | (
                                            (orderk["v1"].apply(lambda x: z == x))
                                            & (orderk["v2"].apply(lambda x: j == x))
                                        )
                                    )

                        l = np.where(l == True)[0]
                        order = orderk.loc[l[0]].to_frame().T
                        inde.append(l[0])
                        inde2.append(indexi[l[0]])
                        if order.v1.iloc[-1] in X3:

                            X3.remove(order.v1.iloc[-1])
                            Y3.append(order.v1.iloc[-1])
                            # X2.remove(int(order2.v1.iloc[-1]))
                        else:
                            X3.remove(order.v2.iloc[-1])
                            Y3.append(order.v2.iloc[-1])

                    while len(order) < (dimen - k):
                        for j in Y3:
                            for z in X3:
                                if j == Y3[0] and z == X3[0]:
                                    l = (
                                        (orderk["v1"].apply(lambda x: j == x))
                                        & (orderk["v2"].apply(lambda x: z == x))
                                    ) | (
                                        (orderk["v1"].apply(lambda x: z == x))
                                        & (orderk["v2"].apply(lambda x: j == x))
                                    )
                                else:
                                    l = (
                                        l
                                        | (
                                            (orderk["v1"].apply(lambda x: j == x))
                                            & (orderk["v2"].apply(lambda x: z == x))
                                        )
                                        | (
                                            (orderk["v1"].apply(lambda x: z == x))
                                            & (orderk["v2"].apply(lambda x: j == x))
                                        )
                                    )

                        l = np.where(l == True)[0]
                        for z in l:
                            if z in inde:
                                continue
                            lst = (
                                list(
                                    order.v2.astype(str)[
                                        order.v1.astype(str) == str(orderk.v1[z])
                                    ]
                                )
                                + list(
                                    order.v1.astype(str)[
                                        order.v2.astype(str) == str(orderk.v1[z])
                                    ]
                                )
                                + list(
                                    order.v2.astype(str)[
                                        order.v1.astype(str) == str(orderk.v2[z])
                                    ]
                                )
                                + list(
                                    order.v1.astype(str)[
                                        order.v2.astype(str) == str(orderk.v2[z])
                                    ]
                                )
                            )
                            l1 = list(
                                order.v2.astype(str)[
                                    order.v1.astype(str) == str(orderk.v1[z])
                                ]
                            ) + list(
                                order.v1.astype(str)[
                                    order.v2.astype(str) == str(orderk.v1[z])
                                ]
                            )
                            lk = l1.copy()
                            while len(lk) > 0:
                                lk2 = lk.copy()
                                lk = []
                                for j in lk2:
                                    ln = list(
                                        order.v2.astype(str)[order.v1.astype(str) == j]
                                    ) + list(
                                        order.v1.astype(str)[order.v2.astype(str) == j]
                                    )
                                    try:
                                        ln.remove(str(orderk.v1[z]))
                                    except:
                                        pass

                                    for s in l1:
                                        try:
                                            ln.remove(s)
                                        except:
                                            pass
                                    l1 = l1 + ln
                                    lk = lk + ln

                            l2 = list(
                                order.v2.astype(str)[
                                    order.v1.astype(str) == str(orderk.v2[z])
                                ]
                            ) + list(
                                order.v1.astype(str)[
                                    order.v2.astype(str) == str(orderk.v2[z])
                                ]
                            )
                            lk = l2.copy()
                            while len(lk) > 0:
                                lk2 = lk.copy()
                                lk = []
                                for j in lk2:
                                    ln = list(
                                        order.v2.astype(str)[order.v1.astype(str) == j]
                                    ) + list(
                                        order.v1.astype(str)[order.v2.astype(str) == j]
                                    )
                                    try:
                                        ln.remove(str(orderk.v2[z]))
                                    except:
                                        pass

                                    for s in l2:
                                        try:
                                            ln.remove(s)
                                        except:
                                            pass
                                    l2 = l2 + ln
                                    lk = lk + ln
                            skip = False
                            for val in l1:
                                if val in l2:
                                    skip = True
                                    break
                            if skip == True:
                                continue
                            # if (len(l1) > 1 and len(l2) >0) or (len(l1) > 0 and len(l2) >1):
                            # continue
                            # if orderk.iloc[z].v1 in X2 and orderk.iloc[z].v2 in X2:
                            # skip = False
                            # if orderk.iloc[z].v2 in Y3:

                            # numapear = sum(((order['v1'].apply(lambda x: orderk.iloc[z].v2== x )) | (order['v2'].apply(lambda x: orderk.iloc[z].v2== x))))
                            # for i in Y2:
                            #   if numapear == sum(((order['v1'].apply(lambda x: i== x )) | (order['v2'].apply(lambda x: i== x)))):
                            #     skip = True
                            #      break
                            # elif orderk.iloc[z].v1 in Y3:

                            #   numapear = sum(((order['v1'].apply(lambda x: orderk.iloc[z].v1== x )) | (order['v2'].apply(lambda x: orderk.iloc[z].v1== x))))
                            #  for i in Y2:
                            #      if numapear ==  sum(((order['v1'].apply(lambda x: i== x )) | (order['v2'].apply(lambda x: i== x)))):
                            #          skip = True
                            #          break
                            # if skip == True:
                            #    continue
                            if len(lst) == len(set(lst)):
                                order = pd.concat(
                                    [order, orderk.loc[z].to_frame().T],
                                    ignore_index=True,
                                )
                                inde.append(z)
                                inde2.append(indexi[z])
                                if order.v1.iloc[-1] in X3:

                                    X3.remove(order.v1.iloc[-1])
                                    Y3.append(order.v1.iloc[-1])
                                    # X2.remove(int(order2.v1.iloc[-1]))
                                else:
                                    X3.remove(order.v2.iloc[-1])
                                    Y3.append(order.v2.iloc[-1])

                                break

                    # orderk = orderk.sort_values(by='tauabs', ascending=False)
                    orderk = order
                    v1_k = np.array([v1_k[ind] for ind in inde2]).T
                    v2_k = np.array([v2_k[ind] for ind in inde2]).T
                    orderk = orderk.reset_index(drop=True)
                    orderk["tree"] = k - 1
                else:
                    orderk = orderk.sort_values(
                        by="tauabs", ascending=False
                    )  # sort this dataframe from highest to lowest tauabs
                    v1_k = np.array(
                        [v1_k[ind] for ind in orderk.index]
                    ).T  # sort array of first node
                    v2_k = np.array(
                        [v2_k[ind] for ind in orderk.index]
                    ).T  # sort array of second node
                    orderk = orderk.reset_index(drop=True)
                    orderk["tree"] = k - 1
                    v1_2 = v1_k.copy()
                    v2_2 = v2_k.copy()

            elif vine == "D":
                orderk = orderk.sort_values(
                    by="tauabs", ascending=False
                )  # sort this dataframe from highest to lowest tauabs
                v1_k = np.array(
                    [v1_k[ind] for ind in orderk.index]
                ).T  # sort array of first node
                v2_k = np.array(
                    [v2_k[ind] for ind in orderk.index]
                ).T  # sort array of second node
                orderk = orderk.reset_index(drop=True)
                orderk["tree"] = k - 1
                v1_2 = v1_k.copy()
                v2_2 = v2_k.copy()

            elif vine == "C":
                if len(orderk) > dimen - k:
                    subnodes = list(order.node)  # see all the unique parent nodes
                    taus = (
                        []
                    )  # list to put the sum of all tauabs in for a specific nodes
                    if (len(Y) + 1 - k) > 1:
                        for i in range(len(Y) + 1 - k):
                            orderksub = orderk[
                                (orderk["v1"].apply(lambda x: x == subnodes[i]))
                                | (orderk["v2"].apply(lambda x: x == subnodes[i]))
                            ]
                            taus.append(
                                sum(orderksub.tauabs)
                            )  # calculate the sum of all taus for a specific variable
                        i = np.where(np.array(taus) == max(taus))[0][
                            0
                        ]  # find where the sum of the taus is the highest to find the vairable to place in the center of the first tree
                        orderk = orderk[
                            (orderk["v1"].apply(lambda x: x == subnodes[i]))
                            | (orderk["v2"].apply(lambda x: x == subnodes[i]))
                        ]  # select rows where this node is included
                    else:
                        orderk = orderk[
                            (orderk["v1"].apply(lambda x: x == subnodes[0]))
                            | (orderk["v2"].apply(lambda x: x == subnodes[0]))
                        ]

                    inde = list(orderk.index)
                    v1_k = np.array(
                        [v1_k[ind] for ind in inde]
                    ).T  # sort array of first node
                    v2_k = np.array(
                        [v2_k[ind] for ind in inde]
                    ).T  # sort array of second node
                    orderk = orderk.reset_index(drop=True)
                    orderk["tree"] = k - 1
                    v1_k = v1_k.copy()
                    v2_k = v2_k.copy()

                else:
                    orderk = orderk.sort_values(by="tauabs", ascending=False)
                    v1_k = np.array([v1_k[ind] for ind in orderk.index]).T
                    v2_k = np.array([v2_k[ind] for ind in orderk.index]).T
                    orderk = orderk.reset_index(drop=True)
                    orderk["tree"] = k - 1

            for j in range(len(orderk)):
                u3 = np.vstack((v1_k[:, j], v2_k[:, j])).T
                cop, rho, aic = bestcop(copsi, u3)
                aics.append(aic)
                rhos.append(rho)
                cops.append(cop)

            orderk["rhos"] = rhos
            orderk["cop"] = cops
            orderk["AIC"] = aics
            locals()["v1_" + str(k)] = v1_k
            locals()["v2_" + str(k)] = v2_k
            locals()["order" + str(k)] = orderk

    order = pd.DataFrame(columns=order1.columns)
    for i in range(1, dimen):
        order = pd.concat([order, locals()["order" + str(i)]]).reset_index(drop=True)

    a = np.empty((dimen, dimen))
    c = np.empty((dimen, dimen))
    a[:] = np.nan
    c[:] = np.nan

    order["used"] = 0
    combinations = list(product([True, False], repeat=dimen))
    for i in list(range(dimen - 1))[::-1]:
        k1 = sorted(
            np.array(
                order[(order.tree == i) & (order["used"] == 0)].node.iloc[0][:2]
            ).astype(int)
        )[::-1]
        order.loc[(order["tree"] == i) & (order["used"] == 0), "used"] = 1
        t1 = i - 1
        ii = dimen - 2 - i
        a[i : dimen - ii, ii] = k1
        s = k1[-1]
        for j in list(range(0, i))[::-1]:
            orde = order[(order.tree == j) & (order["used"] == 0)]
            for k in range(len(orde)):
                arr = np.array(orde.node.iloc[k][:2]).astype(int)
                if np.isin(s, arr) == True:
                    inde = orde.iloc[k].name
                    a[j, ii] = arr[arr != s][0]
                    order["used"][inde] = 1
                    continue

    a[0, dimen - 1] = a[0, dimen - 2]
    orderk = pd.DataFrame(columns=order.columns)
    p = np.empty((dimen, dimen))
    p[:] = np.nan
    p = p.astype(object)
    for i in list(range(dimen - 1)):
        orde = order[order.tree == i]
        for k in list(range(dimen - 1 - i)):
            ak = a[:, k]
            akn = np.array([ak[-1 - k], ak[i]]).astype(int)
            for j in range(len(orde)):
                arr = np.array(orde.node.iloc[j][:2]).astype(int)
                if sum(np.isin(akn, arr)) == 2:
                    orderj = order.loc[[orde.index[j]]]
                    p[i, k] = orderj.rhos.iloc[0]
                    c[i, k] = orderj.cop.iloc[0]
                    if i == 0:
                        orderj.node.iloc[0] = list(akn)
                    else:
                        orderj.node.iloc[0] = (
                            list(akn) + ["|"] + list((ak.astype(int)[:i])[::-1])
                        )
                    orderk = pd.concat([orderk, orderj]).reset_index(drop=True)

    sorder = list(np.diag(a[::-1])[::-1])
    if set(sorder[-len(X) :]) != set(X):
        sorders = samplingorder(a)
        if len(Y) >= len(X):
            for sorder in sorders:
                if set(sorder[-len(X) :]) == set(X):
                    break
        else:
            for sorder in sorders:
                if set(sorder[: len(Y)]) == set(Y):
                    break
        a, p, c = samplingmatrix(a, c, p, sorder)

    if printing == True:
        for i in list(range(0, dimen - 1)):
            orde = orderk[orderk.tree == i].reset_index(drop=True)
            print("** Tree: ", i + 1)
            for j in range(len(orde)):
                if i != 0:
                    nodej = ",".join(map(str, orde.node[j])).replace(",|,", "|")
                else:
                    nodej = ",".join(map(str, orde.node[j]))
                print(
                    nodej,
                    " ---> ",
                    copulas[int(orde.cop[j])],
                    ": parameters = ",
                    orde.rhos[j],
                )

    return a, p, c


# %%


def plotvine(a, plottitle=None, variables=None, savepath=None):
    """
    Plots the vine structure

    Arguments:
        *a* : The vine tree structure provided as a triangular matrix, composed of integers. The integer refers to different variables depending on which column the variable was in u1, where the first column is 0 and the second column is 1, etc.

        *pltotitle* : title of the plot

        *savepath* : path to save the plot

    Returns:
     *plot*



    """
    dimen = a.shape[0]  # dimension of copula
    order = pd.DataFrame(
        columns=["node", "l", "r", "tree"]
    )  # dataframe with vine copula structure.
    s = 0
    for i in list(range(dimen - 1)):
        for k in list(range(dimen - 1 - i)):
            ak = a[:, k]
            akn = np.array([ak[-1 - k], ak[i]]).astype(int)
            if i == 0:
                single_row_values = {
                    "node": list(akn),
                    "l": akn[0],
                    "r": akn[1],
                    "tree": i,
                }
            else:
                single_row_values = {
                    "node": list(akn) + ["|"] + list((ak.astype(int)[:i])[::-1]),
                    "l": list(akn),
                    "r": list((ak.astype(int)[:i])[::-1]),
                    "tree": i,
                }

            order.loc[s] = single_row_values
            s = s + 1

    for t in list(range(dimen - 1)):
        orderk = order[order.tree == t].reset_index(drop=True)
        if t == 0:
            orderk["v1"] = orderk.l
            orderk["v2"] = orderk.r
            rhos = []
            v1_1 = []
            v2_1 = []
            cops = []

            for j in range(len(orderk)):
                v1i = int(orderk.v1[j])  # first node
                v2i = int(orderk.v2[j])  # second node

            locals()["order" + str(t + 1)] = orderk
        else:

            v1k = []
            v2k = []
            for j in range(len(orderk)):
                orderk2 = order[order.tree == t - 1].reset_index(drop=True)
                l = orderk.l[j] + orderk.r[j]
                subnodes = []
                for k in range(len(orderk2)):
                    subnodes.append(sum(1 for item in orderk2.node[k] if item in l))
                subnodes = np.array(subnodes) == len(l) - 1
                orderk2 = orderk2[subnodes].reset_index(drop=True)
                v1k.append(orderk2.node[0])
                v2k.append(orderk2.node[1])
            orderk["v1"] = v1k
            orderk["v2"] = v2k
            locals()["order" + str(t + 1)] = orderk
    n = dimen - 1
    fig, axes = plt.subplots(dimen - 1, 1, figsize=(n * 2, n * 3))  # 5 rows, 1 column
    leg_labels = {}
    if variables != None:
        for i in range(len(variables)):
            leg_labels.update({i: variables[i]})

    for t, ax in zip(
        range(1, dimen), axes.flat
    ):  # Iterate through subplots and loop indices
        if t == 1:
            orderk = locals()["order" + str(t)]
            edges = list(orderk.node)
            edges = [tuple(sublist) for sublist in edges]
            edge_labels = {
                edge: ",".join(map(str, orderk.node[i])) for i, edge in enumerate(edges)
            }
        elif t == 2:
            orderk = locals()["order" + str(t)]
            edges = [
                (",".join(map(str, orderk.v1[i])), ",".join(map(str, orderk.v2[i])))
                for i in range(len(orderk))
            ]
            edge_labels = {
                edge: ",".join(map(str, orderk.node[i])).replace(",|,", "|")
                for i, edge in enumerate(edges)
            }
        else:
            orderk = locals()["order" + str(t)]
            edges = [
                (
                    ",".join(map(str, orderk.v1[i])).replace(",|,", "|"),
                    ",".join(map(str, orderk.v2[i])).replace(",|,", "|"),
                )
                for i in range(len(orderk))
            ]
            edge_labels = {
                edge: ",".join(map(str, orderk.node[i])).replace(",|,", "|")
                for i, edge in enumerate(edges)
            }

        G = nx.Graph()  # Create graph.
        G.add_edges_from(edges)  # Add edges to the graph
        pos = nx.spring_layout(G)  # Position the nodes using the spring layout

        d = nx.degree(G)
        try:
            sizes = len(edges[0][0]) * 400
        except:
            sizes = len(edges[0]) * 400

        nx.draw_networkx_labels(
            G, pos, ax=ax, bbox=dict(facecolor="skyblue"), font_size=13
        )  # Draw labels.

        nx.draw_networkx_edges(
            G,
            pos,
            edge_color="black",
            ax=ax,
        )  # Draw black edges between nodes

        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_color="black", ax=ax, font_size=12
        )  # Draw text labels above each edge

        ax.axis("off")  # Remove the box around the plot

        ax.set_title(f"Tree {t}")  # Set title for the subplot

    # title
    if plottitle != None:
        fig.suptitle(plottitle, fontsize=16, y=0.93)
    if variables != None:
        plt.text(
            0.9,
            0.5,
            "\n".join([f"{key}  :  {value}" for key, value in leg_labels.items()]),
            transform=plt.gcf().transFigure,
            fontsize=15,
            verticalalignment="center",
        )
    # save
    if savepath != None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()  # Show plot.
