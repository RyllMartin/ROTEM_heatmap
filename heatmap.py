import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

def ROTEM_Heatmap(df_data, unit_clm, rotem_clm, diff_clm, diff_g0, diff_g1, parameter_lst, export_path_fig,
                        paired=False, export_path_stats=None, catch_na=6, catch_ties=14, max_size_points=None,
                        adjusting=True, palette_color="binary", title_text=""):

    """
    Function to generate a plot-heatmap visualizing the significance level, p-value and the mean change in percent
    comparing 2 different groups or 2 different points in time via ROTEM-Data.

    Alpha/the significance level is used to decide whether a test is displayed as a point at all. The color intensity
    represents the p-value with more intense colors the smaller the p-value. Point size represents the mean change in
    percent (calculated as (G1-G0)/G0 so as change from G0) with larger points the larger the change in percent.

    If unpaired data, then the data of each parameter of each group are analyzed regarding normal distribution via the
    Shapiro-Wilk test. If the data of both groups follow a normal distribution then the independent t-test (Welch
    t-test, assuming unequal variance) is used to compare the data and calculate the statistics. If the data from either
    group is not normally distributed then the Mann-Whitney-U test is used to compare the data and calculate the
    statistics. The change in percent is calculated as the difference in means (G1.mean() - GO.mean()) divided by G0.mean if
    both G1 and G0 data are normally distributed and calculated as the difference in medians divided by G0.median if
    either data is not normally distributed.

    If paired data, then the array of the change (difference) of each parameter in each unit (G1 - G0 for each unit) is
    analyzed regarding normal distribution via Shapiro-Wilk test. If the data follows a normal distribution then the
    paired-sample t-test is used to compare the data and calculate the statistics. If the data is not normally
    distributed then the Wilcoxon signed-rank test is used to compare the data and calculate the statistics. The change
    in percent is calculated as the mean of the differences divided by G0.mean if the differences are normally
    distributed and calculated as the median of the difference divided by G0.median if the differences are not normally
    distributed. All statistical tests are from the scipy library.

    If adjusting for multiple comparisons, then the alpha level is adjusted for multiple comparisons with the
    Bonferroni correction which is adjusted for correlated tests (as e.g. the A or the AR values are highly correlated):
    alpha = alpha / m
    m = 1+ (M_unadj - 1) * (1 - x/M_unadj)
    M_unadj: number of tests performed
    x: Variance of the eigenvalues derived from the correlation matrix of the variable
    (Nyholt DR. A simple correction for multiple testing for single-nucleotide polymorphisms in linkage disequilibrium
    with each other. Am J Hum Genet. 2004 Apr;74(4):765-9. doi: 10.1086/383251. Epub 2004 Mar 2. PMID: 14997420)

    The code is a 3-step process:
    1. Calculating an initial dataframe with various statistics for all comparisons
    2. Creating the final dataframe from the first dataframe that is passed to the plotter
    3. Creating the visualization from the final dataframe

    Dependencies:
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import scipy.stats as stats
    import matplotlib.pyplot as plt

    df_data : Dataframe; This is the dataframe containing the ROTEM data. The direct source data can be used with
    little editing. The dataframe should have the layout of a few identifying columns (rotem_clm, diff_clm (and
    unit_clm if paired data)) followed by the columns of all the parameters (CT, A5, A10 etc.).

    unit_clm: String; Column label of the column identifying the individual units (eg. individual patients)

    rotem_clm: String; Column_label of the column identifying the kind of ROTEM used (FIBTEM, EXTEM etc.)

    diff_clm: String; Column_label of the column identifying the 2 groups/time points that are to be compared

    diff_g0: Int, float or string; Value or string coding for Group 0 or time point 0 in the diff_clm. NOTE: If the data
    is paired, then the difference and change in percent is calculated as Group 1 - Group 0.

    diff_g1: Int, float or string; Value or string coding for Group 1 or time point 1 in the diff_clm. NOTE: If the data
    is paired, then the difference change in percent is calculated as Group 1 - Group 0.

    parameter_lst: List; List of parameters (eg. CT, A5, A10 etc.) that you want to analyze/visualize. Has to match the
    column names.

    export_path_fig: String; path (including file name and data type for the figure) for the final figure to be exported
    to.

    paired: Boolean, preset False; Set to True if the data is paired (eg. comparing 2 time points on the same patients).

    export_path_stats: String, preset None; Insert path (incl. filename) if you want to export the intermediate
    statistics dataframe. pd.to_excel is used, so the datatype has to be set to .xlsx!

    catch_na: Integer, preset to 6; Catches empty (or almost) empty columns which would either distort the data and/or
    interfere with the testing. Number should be set to an appropriate number depending on the sample size used.

    catch_ties: Integer, for paired data, preset to 14; Catches paired comparisons with a lot of ties (typically an
    issue with the lysis parameters) which would either distort the data and/or interfere with the testing (producing
    unreliable results). Number should be set to an appropriate number depending on the sample size used.

    max_size_points: Float or int, preset to None: Upper limit of individual point size (which represents the effect
    size). If left at preset None, then it will be set to the maximum change in percent of any point of that figure. Can be
    set to a constant value to better visualize changes in change in percent over several heatmaps.

    adjusting: Boolean; preset is True: Whether to adjust for multiple comparisons

    palette_color: seaborn colorpalette, preset to binary; Color palette for the heatmap

    title_text: string, preset to empty; Text for the title of the heatmap

    """

    # make iterator list of different ROTEM-kinds out of the unique values of the ROTEM-column;
    # I disabled this to make sure that the heatmaps had a uniform layout
    rotem_types_lst = df_data[rotem_clm].unique().tolist()

    # column names of the intermediate dataframe containing all the statistics (independent vs dependent) and the
    # column names for the dataframe for the heatmap

    clm_names_ind = ["ROTEM_type", "Test", "p_shap_G0", "p_shap_G1", "p_value_t", "p_value_mann",
                     "effect_size_mean", "effect_size_median"]
    clm_names_dep = ["ROTEM_Art", "Test", "p_shap_diff", "p_value_t", "p_value_wilcox", "effect_size_mean",
                     "effect_size_median", "many_ties"]
    clm_names_fin = ["ROTEM-Art", "Test", "p-value <", "Change in %"]

    # lists that will accumulate the individual rows of the stats and heatmap dataframe
    lst_dfs_stats = []
    lst_dfs_heatmap = []

    ## make the statistics file first
    # otherwise: in rotem_types_lst
    for rotem in ["FIBTEM", "EXTEM", "HEPTEM", "rTPA"]:

        for test in parameter_lst:

            # go down the independent tests path if the data is not paired
            if paired == False:
                # prepare dataframe of individual test for calculations
                df_temp = df_data.loc[(df_data[rotem_clm] == rotem), [diff_clm, test]]

                # catch empty tests -> tests with na values greater than catch_na (default 6) will not be displayed, 6 was
                # chosen for our data as this was a good differentiator between columns missing just a few values and
                # columns missing most values. This will depend on the number of observations per group.
                if df_temp[test].isna().sum() > catch_na:
                    p_shap_0 = np.nan
                    p_shap_1 = np.nan
                    p_value_t = np.nan
                    p_value_mann = np.nan
                    effect_size_mean = np.nan
                    effect_size_median = np.nan
                    df_add = pd.DataFrame([[rotem, test,  p_shap_0, p_shap_1, p_value_t,
                                            p_value_mann, effect_size_mean, effect_size_median]], columns=clm_names_ind)
                    lst_dfs_stats.append(df_add)
                    continue

                # calculate statistics
                statistic_shap_0, p_shap_0 = stats.shapiro(df_temp.loc[(df_temp[diff_clm] == diff_g0), [test]])
                statistic_shap_1, p_shap_1 = stats.shapiro(df_temp.loc[(df_temp[diff_clm] == diff_g1), [test]])

                statistic_t, p_value_t = stats.ttest_ind(df_temp.loc[(df_temp[diff_clm] == diff_g0), [test]],
                                                         df_temp.loc[(df_temp[diff_clm] == diff_g1), [test]], equal_var=False)

                statistic_mann, p_value_mann = stats.mannwhitneyu(df_temp.loc[(df_temp[diff_clm] == diff_g0), [test]],
                                                                  df_temp.loc[(df_temp[diff_clm] == diff_g1), [test]])

                # Change in percent calculates as G1-G0 / G0 -> so as change from G0 in percent!
                effect_size_mean = (df_temp.loc[(df_temp[diff_clm] == diff_g1), [test]].mean() - \
                                    df_temp.loc[(df_temp[diff_clm] == diff_g0), [test]].mean()) / df_temp.loc[
                                       (df_temp[diff_clm] == diff_g0), [test]].mean()

                effect_size_median = (df_temp.loc[(df_temp[diff_clm] == diff_g1), [test]].median() - \
                                      df_temp.loc[(df_temp[diff_clm] == diff_g0), [test]].median()) / df_temp.loc[
                                         (df_temp[diff_clm] == diff_g0), [test]].median()

                # Round values
                p_shap_0 = round(float(p_shap_0), 6)
                p_shap_1 = round(float(p_shap_1), 6)
                p_value_t = round(float(p_value_t), 6)
                p_value_mann = round(float(p_value_mann), 6)
                effect_size_mean = round(float(effect_size_mean), 6)
                effect_size_median = round(float(effect_size_median), 6)

                # add all the values as row to the list of dataframes
                df_add = pd.DataFrame(
                    [[rotem, test,  p_shap_0, p_shap_1, p_value_t, p_value_mann, effect_size_mean, effect_size_median]],
                    columns=clm_names_ind)
                lst_dfs_stats.append(df_add)

            # go down the dependent tests path if the data is paired
            elif paired == True:
                # prepare dataframe of individual test for calculations
                df_temp_G0 = df_data.loc[(df_data[rotem_clm] == rotem) & (df_data[diff_clm] == diff_g0),
                                         [unit_clm, test]]
                df_temp_G1 = df_data.loc[(df_data[rotem_clm] == rotem) & (df_data[diff_clm] == diff_g1),
                                         [unit_clm, test]]


                # df_diff calculated as G1-G0
                df_temp_G0v2 = df_temp_G0.set_index(unit_clm)
                df_temp_G1v2 = df_temp_G1.set_index(unit_clm)
                df_temp_diff = df_temp_G1v2[test] - df_temp_G0v2[test]
                df_temp_diff.reset_index(drop=True, inplace=True)

                df_temp_diff.dropna(inplace=True)

                # catch empty tests -> tests with na values greater than catch_na (default 6) will not be displayed, 6 was
                # chosen for our data as this was a good differentiator between columns missing just a few values and
                # columns missing most values. This will depend on the number of observations per group.
                # With small sample sizes there might be the problem that depending on the ties, distribution of the
                # missing values etc. there are less than 3 values which will lead to errors with Shapiro-Wilk
                if df_temp_G0[test].isna().sum() > catch_na or len(df_temp_diff) < 3:
                    p_shap_diff = np.nan
                    p_value_t = np.nan
                    p_value_wil = np.nan
                    effect_size_mean = np.nan
                    effect_size_median = np.nan
                    ties = np.nan

                    df_add = pd.DataFrame([[rotem, test, p_shap_diff, p_value_t, p_value_wil, effect_size_mean,
                                            effect_size_median, ties]], columns=clm_names_dep)
                    lst_dfs_stats.append(df_add)
                    continue

                # catch too many ties -> tests with numer of ties greater than catch_ties (default 14) will not be
                # displayed/considered, 14 was chosen for our data as this was a good differentiator for our group sizes.
                # This will depend on the number of observations per group.
                if df_temp_diff.isin([0]).sum() > catch_ties:
                    ties = 1
                    statistic_shap_diff, p_shap_diff = stats.shapiro(df_temp_diff)
                    statistic_t, p_value_t = stats.ttest_1samp(a=df_temp_diff, popmean=0, nan_policy="omit")

                    p_shap_diff = round(float(p_shap_diff), 6)
                    p_value_t = round(float(p_value_t), 6)
                    p_value_wil = np.nan

                    # Changes in percent calculates as Diff/ G0 -> so as change from G0 in percent
                    effect_size_mean = df_temp_diff.mean() / df_temp_G0[test].mean()
                    effect_size_median = df_temp_diff.median() / df_temp_G0[test].median()

                    df_add = pd.DataFrame([[rotem, test, p_shap_diff, p_value_t, p_value_wil, effect_size_mean,
                                            effect_size_median, ties]], columns=clm_names_dep)
                    lst_dfs_stats.append(df_add)
                    continue

                # calculate stats
                ties = 0

                statistic_shap_diff, p_shap_diff = stats.shapiro(df_temp_diff)
                statistic_t, p_value_t = stats.ttest_1samp(a=df_temp_diff, popmean=0, nan_policy="omit")
                statistic_wil, p_value_wil = stats.wilcoxon(x=df_temp_diff)

                # round the stats
                p_shap_diff = round(float(p_shap_diff), 6)
                p_value_t = round(float(p_value_t), 6)
                p_value_wil = round(float(p_value_wil), 6)

                # Change in percent calculates as G1-G0/ G0 -> so as change from G0 in percent
                effect_size_mean = df_temp_diff.mean() / df_temp_G0[test].mean()
                effect_size_median = df_temp_diff.median() / df_temp_G0[test].median()

                df_add = pd.DataFrame([[rotem, test, p_shap_diff, p_value_t, p_value_wil, effect_size_mean,
                                        effect_size_median, ties]], columns=clm_names_dep)
                lst_dfs_stats.append(df_add)

    # This now forms our stats dataframe concluding the first part
    df_all_stats = pd.concat(lst_dfs_stats)

    # dealing with the "inf" strings in the ef mean
    for test in ["effect_size_mean", "effect_size_median"]:
        for index in df_all_stats[test].index:
            x = df_all_stats.at[index, test]
            if type(x) is str:
                y = 1e-9
                df_all_stats.at[index, test] = y
            else:
                continue

    # option to export the stats dataframe as an excel frame to a specified path
    if export_path_stats != None:
        df_all_stats.to_excel(export_path_stats, index=False)

    # adjusting for multiple comparisons if adjusting = true
    if adjusting == True:
        ## calculating the adjusted alpha value for multiple comparisons
        # prepare our dataframe for making the correlation matrix: columns with NA values are dropped as well as columns
        # with 0 variance
        df_wide = pd.pivot(df_data, index=[unit_clm, diff_clm], columns=rotem_clm, values=parameter_lst)
        df_wide.columns = df_wide.columns.get_level_values(0) + '_' +  df_wide.columns.get_level_values(1)
        df_wide.reset_index(inplace=True)
        df_wide_2 = df_wide.iloc[:, 2:]
        df_wide_pruned = df_wide_2[[i for i in df_wide_2.columns if df_wide_2[i].std()**2 != 0]]
        df_ready = df_wide_pruned.dropna(axis=1)

        # calculating M_eff, see Nyholt DR. A simple correction for multiple testing for single-nucleotide polymorphisms
        # in linkage disequilibrium with each other. Am J Hum Genet. 2004 Apr;74(4):765-9. doi: 10.1086/383251
        # Epub 2004 Mar 2. PMID: 14997420
        correlation_matrix = df_ready.corr()
        eig_values, v = np.linalg.eig(correlation_matrix.values)
        var_eig = eig_values.std() ** 2
        M_eff = 1 +(df_ready.shape[1] - 1) * (1 - (var_eig/df_ready.shape[1]))

        # calculate the adjusted p-values
        if paired == False:
            df_all_stats.p_t = df_all_stats.p_t * M_eff
            df_all_stats.p_mann = df_all_stats.p_mann * M_eff

        if paired == True:
            df_all_stats.p_t = df_all_stats.p_t * M_eff
            df_all_stats.p_will = df_all_stats.p_will * M_eff


    ## make the final Dataframe that will be passed to the plotter
    if paired == False:
        for rotem_art, test, p_shap_g0, p_shap_g1, p_t, p_mann, ef_mean, ef_median in df_all_stats.itertuples(index=False):

            # Use the median/p-mann if the data does not have a normal distribution. Use the mean/p_t if the data for
            # that parameter was from a normal distribution. P-values are grouped to <0.05, <0.01 and <0.001 .
            # If the p-value is greater than 0.05 the p_value is set to 0.05 and the effect size (change in percent)
            # is set to a minimal value. This is purely for the visualization: If the change in percent is not set to 0
            # as in this case the graph would stop prematurely (display no values) if a lot (or all) of the parameters
            # do not have a significant difference. Setting the change in percent to a minimal value leads to a
            # visually empty graphical output as the resulting point is too small to see. As the point is too small to
            # see, the p-value could be seen as irrelevant and set to 0. However, this would lead to the graph legend
            # displaying 0 as a possible value. Therefore, it is set to 0.05
            if (p_shap_g0 < 0.05 or p_shap_g1 < 0.05) and p_mann < 0.05:
                effect_size = abs(ef_median) * 100
                p_value = 0.05

                if p_mann < 0.01:
                    p_value = 0.01
                    if p_mann < 0.001:
                        p_value = 0.001

            else:
                if p_t < 0.05:
                    effect_size = abs(ef_mean) * 100
                    p_value = 0.05

                    if p_t < 0.01:
                        p_value = 0.01
                        if p_t < 0.001:
                            p_value = 0.001

                else:
                    p_value = 0.05
                    effect_size = 1e-10

            df_add = pd.DataFrame([[rotem_art, test, p_value, effect_size]], columns=clm_names_fin)
            lst_dfs_heatmap.append(df_add)

    elif paired == True:
        for rotem_art, test, p_shap_diff, p_t, p_wil, ef_mean, ef_median, ties in df_all_stats.itertuples(index=False):
            # if too many ties then set the p-value/change in percent to "unnoticable" (see documentation below)
            # Use the median/p-wil if the data does not have a normal distribution. Use the mean/p_t if the data for
            # that parameter was from a normal distribution. P-values are grouped to <0.05, <0.01 and <0.001 .
            # If the p-value is greater than 0.05 the p_value is set to 0.05 and the effect size (change in percent)
            # is set to a minimal value. This is purely for the visualization: If the change in percent is not set to 0
            # as in this case the graph would stop prematurely (display no values) if a lot (or all) of the parameters
            # do not have a significant difference. Setting the change in percent to a minimal value leads to a
            # visually empty graphical output as the resulting point is too small to see. As the point is too small to
            # see, the p-value could be seen as irrelevant and set to 0. However, this would lead to the graph legend
            # displaying 0 as a possible value. Therefore, it is set to 0.05.
            if ties == 1:
                p_value = 0.05
                effect_size = 1e-10
            else:
                if p_shap_diff < 0.05  and p_wil < 0.05:
                    effect_size = abs(ef_median) * 100
                    p_value = 0.05

                    if p_wil < 0.01:
                        p_value = 0.01
                        if p_wil < 0.001:
                            p_value = 0.001

                else:
                    if p_t < 0.05:
                        effect_size = abs(ef_mean) * 100
                        p_value = 0.05

                        if p_t < 0.01:
                            p_value = 0.01
                            if p_t < 0.001:
                                p_value = 0.001
                    else:
                        p_value = 0.05
                        effect_size = 1e-10

            df_add = pd.DataFrame([[rotem_art, test, p_value, effect_size]], columns=clm_names_fin)
            lst_dfs_heatmap.append(df_add)

    # This now forms our final dataframe that can be passed to the plotter, concluding the second part of this function
    df_heatmap = pd.concat(lst_dfs_heatmap)

    # set maxpoints size to the maximum change in percent in the dataframe or a fixed value which was entered. If the max_ef
    # is the minimum (1e-10) that we have set for the "null values" above, meaning that there are no significant
    # differences in all the paramters in all ROTEM-types, then the size-point-max is set to one to have a clear graph
    if max_size_points == None:
        max_ef = float(df_heatmap["Change in %"].max())
        if max_ef == 1e-10:
            size_points_max = 1
        else:
            size_points_max = max_ef
    else:
        size_points_max = max_size_points

    # Make p-values a string so that it is a qualitative and not quantitative variable that can correspond to a
    # qualitative color palette
    df_heatmap["p-value <"] = df_heatmap["p-value <"].astype(str)

    ## Plotting the actual heatmap
    sns.set_theme(style="whitegrid")

    g = sns.relplot(
        data=df_heatmap,
        x="Test", y="ROTEM-Art", hue="p-value <", size="Change in %",
        palette=palette_color, hue_order=["0.05", "0.01", "0.001"],
        edgecolor=".7",
        sizes=(1, 500), size_norm=(0, size_points_max), height=2.25, aspect=5.2
    )

    g.set(xlabel="Parameter", ylabel="")
    g.fig.suptitle(title_text, y=1.155)
    g.despine(left=True, bottom=True)
    g.ax.margins(y=0.2, x=0.02)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)

    plt.savefig(export_path_fig,
                bbox_inches='tight',
                pad_inches=0.5
                )
    plt.clf()
    plt.close()
    # this concludes the third part and therefore the whole function

    # reset the accumulator lists
    lst_dfs_stats = []
    lst_dfs_heatmap = []










