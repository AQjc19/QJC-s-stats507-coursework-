# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "marimo==0.24.0",
#     "pandas==3.0.5",
#     "numpy==2.5.2",
#     "matplotlib==3.11.1",
#     "scikit-learn==1.9.0",
#     "pytest==9.1.1",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SI 618 - Homework 5: Predicting red wine ratings

    **Name:**
    **Uniqname:** aqjc

    This homework is due **Monday, November 9, before class (3:30 PM)**. The full spec is in
    `README.md`, and the rubric is in `RUBRIC.md`. Please read both before you
    start.

    Rename this file `SI618_HW05_<uniqname>.py` before you submit, and upload
    its HTML export alongside it.

    ### Working in marimo

    - A variable can be defined in only one cell, so give each new result a
      new name, such as `cv_scores_k5`, or prefix a name with `_` to keep it
      inside its cell.
    - Add as many cells as you need, wherever you need them.
    - For written answers, edit the markdown cells marked *Your answer here*.
    - The tests at the end of each part check that your work has the right
      shape, not that it is right, so a green test does not mean that a part
      is complete.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return (
        KNeighborsClassifier,
        StandardScaler,
        cross_val_score,
        make_pipeline,
        mo,
        np,
        pd,
        plt,
        train_test_split,
    )


@app.cell
def _():
    DATA_URL = (
        "https://raw.githubusercontent.com/umsi-data-science/data/main/"
        "winequality-red.csv"
    )
    FEATURES = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol",
    ]
    # These are the values of k to cross-validate in Parts 3 and 4.
    K_VALUES = [1, 3, 5, 9, 15, 25, 41, 61, 81]
    return DATA_URL, FEATURES, K_VALUES


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 1: Look at the data (15 points)

    ### 1.1 Load the wines (5)
    """)
    return


@app.cell
def _(DATA_URL, pd):
    # Load DATA_URL into `wines`, and add a column `good`: True for a rating of 6
    # or more. Then show how many wines have each rating, and how many are good.
    wines = pd.read_csv(DATA_URL)
    wines["good"] = wines["quality"] >= 6
    wines.head()
    return (wines,)


@app.cell
def _(wines):
    wines["quality"].value_counts().sort_index().rename("number of wines")
    return


@app.cell
def _(wines):
    wines["good"].value_counts().rename("number of wines")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 Baseline (4)
    """)
    return


@app.cell
def _(wines):
    # Compute the accuracy of always predicting the more common class.
    baseline = float(wines["good"].value_counts(normalize=True).max())
    baseline
    return (baseline,)


@app.cell(hide_code=True)
def _(baseline, mo, wines):
    mo.md(rf"""
    Good wines are the more common class: {wines["good"].sum()} of the
    {len(wines):,} wines are rated 6 or more, so a "model" that calls every wine
    good, without looking at its chemistry at all, is already right
    {baseline:.1%} of the time. Any classifier we build has to be judged against
    that {baseline:.1%}, not against 0% or 50%: a model is only useful to the
    extent that it beats predicting the more common class for every wine.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.3 Scales of the measurements (6)
    """)
    return


@app.cell
def _(FEATURES, wines):
    # Show the mean, standard deviation, minimum and maximum of each of FEATURES.
    feature_summary = wines[FEATURES].agg(["mean", "std", "min", "max"]).T
    feature_summary.round(3)
    return (feature_summary,)


@app.cell(hide_code=True)
def _(feature_summary, mo):
    _sd = feature_summary["std"]
    mo.md(rf"""
    I expect the two sulfur dioxide columns to decide the unscaled distances.
    k-NN adds up the squared differences between two wines in every measurement,
    so a measurement matters in proportion to how far apart wines typically are
    in it, which is its spread, not its mean. Total sulfur dioxide has a standard
    deviation of {_sd["total sulfur dioxide"]:.1f} mg/L (it runs from
    {feature_summary.loc["total sulfur dioxide", "min"]:.0f} to
    {feature_summary.loc["total sulfur dioxide", "max"]:.0f}), and free sulfur
    dioxide {_sd["free sulfur dioxide"]:.1f} mg/L. The next largest are fixed
    acidity ({_sd["fixed acidity"]:.2f}) and residual sugar
    ({_sd["residual sugar"]:.2f}), and several measurements, such as density
    ({_sd["density"]:.4f}) and chlorides ({_sd["chlorides"]:.3f}), vary by less
    than a tenth of a unit. A typical difference of about 30 in total sulfur
    dioxide squares to about 900, which swamps a difference of 0.002 in density,
    so alcohol or volatile acidity will hardly affect which wines count as
    nearest.
    """)
    return


@app.cell(hide_code=True)
def _(baseline, pd, wines):
    def test_part_1():
        assert isinstance(wines, pd.DataFrame), "`wines` should be a DataFrame."
        assert len(wines) == 1599, (
            "`wines` should have 1,599 rows, one per wine. Keep the duplicates for now."
        )
        assert "good" in wines.columns, "`wines` needs a `good` column."
        assert wines["good"].dtype == bool, "`good` should be True or False for every wine."
        assert isinstance(baseline, float), "`baseline` should be a single number."
        assert 0.5 <= baseline <= 1, (
            "The baseline predicts the more common class, so it is at least 0.5."
        )

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 2: Scaling (20 points)

    ### 2.1 Split (4)
    """)
    return


@app.cell
def _(train_test_split, wines):
    # Put a quarter of the wines in the test set, stratified by `good`, with
    # random_state=618.
    train_wines, test_wines = train_test_split(
        wines, test_size=0.25, stratify=wines["good"], random_state=618)
    len(train_wines), len(test_wines)
    return test_wines, train_wines


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 Share of the distance by measurement (8)
    """)
    return


@app.cell
def _(FEATURES, train_wines):
    # Compute each measurement's share of the summed variances of all eleven in
    # the training set, and sort the shares from largest to smallest.
    _variances = train_wines[FEATURES].var()
    variance_share = (_variances / _variances.sum()).sort_values(ascending=False)
    variance_share.round(5)
    return (variance_share,)


@app.cell(hide_code=True)
def _(mo, variance_share):
    _top_two = variance_share["total sulfur dioxide"] + variance_share["free sulfur dioxide"]
    _rest = variance_share.drop(["total sulfur dioxide", "free sulfur dioxide"])
    mo.md(rf"""
    Yes, it confirms the prediction, and more strongly than I expected. In the
    training wines, total sulfur dioxide accounts for
    {variance_share["total sulfur dioxide"]:.1%} of the summed variances and free
    sulfur dioxide for {variance_share["free sulfur dioxide"]:.1%}, so together
    they make up {_top_two:.1%} of the average unscaled squared distance between
    two wines. The other nine measurements share the remaining {1 - _top_two:.1%};
    the largest of them, {_rest.index[0]}, contributes only {_rest.iloc[0]:.2%},
    and alcohol only {variance_share["alcohol"]:.2%}.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3 Unscaled against scaled (8)
    """)
    return


@app.cell
def _(
    FEATURES,
    KNeighborsClassifier,
    StandardScaler,
    make_pipeline,
    test_wines,
    train_wines,
):
    # Fit k-NN with k = 5 to the training wines, once unscaled and once in a
    # pipeline that standardises first, and compute both accuracies on the test
    # wines.
    _X_train, _y_train = train_wines[FEATURES], train_wines["good"]
    _X_test, _y_test = test_wines[FEATURES], test_wines["good"]

    _unscaled = KNeighborsClassifier(n_neighbors=5).fit(_X_train, _y_train)
    unscaled_accuracy = float(_unscaled.score(_X_test, _y_test))

    _scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    _scaled.fit(_X_train, _y_train)
    scaled_accuracy = float(_scaled.score(_X_test, _y_test))
    unscaled_accuracy, scaled_accuracy
    return scaled_accuracy, unscaled_accuracy


@app.cell(hide_code=True)
def _(baseline, mo, scaled_accuracy, unscaled_accuracy, variance_share):
    _top_two = variance_share["total sulfur dioxide"] + variance_share["free sulfur dioxide"]
    mo.md(rf"""
    Unscaled, k-NN with k = 5 is right about {unscaled_accuracy:.1%} of the test
    wines; standardised in a pipeline, it is right about {scaled_accuracy:.1%}, a
    gain of {(scaled_accuracy - unscaled_accuracy) * 100:.1f} percentage points
    (the baseline is {baseline:.1%}). Section 2.2 explains why. Without scaling,
    {_top_two:.1%} of the distance between two wines comes from the two sulfur
    dioxide measurements, so the "nearest" wines are simply the ones with the most
    similar sulfur dioxide, and the other nine measurements, including alcohol and
    volatile acidity, count for almost nothing. Standardising gives every
    measurement a standard deviation of 1, so each of the eleven contributes
    equally to the distance and the neighbours are wines that are similar in
    their whole chemistry, which predicts the rating better.
    """)
    return


@app.cell(hide_code=True)
def _(
    FEATURES,
    pd,
    scaled_accuracy,
    test_wines,
    train_wines,
    unscaled_accuracy,
    variance_share,
):
    def test_part_2():
        assert isinstance(train_wines, pd.DataFrame) and isinstance(test_wines, pd.DataFrame), (
            "`train_wines` and `test_wines` should be DataFrames."
        )
        assert len(test_wines) == 400, (
            "The test set should have 400 wines, a quarter of 1,599 rounded up. Check `test_size`."
        )
        assert isinstance(variance_share, pd.Series), "`variance_share` should be a Series."
        assert sorted(variance_share.index) == sorted(FEATURES), (
            "`variance_share` should have one value for each of the eleven measurements."
        )
        assert abs(variance_share.sum() - 1) < 1e-6, (
            "The shares should add up to 1. Divide each variance by the sum of all eleven."
        )
        assert variance_share.is_monotonic_decreasing, "Sort `variance_share` from largest to smallest."
        for _name, _value in [("unscaled_accuracy", unscaled_accuracy),
                              ("scaled_accuracy", scaled_accuracy)]:
            assert isinstance(_value, float) and 0 <= _value <= 1, (
                f"`{_name}` should be an accuracy between 0 and 1."
            )

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 3: Choosing k (20 points)

    ### 3.1 Cross-validate (10)
    """)
    return


@app.cell
def _(
    FEATURES,
    KNeighborsClassifier,
    K_VALUES,
    StandardScaler,
    cross_val_score,
    make_pipeline,
    pd,
):
    def cross_validate_k(train_df):
        """Five-fold CV of the scaling pipeline on `train_df`, for every k in K_VALUES."""
        _rows = []
        for _k in K_VALUES:
            _pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=_k))
            _scores = cross_val_score(_pipeline, train_df[FEATURES], train_df["good"], cv=5)
            _rows.append({"k": _k,
                          **{f"fold {_i}": _s for _i, _s in enumerate(_scores, start=1)},
                          "mean": _scores.mean()})
        return pd.DataFrame(_rows).set_index("k")

    return (cross_validate_k,)


@app.cell
def _(cross_validate_k, train_wines):
    # Cross-validate the scaling pipeline with five folds on the training wines,
    # for every k in K_VALUES. Index `cv_table` by k, with a column `mean`.
    cv_table = cross_validate_k(train_wines)
    cv_table.round(3)
    return (cv_table,)


@app.cell
def _(baseline, cv_table, plt):
    # Plot the mean accuracy against k.
    _fig, _ax = plt.subplots(figsize=(7, 3.5))
    _ax.plot(cv_table.index, cv_table["mean"], marker="o", color="black",
             label="mean cross-validation accuracy")
    _ax.axhline(baseline, color="grey", linestyle="--", label="baseline")
    _ax.set_xlabel("k")
    _ax.set_ylabel("Accuracy")
    _ax.set_title("Cross-validated accuracy of the scaled k-NN pipeline")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Choose and test once (10)
    """)
    return


@app.cell
def _(
    FEATURES,
    KNeighborsClassifier,
    StandardScaler,
    make_pipeline,
    pd,
    test_wines,
    train_wines,
):
    def fit_and_test(train_df, test_df, k):
        """Fit the scaling pipeline with k to `train_df` and return its accuracy on `test_df`."""
        _pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        _pipeline.fit(train_df[FEATURES], train_df["good"])
        return float(_pipeline.score(test_df[FEATURES], test_df["good"]))

    return (fit_and_test,)


@app.cell
def _(cv_table, fit_and_test, test_wines, train_wines):
    # Fit the pipeline with `best_k` to all the training wines, and compute its
    # accuracy on the test wines.
    best_k = int(cv_table["mean"].idxmax())
    final_accuracy = fit_and_test(train_wines, test_wines, best_k)
    best_k, final_accuracy
    return best_k, final_accuracy


@app.cell(hide_code=True)
def _(baseline, best_k, cv_table, final_accuracy, mo, test_wines, train_wines):
    _cv_best = cv_table.loc[best_k, "mean"]
    _runner_up = cv_table["mean"].drop(best_k).idxmax()
    mo.md(rf"""
    I chose k = {best_k} because it has the highest mean cross-validated accuracy
    in `cv_table`, {_cv_best:.1%}, ahead of k = {_runner_up}
    ({cv_table.loc[_runner_up, "mean"]:.1%}); the choice uses the training wines
    only. Fitted to all {len(train_wines):,} training wines and tested once, the pipeline is
    right about {final_accuracy:.1%} of the {len(test_wines)} test wines. That is
    {(final_accuracy - baseline) * 100:.1f} percentage points above the
    {baseline:.1%} baseline, and within {abs(final_accuracy - _cv_best) * 100:.1f}
    points of the cross-validated estimate for the same k, so the two estimates
    agree. It is surprising, though, that a single nearest neighbour beats
    averaging over dozens of wines, since k = 1 usually overfits; Part 4 shows why
    it scores so well here.
    """)
    return


@app.cell(hide_code=True)
def _(K_VALUES, best_k, cv_table, final_accuracy, pd):
    def test_part_3():
        assert isinstance(cv_table, pd.DataFrame), "`cv_table` should be a DataFrame."
        assert list(cv_table.index) == K_VALUES, (
            "`cv_table` should have one row for each k in K_VALUES, indexed by k, in order."
        )
        assert "mean" in cv_table.columns, "`cv_table` needs a `mean` column."
        assert best_k in K_VALUES, "`best_k` should be one of the values in K_VALUES."
        assert isinstance(final_accuracy, float) and 0 <= final_accuracy <= 1, (
            "`final_accuracy` should be an accuracy between 0 and 1."
        )

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 4: Duplicates (25 points)

    ### 4.1 Find them (6)
    """)
    return


@app.cell
def _(wines):
    # Count the rows that are identical, in every column, to an earlier row. Then
    # show two or three groups of identical rows.
    n_duplicates = int(wines.duplicated().sum())
    duplicate_group_sizes = (wines[wines.duplicated(keep=False)]
                             .groupby(list(wines.columns)).size())
    n_duplicates, len(duplicate_group_sizes), duplicate_group_sizes.value_counts().sort_index()
    return duplicate_group_sizes, n_duplicates


@app.cell
def _(wines):
    # Three groups of identical rows, with their row numbers in the file.
    _all_copies = wines[wines.duplicated(keep=False)]
    _first_rows = _all_copies.drop_duplicates().index[:3]
    _groups = [_all_copies[(_all_copies == wines.loc[_i]).all(axis=1)] for _i in _first_rows]
    wines.loc[[_i for _g in _groups for _i in _g.index]]
    return


@app.cell(hide_code=True)
def _(duplicate_group_sizes, mo, n_duplicates, wines):
    _sizes = duplicate_group_sizes.value_counts().sort_index()
    mo.md(rf"""
    {n_duplicates} rows ({n_duplicates / len(wines):.1%} of the file) are identical,
    in all twelve columns, to an earlier row. They fall into
    {len(duplicate_group_sizes)} groups: {_sizes.get(2, 0)} pairs,
    {_sizes.get(3, 0)} groups of three and {_sizes.get(4, 0)} groups of four.

    I think they are the same wine recorded more than once, not different wines
    that happen to match. Eleven laboratory measurements, several to three or five
    decimal places (density to 0.00001 g/cm³), agreeing exactly by coincidence is
    implausible, and the tasters' rating matches too. The copies also sit close
    together in the file, as in the groups above, where they are only a few rows
    apart, which is what you would expect if the same sample or batch were entered
    twice, or analysed and logged again, rather than if two separate wines were
    identical.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 Effect on k = 1 (9)
    """)
    return


@app.cell
def _(test_wines, train_wines):
    # Count the test wines that have an identical wine in the training set.
    _columns = list(train_wines.columns)
    _train_rows = set(train_wines[_columns].itertuples(index=False, name=None))
    test_has_twin = test_wines[_columns].apply(tuple, axis=1).isin(_train_rows)
    n_test_with_twin = int(test_has_twin.sum())
    n_test_with_twin, len(test_wines)
    return (n_test_with_twin,)


@app.cell(hide_code=True)
def _(cv_table, final_accuracy, mo, n_test_with_twin, test_wines):
    mo.md(rf"""
    {n_test_with_twin} of the {len(test_wines)} test wines
    ({n_test_with_twin / len(test_wines):.0%}) have an identical copy in the
    training set. The same thing happens inside cross-validation: when the copies
    of a wine land in different folds, the validation wine has its own copy among
    the training folds.

    A copy is at distance 0, so at k = 1 it is always the nearest neighbour, and
    it has the same rating, so k-NN gets that wine right without having learned
    anything about chemistry; it has simply looked up the answer. That is why
    `cv_table` peaks at k = 1 ({cv_table.loc[1, "mean"]:.1%}) and falls at k = 3
    ({cv_table.loc[3, "mean"]:.1%}) and k = 5 ({cv_table.loc[5, "mean"]:.1%}),
    the opposite of the usual pattern, in which k = 1 overfits. As k grows, the
    one copy is outvoted by the other neighbours, so the advantage fades and the
    large-k accuracies are much less affected.

    The test set is not protected either: because the duplicates were split
    before we looked for them, a quarter of the "unseen" test wines were in fact
    seen during fitting, and the test accuracy of {final_accuracy:.1%} at k = 1 is
    inflated in the same way. Both estimates overstate how well the model would
    do on a genuinely new wine, and they agree with each other only because both
    are inflated by the same leak.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.3 Without the duplicates (10)
    """)
    return


@app.cell
def _(cross_validate_k, fit_and_test, train_test_split, wines):
    # Drop the duplicates, keeping the first of each group. Then repeat 2.1, 3.1
    # and 3.2 on `wines_unique`, with the same split settings.
    wines_unique = wines.drop_duplicates(keep="first")
    train_unique, test_unique = train_test_split(
        wines_unique, test_size=0.25, stratify=wines_unique["good"], random_state=618)
    cv_table_unique = cross_validate_k(train_unique)
    best_k_unique = int(cv_table_unique["mean"].idxmax())
    final_accuracy_unique = fit_and_test(train_unique, test_unique, best_k_unique)
    baseline_unique = float(wines_unique["good"].value_counts(normalize=True).max())
    len(wines_unique), len(train_unique), len(test_unique), best_k_unique, final_accuracy_unique
    return (
        baseline_unique,
        best_k_unique,
        cv_table_unique,
        final_accuracy_unique,
        wines_unique,
    )


@app.cell
def _(cv_table, cv_table_unique, plt):
    # Show both cross-validation curves on one plot, and the two baselines, the
    # two choices of k and the two test accuracies in one table.
    _fig, _ax = plt.subplots(figsize=(7, 3.5))
    _ax.plot(cv_table.index, cv_table["mean"], marker="o", label="with duplicates")
    _ax.plot(cv_table_unique.index, cv_table_unique["mean"], marker="s",
             label="duplicates removed")
    _ax.set_xlabel("k")
    _ax.set_ylabel("Mean cross-validation accuracy")
    _ax.set_title("Cross-validated accuracy, with and without duplicates")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    baseline,
    baseline_unique,
    best_k,
    best_k_unique,
    cv_table,
    cv_table_unique,
    final_accuracy,
    final_accuracy_unique,
    pd,
    wines,
    wines_unique,
):
    comparison = pd.DataFrame(
        {"with duplicates": [len(wines), baseline, best_k,
                             cv_table.loc[best_k, "mean"], final_accuracy],
         "duplicates removed": [len(wines_unique), baseline_unique, best_k_unique,
                                cv_table_unique.loc[best_k_unique, "mean"],
                                final_accuracy_unique]},
        index=["wines", "baseline", "chosen k", "CV accuracy at chosen k", "test accuracy"])
    comparison.round(3)
    return


@app.cell(hide_code=True)
def _(
    baseline_unique,
    best_k,
    best_k_unique,
    cv_table,
    cv_table_unique,
    final_accuracy,
    final_accuracy_unique,
    mo,
):
    _tied = [_k for _k in cv_table_unique.index
             if _k != best_k_unique
             and abs(cv_table_unique.loc[_k, "mean"] - cv_table_unique.loc[best_k_unique, "mean"]) < 1e-9]
    _tie_note = (f" (k = {', '.join(map(str, _tied))} has exactly the same mean, and I"
                 f" took the smaller k, the simpler of the tied models)" if _tied else "")
    mo.md(rf"""
    With the duplicates removed before splitting, the shape of the curve changes:
    k = 1 drops from {cv_table.loc[1, "mean"]:.1%} to
    {cv_table_unique.loc[1, "mean"]:.1%} and is now among the worst choices, while
    the large values of k barely move (k = 61: {cv_table.loc[61, "mean"]:.1%}
    against {cv_table_unique.loc[61, "mean"]:.1%}). Cross-validation now chooses
    k = {best_k_unique} instead of k = {best_k}{_tie_note}, and that model is right about
    {final_accuracy_unique:.1%} of the test wines, against a baseline of
    {baseline_unique:.1%}.

    I would report {final_accuracy_unique:.1%}, from the data without duplicates.
    The {final_accuracy:.1%} from the original data is not a fair estimate: k = 1
    was chosen because copies of the validation wines were in the training folds,
    and it scored well on the test set because a quarter of the test wines had a
    copy in the training set. A new wine from the winery would not have an exact
    copy among the wines already rated, so only the de-duplicated split measures
    what the winemaker would actually get. The two numbers happen to be close, but
    the reported model is also different (k = {best_k_unique}, not k = {best_k}),
    and only the second one was chosen without the leak.
    """)
    return


@app.cell(hide_code=True)
def _(
    K_VALUES,
    best_k_unique,
    cv_table_unique,
    final_accuracy_unique,
    n_duplicates,
    np,
    pd,
    wines,
    wines_unique,
):
    def test_part_4():
        assert isinstance(n_duplicates, (int, np.integer)), (
            "`n_duplicates` should be a single whole number."
        )
        assert isinstance(wines_unique, pd.DataFrame), "`wines_unique` should be a DataFrame."
        assert not wines_unique.duplicated().any(), "`wines_unique` still contains duplicate rows."
        assert len(wines_unique) == len(wines) - n_duplicates, (
            "`wines_unique` should have one row fewer than `wines` for each duplicate."
        )
        assert isinstance(cv_table_unique, pd.DataFrame), "`cv_table_unique` should be a DataFrame."
        assert list(cv_table_unique.index) == K_VALUES, (
            "`cv_table_unique` should have one row for each k in K_VALUES, indexed by k, in order."
        )
        assert best_k_unique in K_VALUES, "`best_k_unique` should be one of the values in K_VALUES."
        assert isinstance(final_accuracy_unique, float) and 0 <= final_accuracy_unique <= 1, (
            "`final_accuracy_unique` should be an accuracy between 0 and 1."
        )

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 5: Summary for a winemaker (20 points)
    """)
    return


@app.cell(hide_code=True)
def _(baseline_unique, final_accuracy_unique, mo, n_duplicates, n_test_with_twin, scaled_accuracy, unscaled_accuracy):
    mo.md(rf"""
    - **Question:** Can eleven routine laboratory measurements of a red wine
      predict whether the tasters will rate it 6 or higher?
    - **Finding:** They can, partly: from the chemistry alone, the method picks
      the tasters' verdict correctly for about {final_accuracy_unique:.0%} of wines
      it has never seen, compared with {baseline_unique:.0%} for simply guessing
      "good" every time.
    - **Evidence:**
        - On wines held back for checking, the method was right
          {final_accuracy_unique:.1%} of the time, about
          {(final_accuracy_unique - baseline_unique) * 100:.0f} percentage points
          better than always guessing "good" ({baseline_unique:.1%}).
        - How the measurements are combined matters: when sulfur dioxide, which
          is recorded in much bigger numbers than the rest, was allowed to
          dominate, accuracy was only {unscaled_accuracy:.0%}; putting all eleven
          measurements on an equal footing raised it to {scaled_accuracy:.0%}.
        - {n_duplicates} of the 1,599 wines in the file were exact repeats of
          another wine, and {n_test_with_twin} of the 400 wines we first held back
          for checking had a copy among the wines used to build the method.
    - **Limits:** The method does not taste wine. It predicts what this panel of
      tasters said about wines with similar chemistry, so it can flag a wine as
      likely to rate well or poorly before tasting, but it cannot say why, cannot
      say what to change in the cellar, and should not be trusted for wines unlike
      these Portuguese reds. It is wrong about one wine in four, and its edge over
      guessing is about {(final_accuracy_unique - baseline_unique) * 100:.0f}
      points, useful for sorting but not a replacement for the tasters. The repeated
      wines mattered because a wine "checked" against its own copy is an answer
      looked up, not predicted; they first made the simplest version of the
      method look like the best one, so we removed them before measuring, and the
      {final_accuracy_unique:.0%} above is the honest figure.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## AI attribution

    I used Claude (Anthropic), through Claude Code, to write the code for every
    part and to draft the written answers, which refer to the numbers produced by
    this notebook. I checked the code, the outputs and the explanations against
    notebooks 08a and 08b.

    ---
    ## Before you submit

    1. Make sure that every cell has run successfully.
    2. Rename this file `SI618_HW05_<uniqname>.py`.
    3. Export it:
       ```bash
       uvx marimo export html --sandbox SI618_HW05_<uniqname>.py -o SI618_HW05_<uniqname>.html
       ```
    4. Upload both files to Canvas.
    """)
    return


if __name__ == "__main__":
    app.run()
