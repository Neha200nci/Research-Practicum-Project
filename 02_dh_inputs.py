"""Dunnhumby modelling inputs: segmentation features, alternatives, alternative-week table, choice occasions."""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
LOG = {}
EST_END, N_WEEKS = 75, 102

# ---------------- 1. household segmentation features
hh = pd.read_parquet(IN / "households.parquet")
FEATS = ["basket_value", "items_per_basket", "trips_per_shop_week", "pl_share", "deal_intensity", "promo_line_share", "price_tier", "mf_hhi"]
seg = hh[hh.eligible_seg].copy()
LOG["households_total"] = int(len(hh)); LOG["households_eligible_seg"] = int(len(seg)); LOG["with_demographics"] = int(hh.has_demo.sum())
LOG["eligible_with_demographics"] = int(seg.has_demo.sum())
seg["mf_hhi_imputed"] = seg.mf_hhi.isna()
seg["mf_hhi"] = seg.mf_hhi.fillna(seg.mf_hhi.median())
X = seg[FEATS].copy()
for c in ["basket_value", "items_per_basket", "trips_per_shop_week"]:
    X[c] = np.log1p(X[c])
lo, hi = X.quantile(0.01), X.quantile(0.99)
X = X.clip(lo, hi, axis=1)
mu, sd = X.mean(), X.std()
Z = ((X - mu) / sd).add_suffix("_z")
out = pd.concat([seg[["household_key", "has_demo", "mf_hhi_imputed", "home_store", "trips", "spend", "shop_weeks"] + FEATS], Z,
                 seg[["age_desc", "income_desc", "hh_comp_desc", "household_size_desc", "kid_category_desc", "homeowner_desc", "marital_status_code"]]], axis=1)
out.to_parquet(OUT / "hh_segmentation_features.parquet", index=False)
X.corr().round(2).to_csv(OUT / "hh_feature_correlations.csv")
json.dump({"features": FEATS, "log1p": ["basket_value", "items_per_basket", "trips_per_shop_week"],
           "winsor_lo": lo.to_dict(), "winsor_hi": hi.to_dict(), "mean": mu.to_dict(), "std": sd.to_dict()}, open(OUT / "hh_feature_scaler.json", "w"), indent=1)
LOG["feature_corr_max_offdiag"] = float(X.corr().where(~np.eye(len(FEATS), dtype=bool)).abs().max().max())

# ---------------- 2-4. per category
for cat in ["cereal", "soup"]:
    sm = pd.read_parquet(IN / f"sku_master_{cat}.parquet")
    sw = pd.read_parquet(IN / f"sku_week_{cat}.parquet")
    oc = pd.read_parquet(IN / f"occasions_{cat}.parquet")
    ln = pd.read_parquet(IN / f"lines_{cat}.parquet", columns=["household_key", "week_no", "product_id", "split", "quantity", "sales_value", "store_id"])
    # alternatives
    g = sm.groupby("alt_id")
    am = pd.DataFrame({"n_sku": g.size(), "is_tail": g.core.max().eq(False), "is_private": g.is_private.max(),
                       "mf_group": g.mf_group.first(), "sub_group": g.sub_group.first(),
                       "size_oz": g.apply(lambda d: np.average(d.size_oz.fillna(d.size_oz.median()), weights=d.est_lines + 1)),
                       "size_bucket": g.size_bucket.agg(lambda s: s.value_counts().index[0]),
                       "est_lines": g.est_lines.sum(), "total_lines": g.total_lines.sum(),
                       "first_week": g.first_week.min(), "last_week": g.last_week.max()}).reset_index()
    est_choice = oc[(oc.bought == 1) & (oc.split == "est")].chosen_alt_id.value_counts()
    hold_choice = oc[(oc.bought == 1) & (oc.split != "est")].chosen_alt_id.value_counts()
    am["est_choices"] = am.alt_id.map(est_choice).fillna(0).astype(int)
    am["hold_choices"] = am.alt_id.map(hold_choice).fillna(0).astype(int)
    am["new_in_holdout"] = am.est_lines == 0
    am["sparse_est"] = am.est_choices < 30
    am.to_parquet(OUT / f"alternatives_{cat}.parquet", index=False)

    # alternative x week grid (weeks 1..102): available if any member SKU is available that week
    sw = sw.merge(sm[["product_id"]], on="product_id")
    sw["alt_id"] = sw.alt_id
    sw["w"] = sw.units.fillna(0) + 1e-3
    a = sw[sw.available].copy()
    a["p_w"] = a.shelf_price_filled * a.w; a["p_ok"] = a.shelf_price_filled.notna() * a.w
    for c in ["promo_share", "disc_depth", "display_share", "mailer_share"]:
        a[c + "_w"] = a[c].fillna(0) * a.w
    ag = a.groupby(["alt_id", "week_no"]).agg(n_members_avail=("product_id", "size"), units=("units", "sum"), w=("w", "sum"),
                                              p=("p_w", "sum"), pok=("p_ok", "sum"), pr=("promo_share_w", "sum"), dd=("disc_depth_w", "sum"),
                                              di=("display_share_w", "sum"), ma=("mailer_share_w", "sum"), price_obs=("price_observed", "mean")).reset_index()
    ag["price"] = ag.p / ag.pok.replace(0, np.nan)
    ag["promo_share"] = ag.pr / ag.w; ag["disc_depth"] = ag.dd / ag.w; ag["display_share"] = ag.di / ag.w; ag["mailer_share"] = ag.ma / ag.w
    grid = pd.MultiIndex.from_product([am.alt_id, range(1, N_WEEKS + 1)], names=["alt_id", "week_no"]).to_frame(index=False)
    aw = grid.merge(ag[["alt_id", "week_no", "n_members_avail", "units", "price", "price_obs", "promo_share", "disc_depth", "display_share", "mailer_share"]],
                    on=["alt_id", "week_no"], how="left")
    aw["available"] = aw.n_members_avail.notna()
    aw["n_members_avail"] = aw.n_members_avail.fillna(0).astype(int)
    aw["split"] = np.where(aw.week_no <= EST_END, "est", "holdout")
    aw = aw.merge(am[["alt_id", "is_tail", "is_private", "size_oz"]], on="alt_id")
    aw.to_parquet(OUT / f"alt_week_{cat}.parquet", index=False)

    # purchase lines at alternative level (for models that use every purchase, not only the top-spend SKU)
    cl = ln.merge(sm[["product_id", "alt_id"]], on="product_id").groupby(["household_key", "week_no", "alt_id", "split"], as_index=False).agg(
        units=("quantity", "sum"), spend=("sales_value", "sum"), store_id=("store_id", "first"))
    cl.to_parquet(OUT / f"choice_lines_{cat}.parquet", index=False)
    # choice occasions: household-week with a basket; chosen = top-spend alternative or OUTSIDE
    o = oc.copy()
    o["chosen"] = np.where(o.bought == 1, o.chosen_alt_id, "OUTSIDE")
    o["multi_sku"] = o.n_skus > 1
    avail_n = aw[aw.available].groupby("week_no").size()
    o["n_available"] = o.week_no.map(avail_n).fillna(0).astype(int)
    key = set(zip(aw[aw.available].alt_id, aw[aw.available].week_no))
    o["chosen_available"] = [(c == "OUTSIDE") or ((c, w) in key) for c, w in zip(o.chosen, o.week_no)]
    o["hh_segmentable"] = o.household_key.isin(seg.household_key)
    o["store_eligible"] = o.store_id.map(pd.read_parquet(IN / "stores.parquet").set_index("store_id").eligible).fillna(False)
    keep = ["household_key", "week_no", "split", "store_id", "main_store_id", "store_eligible", "bought", "chosen", "n_skus", "multi_sku",
            "cat_spend", "cat_units", "week_spend", "week_baskets", "n_available", "chosen_available", "hh_segmentable"]
    o[keep].to_parquet(OUT / f"choice_occasions_{cat}.parquet", index=False)
    e = o[o.split == "est"]
    LOG[cat] = dict(
        alternatives=int(len(am)), core=int((~am.is_tail).sum()), tail=int(am.is_tail.sum()), new_in_holdout=int(am.new_in_holdout.sum()),
        sparse_alts_est_lt30_choices=int(am.sparse_est.sum()),
        occasions=int(len(o)), est_occasions=int(len(e)), hold_occasions=int((o.split != "est").sum()),
        est_category_choices=int(e.bought.sum()), hold_category_choices=int(o[o.split != "est"].bought.sum()),
        outside_share_est=float(1 - e.bought.mean()),
        multi_sku_share_of_choices=float(o[o.bought == 1].multi_sku.mean()),
        chosen_not_available=int((~o.chosen_available).sum()),
        mean_available_alts_per_week=float(avail_n.mean()), min_available=int(avail_n.min()),
        price_missing_available_alt_weeks=float(aw[aw.available].price.isna().mean()),
        occasions_segmentable_share=float(o.hh_segmentable.mean()), est_choices_segmentable=int(e[e.bought == 1].hh_segmentable.sum()),
        est_choices_segmentable_eligible_store=int(e[(e.bought == 1) & e.hh_segmentable & e.store_eligible].shape[0]),
        choice_lines=int(len(cl)), lines_per_choice_week=float(len(cl) / max(int(oc.bought.sum()), 1)),
        holdout_choices_on_new_alts=int(o[(o.split != "est") & o.chosen.isin(am[am.new_in_holdout].alt_id)].shape[0]),
    )
    print(cat, json.dumps(LOG[cat], indent=1))
json.dump(LOG, open(OUT / "inputs_log.json", "w"), indent=1, default=str)
