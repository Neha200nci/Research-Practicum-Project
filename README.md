[README upload.md](https://github.com/user-attachments/files/34753722/README.upload.md)
# Which products should a supermarket remove? Using shopper segments to decide

This is my MSc Data Analytics research project at the National College of Ireland.

**Student:** Neha Amudapuram (25109936)

## What the project is about

Supermarkets sometimes need to remove products from their shelves. The simple way is to remove the products that sell the least. But when a product is removed, some shoppers switch to another product and some just stop buying. Different types of shoppers might react differently.

In this project I group shoppers into types (segments) based on how they shop, and I check whether this helps decide which products to remove. I use the Dunnhumby *Complete Journey* dataset (2,500 households over 102 weeks), and I focus on breakfast cereal.

## My research questions

1. **RQ1:** Can a model that knows the shopper segments predict what people buy better than one model for everyone?
2. **RQ2:** Does using segments change which products should be removed, and does it give better revenue or profit?

## What I have done so far

| Step | File | Status |
|---|---|---|
| Clean the data and explore it | `01_Dunnhumby_EDA_DataPrep.ipynb` | Done |
| Prepare the data for modelling | `02_dh_inputs.py` | Done |
| Group households into segments | `03_Household_Segmentation.ipynb` | Done |
| Build the models | `04_Fit_Choice_Models.ipynb` | Done |
| Test the models on unseen weeks (RQ1) | `05_RQ1_Holdout_Comparison.ipynb` | Done |
| See where shoppers go when a product is removed | `06_Demand_Transfer.ipynb` | Done |
| Make lists of products to remove (RQ2) | `07_Removal_Lists.ipynb` | Done |
| Work out the profit of each list | `08_Profit.ipynb` | Done |
| Store-level analysis | – | To do |
| Check against products that were really removed | – | To do |

The notebooks are saved with their results, so you can open them on GitHub and see the tables and charts without running anything.

## How I did it

- **Time split:** I used weeks 1–75 to build everything, and kept weeks 76–102 aside to test the models at the end.
- **Products:** 207 main cereals, plus 20 groups of cereals that are rarely bought (227 options in total).
- **One choice:** one household in one shopping week. If they bought cereal, I take the cereal they spent the most on. If they shopped but didn't buy cereal, that counts as "no cereal".
- **Segments:** I used k-means to group 2,039 households into 4 segments using seven shopping habits, like basket size, how often they shop, how much own-brand they buy and how much they buy on offer. I didn't use cereal purchases for this, so the segments are not built from the same choices I later try to predict.
- **The model** answers two questions every week:
  1. Will this household buy cereal this week?
  2. If yes, which cereal will they buy (based on price, discounts and display)?
- **Three versions of the model:**
  - one model for everyone (pooled)
  - a model where each segment reacts differently (segment model)
  - the segment model with the segments shuffled at random (placebo), to check that any improvement is real and not just luck
- **Removal lists:** I made three lists of 21 products to remove: the lowest-selling products, the pooled model's choice and the segment model's choice. I checked each list with both models.
- **Profit:** Dunnhumby has no cost data, so I assumed margins of 35% for own-brand products and 26% for branded ones.

## What I found so far

**RQ1: yes.** On the weeks the models had never seen, the segment model predicted which cereal people bought better than the pooled model. The shuffled version did not improve anything, so the improvement comes from real differences between shoppers.

| Test weeks 76–102 | Pooled | Segment | Placebo |
|---|---|---|---|
| Prediction score per purchase (higher is better) | −4.957 | −4.912 | −4.958 |
| Bought product was in the model's top 5 | 13.2% | 14.5% | 13.3% |

**RQ2: not much difference.** The pooled model and the segment model picked almost the same products to remove (20 out of 21 were the same).

| Removal list | Revenue change | Profit change |
|---|---|---|
| Lowest-selling products | +0.1% to +0.2% | −1.0% |
| Pooled model | +2.0% to +2.3% | −5.9% to −6.2% |
| Segment model | +2.0% to +2.3% | −4.7% to −5.0% |

Both model lists mostly removed cheap own-brand cereals. Their buyers switched to more expensive brands, so revenue went up. But own-brand products make more profit per sale, so profit went down. The segment list lost a bit less profit than the pooled list.

Charts:
- [`outputs/rq1_holdout/rq1_gain_over_pooled.png`](outputs/rq1_holdout/rq1_gain_over_pooled.png)
- [`outputs/removal_lists/rq2_removal_lists.png`](outputs/removal_lists/rq2_removal_lists.png)
- [`outputs/profit/profit_by_list.png`](outputs/profit/profit_by_list.png)

## What is in this repository

| Folder or file | What's inside |
|---|---|
| Notebooks `01`–`08` and `02_dh_inputs.py` | All my code, in the order I ran it |
| `figures/` | Charts from exploring the data |
| `step7_outputs/` | Segment results and charts |
| `outputs/` | Model results, tables and charts from notebooks 04–08 |
| `extras_dominicks/` | Some early work on the Dominick's dataset (not used in the current analysis) |
| `raw/` | Empty folder for the data |

**The data is not included.** The Dunnhumby dataset belongs to dunnhumby, so I haven't uploaded it or any files made from it at household level. You can download it yourself (see below).

## How to run it

1. Download *The Complete Journey* from the [dunnhumby website](https://www.dunnhumby.com/source-files/) (it's also on Kaggle as "dunnhumby – The Complete Journey"). Put the 8 CSV files in the `raw/` folder.
2. Install the packages:
   ```
   pip install -r requirements.txt
   ```
3. Open Jupyter from this folder and run the files in order:
   - `01_Dunnhumby_EDA_DataPrep.ipynb` (this one takes a few minutes)
   - `python 02_dh_inputs.py data/processed modelling_inputs_dunnhumby`
   - notebooks `03` to `08` (each takes less than a minute)

## Limitations

- If a household bought more than one cereal in a week, I only use the one they spent the most on.
- I can only tell whether a product was available across the whole chain, not in each store.
- The profit results depend on the margins I assumed.
- The removal results are what the model predicts, not what actually happened.
