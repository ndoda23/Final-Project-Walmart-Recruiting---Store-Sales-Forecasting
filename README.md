# Final-Project-Store-Sales-Forecasting

ალეკო მამუკაშვილი , ნიკოლოზ დოდაშვილი.

competition-ის ლინკი: https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting

#  პროექტის მიმოხილვა: Walmart-ის მაღაზიების გაყიდვების პროგნოზირება

##  პროექტის მიზანი
პროექტის მთავარი ამოცანაა **Walmart-ის 45 სხვადასხვა მაღაზიის** კონკრეტული დეპარტამენტების ყოველკვირეული გაყიდვების ზუსტი პროგნოზირება. მოდელი ეყრდნობა ისტორიულ მონაცემებსა და რეგიონალურ გარემო პირობებს. მთავარ გამოწვევას წარმოადგენს გაყიდვების ფრედიქშენი სადღესასწაულო კვირებში, როდესაც მომხმარებელთა ქცევა მკვეთრად იცვლება.

##  მონაცემთა მახასიათებლები (Features)
მოდელი საპროგნოზოდ იყენებს შემდეგ მონაცემებს:
* **მაღაზიის და დეპარტამენტის ინფო:** უნიკალური იდენტიფიკატორები (ID).
* **დროითი მონაცემები:** ყოველკვირეული გაყიდვების ისტორია და სადღესასწაულო კვირის ნიშნული (`IsHoliday`).
* **მაკრო ფაქტორები:** ფასდაკლებები (Markdowns), რეგიონალური ტემპერატურა, საწვავის ფასი, CPI (სამომხმარებლო ფასების ინდექსი) და უმუშევრობის დონე.

##  შეფასების კრიტერიუმი (Evaluation)
მოდელის სიზუსტე ფასდება **შეწონილი საშუალო აბსოლუტური ცდომილებით (WMAE)**. ამ ფორმულის მიხედვით, სადღესასწაულო კვირებში დაშვებულ შეცდომას **5-ჯერ მეტი წონა** აქვს, ვიდრე ჩვეულებრივ კვირებში დაშვებულ შეცდომას:

$$WMAE = \frac{1}{\sum w_i} \sum_{i=1}^{n} w_i |y_i - \hat{y}_i|$$

* $w_i = 5$, თუ კვირა ემთხვევა დღესასწაულს
* $w_i = 1$, ნებისმიერ სხვა შემთხვევაში

# LightGBM 

MLflow (DagsHub): `LightGBM_Training` ექსპერიმენტი - [LightGBM MlFlow](https://dagshub.com/aleko-mamukashvili/Store-Sales-Forecasting.mlflow/#/experiments/2)

გადავწყვიტე LightGBM-ისთვის feature engineering-ის ყოველი ეტაპი მონაცემებზე დაკვირვებით და კონკრეტული სტატისტიკის საფუძველზე გამეკეთებინა ამ README-ში აღვწერ თუ რას ვხედავდი EDA-ში, რა გადაწყვეტილებას ვიღებდი ამის საფუძველზე, და რა შედეგი მოჰყვა ყოველ ცვლილებას.

---

## EDA - გადაწყვეტილებები მონაცემების საფუძველზე

### Missing Values

`MarkDown1-5` სვეტებში 60-70%-მდე NaN აღმოვაჩინე:

![Missing Values](images/missing_values.png)

**გადაწყვეტილება:** ეს არ არის random-missing, Walmart-მა MarkDown-ების ჩაწერა მხოლოდ 2011 წლის ნოემბრიდან დაიწყო. ამიტომ NaN ნიშნავს, რომ ამ კვირას promo არ ყოფილა და არა მონაცემების დაკარგვას ამიტომ ისინი შევავსე `0`-ით და არა median/mean-ით.

### Holiday Spikes

საერთო კვირეული გაყიდვების ტრენდი

![Holiday Spikes](images/holiday_spikes.png)

**გადაწყვეტილება:** spike-ები ცალსახად ჩანს Thanksgiving/Christmas-თან რაც ადასტურებს, რატომ სჭირდება Kaggle-ის WMAE-ს holiday-კვირების 5x წონა, და რატომ გამოვიყენე იგივე წონა sample_weight-ად ტრენინგშიც (არა მხოლოდ შეფასებაში).

### Lag-ის სიგრძის დასაბუთება

Store 1 / Dept 1-ის მიმდინარე გაყიდვები vs 52 კვირით ადრინდელი:

![Lag Correlation](images/lag_correlation.png)

Pearson r = **0.573** — საკმაოდ ძლიერი წლიური სეზონურობა.

**გადაწყვეტილება:** ავირჩიე lag = **51, 52, 53 კვირა** (არა მხოლოდ 52), რადგან კვირის ნომრები წლიდან წლამდე ოდნავ იცვლის კალენდარული shift-ის გამო

### Store Type & Size

![Type and Size](images/type_size.png)

**გადაწყვეტილება:** Type A > B > C ცალსახად, Size-თან დადებითი კავშირიც ჩანს.

### Correlation - გარე ეკონომიკური ცვლადები

![Correlation Matrix](images/correlation_heatmap.png)

`Temperature`, `Fuel_Price`, `CPI`, `Unemployment` თითქმის ნულოვან წრფივ კორელაციას აჩვენებენ `Weekly_Sales`-თან.

**გადაწყვეტილება:** თავიდან **არ** ამოვშალე (LightGBM-ს შეუძლია non-linear ურთიერთქმედების დაჭერა), მაგრამ მოგვიანებით ცალკე ablation-ით შევამოწმე ეს ვარაუდი 

---

## Train/Validation Split - კრიტიკული გაკვეთილი

პირველი მცდელობისას ვალიდაციად ავიღე უბრალოდ `train.csv`-ის ბოლო 12 კვირა. შედეგად WMAE ≈ **1570**  იყო არარეალურად კარგი 

**ანალიზის შედეგი:** `train.csv` მთლიანად მთავრდება 2012 წლის ოქტომბერში, Thanksgiving-ისა და Christmas 2012-მდე. ჩემი ბოლო 12 კვირა validation შემთხვევით გამორიცხავდა ზუსტად იმ ორ ყველაზე რთულ, უმაღლეს ვარიაციულ holiday-კვირას, რომლებზეც რეალურად ფასდება მოდელი Kaggle-ზე.

**გადაწყვეტილება:** ვალიდაციის window შევცვალე ისე, რომ სავალდებულოდ მოიცავდეს Thanksgiving/Christmas/Super Bowl პერიოდს:

```
Train:  2010-02-05 → 2011-10-31
Val:    2011-11-01 → 2012-02-15  (მოიცავს Thanksgiving 2011, Christmas 2011, Super Bowl 2012)
```

ამ ცვლილების შემდეგ იგივე feature-set-ის WMAE **1570-დან 2831-მდე** ავიდა.

---


## Feature Engineering 

ყველა transformer ცალკე კლასია და ერთიან `Pipeline`-შია ჩაწყობილი:

- **`LagFeatureBuilder`** — 51/52/53-კვირიანი lag, ვექტორიზებული merge-ით 
- **`GroupStatsFeatureBuilder`** — (Store, Dept) დონეზე Mean/Median/Std
- **`TemporalFeatureBuilder`** — Year/Month/WeekOfYear/IsHoliday/Type (ordinal)
- **`LGBMFinalEstimator`** — `lgb.train()`-ის sklearn-wrapper, პაიფლაინის ბოლო element


### LagFeatureBuilder - ყველაზე მნიშვნელოვანი feature

**რატომ 51/52/53 კვირა და არა ზუსტად 52 :** კვირის ნომერი კალენდარულად ოდნავ გადაადგილდება წლიდან წლამდე ამიტომ ზუსტად ერთი წლის წინანდელი row ხანდახან 51-ე ან 53-ე კვირაზეც მოხვდება. სამივე კვირა მოდელს არჩევანის საშუალებას აძლევს რომ  ერთ fixed წერტილზე არ იყოს დამოკიდებული.


### GroupStatsFeatureBuilder - target encoding

ეს ფაქტობრივად **target encoding**-ის ერთი ფორმაა - (Store, Dept) წყვილს ვანიჭებთ მისივე ისტორიული target-ის სტატისტიკას.

**საინტერესო აღმოჩენა:** Group Stats-ის Lag-ზე დამატებამ სინამდვილეში ოდნავ **გააუარესა** შედეგი (Lag-only: WMAE=2652 → +Group: WMAE=2831). ორივეს ერთად ყოფნა ნაწილობრივ redundant აღმოჩნდა ამ კონკრეტულ feature-set-ში.

### TemporalFeatureBuilder - თარიღიდან რიცხვითი feature-ები

 `Type`-ის ordinal encoding (A→0, B→1, C→2) საკმარისი აღმოჩნდა tree-based მოდელისთვის  One-Hot-თან შედარებითი ტესტიც ჩავატარეთ  და ordinal ოდნავ სჯობდა კიდეც.


---

## ექსპერიმენტები

### Run 1 — Baseline (მხოლოდ temporal features)

```
wmae_val: 5893.29
```

![Run 1](images/run1_baseline_actual_vs_pred.png)


### Run 2 - + Lag Features

```
wmae_val: 2651.87   
```

![Run 2](images/run2_lag_actual_vs_pred.png)

**დაკვირვება:** ეს არის ყველაზე დიდი ცალკეული გაუმჯობესება მთელ პროექტში  holiday spike-ები ახლა ბევრად უკეთაა დაჭერილი.

### Run 3 - + Group Statistics

```
wmae_val: 2830.98
```

![Run 3](images/run3_group_actual_vs_pred.png)

**დაკვირვება:**  Group Stats-ის დამატებამ ცოტათი **გააუარესა** შედეგი (Lag-only იყო 2651.87). 

Feature importance ადასტურებს, რომ lag/group ორივე რეალურად გამოიყენება:

![Feature Importance](images/feature_importance.png)

### Cross-Validation - Rolling-window (3 fold, 8-კვირიანი)

```
Fold 1 (2011-05-13 → 07-08): WMAE = 1932.4
Fold 2 (2011-07-08 → 09-02): WMAE = 1833.6
Fold 3 (2011-09-02 → 10-28): WMAE = 1624.7
mean = 1796.9  ±  128.3
```

![CV Results](images/cv_results.png)

**შენიშვნა:** ეს fold-ები (მაისი-ოქტომბერი) თავად არ მოიცავს holiday-კვირას, ამიტომ CV-ს საშუალო  დაბალია holiday-inclusive validation-თან  შედარებით. ეს დამატებით ადასტურებს, რომ holiday-კვირები რეალურად ყველაზე რთული პროგნოზირებადია.

### Hyperparameter Search

```
lr=0.05, depth=-1, 300 rounds:  wmae_val = 3095.5
lr=0.05, depth=6,  300 rounds:  wmae_val = 3083.5  
lr=0.03, depth=8,  500 rounds:  wmae_val = 3123.4
```

![HPO Comparison](images/hpo_comparison.png)

**დაკვირვება:** HPO-ს ყველა კონფიგურაცია **უარესია**, ვიდრე Run 3-ის default. ბრმა hyperparameter search ამ feature-set-ზე ვერ სჯობნის default-ს.

---

## დამატებითი ექსპერიმენტები (Ablations, Control, Sanity Checks)

| Run | WMAE (val) | დასკვნა |
|---|---|---|
| Explicit MarkDown Imputation | 2831.0 | ზუსტად იგივე, რაც native NaN-handling, MarkDown feature-ები საერთოდ არ გამოიყენება split-ებში |
| Total MarkDown (1 sum-column) | 2831.0 | იგივე მიზეზით, 0 გავლენა |
| Type One-Hot (ordinal-ის ნაცვლად) | 2998.6 | ოდნავ უარესი. ordinal საკმარისია tree-ებისთვის |
| Log-Transform Target (signed-log1p) | 2845.7 | თითქმის არაფერი შეიცვალა, tree-ებს target-ის მონოტონური ტრანსფორმაცია არ სჭირდება |
| Rolling Features (4w/8w mean/std) | 4529.5 | **გააუარესა**  |
| **Drop Weak External Features** | **2746.5** | **გააუმჯობესა** — Temperature/Fuel_Price/CPI/Unemployment სინამდვილეში noise-ს უფრო მატებდნენ, ვიდრე სიგნალს |
| Unweighted Training (control) | 3123.6 | გაუარესდა, როგორც მოსალოდნელი იყო. ადასტურებს holiday-წონის საჭიროებას |
| Shuffled Target (sanity check) | 14909.6 | ჩავარდა, როგორც უნდა ჩავარდნილიყო|
| 5%-იანი მონაცემი (data-quantity ტესტი) | 4705.9 | მკვეთრად უარესი. სრული history რეალურად საჭიროა |
| Extreme Overfit (depth=20, 1000 rounds) | 2571.1 | საუკეთესო რიცხვი, მაგრამ **არ ავირჩიე production-ისთვის**. overfitting gap  |
| Huber Loss Objective | 16392.8 |  default `alpha` არ იყო მორგებული Weekly_Sales-ის მასშტაბზე  |

![Experiment Summary](images/experiment_summary.png)

---

## საბოლოო მოდელი

ყველა ლეგიტიმური  კონფიგურაცია შევადარე ერთმანეთს, მათ შორის Section-ის ablation-ებიც, არა მხოლოდ HPO grid. საბოლოოდ არჩეულია **`DropWeakExternalFeatures`** კონფიგურაცია (lag + group stats + temporal, external weak features მოცილებული, default hyperparameters):

```
wmae_val: 2746.5
```

დარეგისტრირებულია MLflow Model Registry-ში, როგორც `LightGBM_WalmartSales`.

## Kaggle Submission

`model_inference.ipynb`-ით გენერირებული submission-ის რეალური Kaggle score:

```
Public Score:   2983.93
Private Score:  3161.70
```


![Submission](images/lightGBM_submission.png)
