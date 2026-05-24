# Karštųjų subnykštukių dvinarių sistemų klasifikavimas naudojant Gaia DR3 XP spektrų duomenis

Bakalauro darbo (Duomenų mokslas, VU MIF) kodo aplankas. 
Projekto tikslas —klasifikuoti karštųjų subnykštukių (hot subdwarf) žvaigždžių dvinares sistemas
naudojant Gaia DR3 BP/RP spektrų bazinių funkcijų plėtinius (Chebyshev, Legendre,
B-spline) bei mašininio mokymo metodus. 
Etaloninis darbas: Ambrosch et al. (2026),
*A&A*.

## Apimtis

Šiame aplanke pateikiama tik pagrindinė BP/RP spektrų bazinių funkcijų
plėtinių eksperimentų dalis:

- įvesčių paruošimas iš Gaia DR3 XP koeficientų;
- BP ir RP spektrų bazinių funkcijų plėtinių požymių generavimas;
- (K_BP × K_RP) gardelės klasifikavimo eksperimentai (LR, RF, SVM, XGBoost);
- tiriamoji duomenų analizė (EDA);
- B-spline bazės multikolinearumo analizė;
- pagrindinių rezultatų vizualizacijos.


## Reikalavimai

- Python ≥ 3.10
- Priklausomybės nurodytos `requirements.txt`

## Diegimas

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Struktūra

| Failas | Paskirtis |
|---|---|
| `01_prepare_inputs.py` / `.ipynb` | Įvesties duomenų paruošimas |
| `02_generate_basis_features.py` / `.ipynb` | Bazinių funkcijų plėtinių požymių generavimas |
| `08_hpo_preliminary.py` | Preliminari hiperparametrų paieška |
| `08_kbp_krp_grid_{lr,rf,svm,xgb}.py` | Pagrindiniai (K_BP × K_RP) gardelės eksperimentai |
| `smoke_07_kbp_krp_grid.py` | Greitas tikrinimo paleidimas (smoke test) |
| `eda_figures.ipynb` | Žvalgomosios analizės paveikslai |
| `12_bspline_multicollinearity.ipynb` | B-spline bazės multikolinearumo analizė |
| `rezultatu_vizualizacijos.ipynb` | Galutinių rezultatų vizualizacijos |
| `data/` | Pagrindiniai įvesties failai |

## Būtini duomenų failai

Aplanke `data/` turi būti:

- `bp_sampled_spectra.csv` — BP spektrai
- `rp_sampled_spectra.csv` — RP spektrai
- `splits.json` — pagrindinis kryžminio patikrinimo skaidinys
- `splits_rskf.json` — pakartotinio stratifikuoto K-fold skaidinys (50 padalijimų)

## Vykdymo seka

1. *(neprivaloma)* `01_prepare_inputs.py`  — jei reikia persigeneruoti įvestis.
2. `02_generate_basis_features.py` — bazinių funkcijų plėtinių požymių paruošimas.
3. `08_hpo_preliminary.py` — preliminari hiperparametrų paieška.
4. Pagrindiniai (K_BP × K_RP) gardelės eksperimentai:
   - `08_kbp_krp_grid_lr.py`
   - `08_kbp_krp_grid_rf.py`
   - `08_kbp_krp_grid_svm.py`
   - `08_kbp_krp_grid_xgb.py`
5. Analizė užrašinėse:
   - `eda_figures.ipynb`
   - `12_bspline_multicollinearity.ipynb`
   - `rezultatu_vizualizacijos.ipynb`

Greitas patikrinimas prieš pilną paleidimą:

```bash
python smoke_07_kbp_krp_grid.py --clf LR --basis chebyshev
```


## Literatūra

- Ambrosch, M. et al. (2026). *Detection of hot subdwarf binaries and sdB stars
  using machine learning methods and a large sample of Gaia XP spectra.*
  A&A (accepted).
- Gaia Collaboration; Vallenari, A. et al. (2023). *Gaia Data Release 3.* A&A 674, A1.
