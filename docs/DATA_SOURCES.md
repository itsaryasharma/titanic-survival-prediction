# Data Sources: Titanic Survival Prediction

| | |
|---|---|
| Dataset | `titanic3` |
| Publisher | Vanderbilt University Department of Biostatistics (Frank Harrell) |
| URL | https://hbiostat.org/data/repo/titanic3.csv (description: https://hbiostat.org/data/repo/titanic.html) |
| Provenance | "Thomas Cason of UVa has greatly updated and improved the titanic data frame using the Encyclopedia Titanica and created a new dataset called titanic3." Earlier sources include Eaton & Haas (1994), *Titanic: Triumph and Tragedy* |
| Size | 1,309 passengers × 14 columns (crew not included) |
| Terms | "Permission is granted to anyone wishing to use the data sets provided here. Please reference the original paper … and note '**Data obtained from http://hbiostat.org/data courtesy of the Vanderbilt University Department of Biostatistics.**'" |
| Local copy | `data/titanic3.csv` (committed under the permission above) |
| SHA-256 | `db6df9666818c69a753cd85d743e01502c8518b00579b6aada0d4fc5a66ccb9d` (downloaded 19 Oct 2026; verified by `src/data.py` on re-download and by a unit test) |

## Columns

| Column | Meaning | Used? |
|---|---|---|
| `pclass` | Ticket class 1/2/3 (proxy for socio-economic status) | ✅ (inside `sex_class`) |
| `survived` | 1 = survived, 0 = died | **target** |
| `name` | Passenger name | ❌ identifier |
| `sex` | male / female | ✅ (inside `sex_class`) |
| `age` | Years (fractional for infants); **263 missing** | ✅ imputed + missing indicator; also `is_child` |
| `sibsp` | Siblings + spouses aboard | ✅ (`family_type`) |
| `parch` | Parents + children aboard | ✅ (`family_type`) |
| `ticket` | Ticket number | ❌ identifier |
| `fare` | Fare in 1912 pounds; 1 missing; 17 zero | ✅ as `log(1 + fare)` |
| `cabin` | Cabin number; 1,014 missing (77%) | ❌ too sparse |
| `embarked` | S = Southampton, C = Cherbourg, Q = Queenstown; 2 missing | ✅ |
| `boat` | Lifeboat number; 823 missing | ❌ **leakage** (only known after the outcome) |
| `body` | Body identification number; 1,188 missing | ❌ **leakage** (only exists if the passenger died) |
| `home.dest` | Home / destination; 564 missing | ❌ free text, sparse |

## Leakage evidence from the data
- Passengers with a `boat` value: 477 of 486 survived (98.1%).
- Passengers with a `body` value: 0 of 121 survived.
