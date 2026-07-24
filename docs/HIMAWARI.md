# Tutorial Access Himawari Data

## Overview

**Himawari** data (Japanese geostationary meteorological satellites) is accessible on:

- **JMA** (Japan Meteorological Agency): https://www.data.jma.go.jp/mscweb/en/index.html
- **P-Tree System** (JAXA/EORC): https://www.eorc.jaxa.jp/ptree/index.html

JAXA's P-Tree System provides both the **Himawari Standard Data** (supplied by JMA) and the **derived geophysical data** produced by JAXA.

To have access to data, go to the registration page:
      https://www.eorc.jaxa.jp/ptree/registration_top.html

JAXA reviews the application, then activates your access rights to Himawari data. A final email confirms when you can download the data. This final email is where you'll find your login credentials:
- **UID** (user ID)
- **PW** (password)

---

# Directory Structure and File Naming Convention
## Level 2 (every 10 minutes)

### Short Wave Radiation (SWR)-Level 2

**Himawari**
```
/pub/himawari
  └── L2
       └── PAR
            └── [VER]
                 └── [YYYYMM]
                      └── [DD]
```

---

## Level 3 (hourly, daily, monthly)

### Short Wave Radiation (SWR)-Level 3

**Himawari**
```
/pub/himawari
  └── L3
       └── PAR
            └── [VER]
                 └── [YYYYMM]
                      └── [DD]
                      └── [daily]
                      └── [monthly]
```

**Path variable legend**

| Variable | Meaning |
|---|---|
| `VER` | Algorithm version |
| `YYYY` | 4-digit year of observation start time (timeline) |
| `MM` | 2-digit month of the timeline |
| `DD` | 2-digit day of the timeline |
| `hh` | 2-digit hour of the timeline |

---

# File Naming Convention

## Level 2

### Short Wave Radiation
```
Hnn_YYYYMMDD_hhmm_RFLVER_FLDK_xxxxx_yyyyy.nc   (5 km)
Hnn_YYYYMMDD_hhmm_rFLVER_FLDK_xxxxx_yyyyy.nc   (1 km)
```

| Variable | Meaning |
|---|---|
| `nn` | 2-digit Himawari satellite number (`08` = Himawari-8, `09` = Himawari-9) |
| `YYYY` | 4-digit year of observation start time |
| `MM` | 2-digit month |
| `DD` | 2-digit day |
| `hh` | 2-digit hour |
| `mm` | 2-digit minutes |
| `VER` | Algorithm version |
| `xxxxx` | Pixel number |
| `yyyyy` | Line number |

**Example:** `NC_H08_20150727_0800_RFL001_FLDK_02801_02401.nc`

---

## Level 3

### Short Wave Radiation

```
Hnn_YYYYMMDD_hhmm_LL_RFLVER_FLDK.xxxxx_yyyyy.nc   (5 km)
Hnn_YYYYMMDD_hhmm_LL_rFLVER_FLDK.xxxxx_yyyyy.nc   (1 km)
```

| Variable | Meaning |
|---|---|
| `nn` | 2-digit Himawari satellite number (`08` = Himawari-8, `09` = Himawari-9) |
| `YYYY` | 4-digit year of observation start time |
| `MM` | 2-digit month |
| `DD` | 2-digit day |
| `hh` | 2-digit hour |
| `mm` | 2-digit minutes |
| `LL` | Temporal resolution (`1H` = hourly, `1D` = daily, `1M` = monthly) |
| `VER` | Algorithm version |
| `xxxxx` | Pixel number |
| `yyyyy` | Line number |

**Example:** `H08_20150727_0800_1H_RFL001_FLDK.02801_02401.nc`

for more information : https://www.eorc.jaxa.jp/ptree/userguide.html
