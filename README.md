**Athlete Report Generator**
**This tool automatically generates individual athlete profiles based on field assessments of movement quality, strength, and balance. It is part of a research-grade pilot system built to support youth athlete screening and training decisions.

** **What the App Does****

Automatically classifies each athlete into one of four functional profiles:

Functionally Weak

Strength-Deficient

Stability-Deficient

No Clear Dysfunction

Generates a radar chart of motor deficits

Flags athletes by training priority

Outputs a custom PDF per athlete (optional)

Designed for real-time use by coaches, S&C, and physios




**Input Requirements**
The app currently supports Excel (.xlsx) files with specific column names. These must match exactly due to current pilot-phase constraints.
Required Columns
Please include one row per athlete with the following column headers (case-sensitive):

ID  
Sex_(M=1, F=2)  
Chronologic_Age  
Sport  
Sports (1 = individual, 2 = team sports)  
Geographic_Factor  
Weight (kg)  
Height (cm)  
BMI  
HAbd_L  
HAbd_P  
KE_L  
KE_P  
KF_L  
KF_P  
AP_L  
AP_P_PEAK_FORCE_(KG)_Normalized_to_body_weight  
YBT_ANT_L_Normalized  
YBT_ANT_R_Normalized  
YBT_PM_L_Normalized  
YBT_PM_R_Normalized  
YBT_PL_L_Normalized  
YBT_PL_R_Normalized  
YBT_COMPOSITE_R  
YBT_COMPOSITE_L  
FMS_TOTAL  

 **Coming Soon**
Simpler input template (fewer, cleaner column names)

Multi-language support

Batch reporting (PDF download for all)

**Notes:**
Strength values (HAbd, KE, KF, AP) must be normalized to body weight (%BW).

Y-Balance Test (YBT) values should be normalized to leg length (%LL).

FMS_TOTAL is the sum score (0–21) from the Functional Movement Screen.

**Troubleshooting**
If the app fails to process your file:

Make sure column names match exactly (including spacing and case).

Ensure no missing values in required columns.

Use .xlsx format only (not .csv or .xls).

**Citation**
This app is part of the open science project:
"An Interpretable Machine Learning Framework for Athlete Motor Profiling Using Multi-Domain Field Assessments: A Proof-of-Concept Study"
(Wilczyński et al., 2025
