# Transformer Zero — demo answer key

Expert (gold) label vs. the fine-tuned model's label for each built-in example (`Load an example` dropdown), using the **Fast (batched)** engine. ✅ = match, ❌ = mismatch.
Pick an all-✅ example (e.g. #7) to show it working; use a ❌ row to discuss a real error.
**Overall on these 8 examples: 77/99 criteria correct (~78%).**


## Example 1: sigir-20141 / NCT00149227 — Add-on Effects of Valsartan on Morbi- Mortality (KYOTO HEART Study)

| ✓ | type | criterion | expert | model |
|---|------|-----------|--------|-------|
| ✅ | inclusion | Clinical diagnosis of hypertension | included | included |
| ❌ | inclusion | Clinical diagnosis of one or more risk factors, such as… | included | not enough information |
| ✅ | exclusion | Patients who have already been administered ARB | not excluded | not excluded |
| ❌ | exclusion | Patients with IHD within 6 months after percutaneous co… | not applicable | not excluded |
| ✅ | exclusion | Severe/malignant/secondary hypertensive patients | not excluded | not excluded |
| ❌ | exclusion | Pregnant women and women of childbearing potential | not enough information | not applicable |
| ✅ | exclusion | History of heart failure, unstable angina, myocardial i… | not excluded | not excluded |
| ✅ | exclusion | Arrhythmia needed to be treated or accompanied with sym… | not excluded | not excluded |
| ✅ | exclusion | Severe renal impairment (Serum creatinine >3.0 mg/dl) | not excluded | not excluded |
| ✅ | exclusion | Severe hepatic impairment (Hepatic failure, Cirrhosis, … | not excluded | not excluded |

*Example 1: 7/10 correct.*

## Example 2: sigir-20141 / NCT01397994 — Study to Assess Efficacy of Nicorandil+Atenolol vs Atenolol in Treatment of Chro…

| ✓ | type | criterion | expert | model |
|---|------|-----------|--------|-------|
| ✅ | inclusion | Patients of chronic stable angina with abnormal Exercis… | not enough information | not enough information |
| ✅ | inclusion | Male and female | included | included |
| ✅ | inclusion | Age 25 to 65 years | included | included |
| ✅ | inclusion | Patient must understand and be willing, able and likely… | included | included |
| ✅ | inclusion | Patient must be able to give voluntary written informed… | included | included |
| ✅ | exclusion | Hypertension of > 170/100 mm of Hg | not excluded | not excluded |
| ✅ | exclusion | Valvular heart disease and cardiomyopathy | not excluded | not excluded |
| ✅ | exclusion | Myocardial infarction in < 6 months | not excluded | not excluded |
| ❌ | exclusion | Unstable angina | not enough information | not excluded |
| ✅ | exclusion | Congestive cardiac failure | not excluded | not excluded |
| ✅ | exclusion | Severe anemia (Hb 7G/dl) | not excluded | not excluded |
| ✅ | exclusion | Cardiac arrhythmias or II or III degree AV block | not excluded | not excluded |
| ✅ | exclusion | Significant liver or renal dysfunction | not excluded | not excluded |
| ✅ | exclusion | IDDM (Type-1 diabetes mellitus) | not excluded | not excluded |
| ❌ | exclusion | Systolic blood pressure < 100 mm Hg | not excluded | not enough information |
| ❌ | exclusion | Pregnant and nursing women | not enough information | not applicable |
| ✅ | exclusion | Known hypersensitivity to nicorandil | not excluded | not excluded |
| ✅ | exclusion | On calcium channel blockers | not excluded | not excluded |
| ✅ | exclusion | Patients not eligible for Tc 99m SPECT | not enough information | not enough information |
| ❌ | exclusion | Patients in whom beta blockers are contraindicated | not enough information | not excluded |
| ✅ | exclusion | Geographical inaccessibility for treatment or follow-up… | not excluded | not excluded |

*Example 2: 17/21 correct.*

## Example 3: sigir-201415 / NCT00180739 — Safety Trial of Magnetic Resonance (MR) Guided Focused Ultrasound Surgery (FUS) …

| ✓ | type | criterion | expert | model |
|---|------|-----------|--------|-------|
| ✅ | inclusion | 1. Subject with uterine fibroids, who desire pregnancy … | not enough information | not enough information |
| ✅ | inclusion | 2. Use or non use of non-steroidal treatments for exces… | not enough information | not enough information |
| ✅ | inclusion | 3. Clinically normal PAP smear within timing of Nationa… | not enough information | not enough information |
| ✅ | inclusion | 4. Able and willing to give consent and able to attend … | included | included |
| ✅ | inclusion | 5. Able to communicate sensations during the MRgFUS pro… | not enough information | not enough information |
| ✅ | inclusion | 6. Having uterine fibroids that are device accessible (… | not enough information | not enough information |
| ✅ | inclusion | 7. Tumor(s) are clearly visible on non-contrast MRI. | not enough information | not enough information |
| ✅ | inclusion | 8. Largest fibroid 8 cm in diameter or 12 cm if receivi… | not enough information | not enough information |
| ✅ | exclusion | 1. Patient is pregnant as confirmed by pregnancy test a… | not excluded | not excluded |
| ✅ | exclusion | 2. Uterine size >20 weeks as evaluated by US or MR. | not excluded | not excluded |
| ❌ | exclusion | 3. Patients who are considered high risk pregnancy due … | not enough information | not excluded |
| ✅ | exclusion | 4. Patients with fibroid that is more than 50% sub-muco… | not enough information | not enough information |
| ❌ | exclusion | 5. Patients with adenomyosis | not enough information | not excluded |
| ✅ | exclusion | 6. Patient is on dialysis | not excluded | not excluded |
| ✅ | exclusion | 7. Hematocrit is < 25 | not excluded | not excluded |
| ✅ | exclusion | 8. Patient has hemolytic anemia | not excluded | not excluded |
| ✅ | exclusion | 9. Patient has unstable cardiac status including: § Uns… | not excluded | not excluded |
| ❌ | exclusion | 10. Patient has an ASA score of >2 | not enough information | not excluded |
| ✅ | exclusion | 11. Patient has severe cerebrovascular disease (multipl… | not excluded | not excluded |
| ✅ | exclusion | 12. Patient is on anti-coagulation therapy or has an un… | not excluded | not excluded |
| ❌ | exclusion | 13. Evidence of uterine pathology other than leiomyoma | excluded | not excluded |
| ✅ | exclusion | 14. Patient has an active pelvic infection or history o… | not excluded | not excluded |
| ✅ | exclusion | 15. Patient has an undiagnosed pelvic mass outside the … | not excluded | not excluded |
| ✅ | exclusion | 16. Patient weight >110 kg | not enough information | not enough information |
| ✅ | exclusion | 17. Subject with extensive abdominal scarring in an are… | not excluded | not excluded |
| ✅ | exclusion | 18. Subject with standard contraindications for MR imag… | not excluded | not excluded |
| ✅ | exclusion | 19. Known intolerance to the MRI contrast agent (e.g. G… | not excluded | not excluded |
| ✅ | exclusion | 20. Individuals who are not able or willing to tolerate… | not excluded | not excluded |
| ✅ | exclusion | 21. Patient with an intrauterine contraceptive device a… | not excluded | not excluded |
| ✅ | exclusion | 22. Women who are breast feeding. | not excluded | not excluded |
| ✅ | exclusion | 23. Five or more fibroids, bigger then 3cm diameter, ea… | not enough information | not enough information |

*Example 3: 27/31 correct.*

## Example 4: sigir-201415 / NCT00277680 — Laparoscopic Occlusion of Uterine Vessels Compared to Uterine Fibroid Embolizati…

| ✓ | type | criterion | expert | model |
|---|------|-----------|--------|-------|
| ✅ | inclusion | Menorrhagia and/or bulk symptoms associated with uterin… | not enough information | not enough information |
| ✅ | exclusion | Malignancy | not excluded | not excluded |
| ✅ | exclusion | Current or planned pregnancy | not excluded | not excluded |
| ✅ | exclusion | Small submucous fibroids suitable for hysteroscopic res… | not excluded | not excluded |
| ❌ | exclusion | Postmenopausal women | not excluded | not applicable |
| ✅ | exclusion | Suspected or known adenomyosis | not excluded | not excluded |
| ✅ | exclusion | Uterus size exceeding the umbilical level | not excluded | not excluded |
| ✅ | exclusion | Contraindications against laparoscopic surgery | not excluded | not excluded |

*Example 4: 7/8 correct.*

## Example 5: sigir-20142 / NCT00711399 — Assessment of Cough and Wheeze With Breath Sound Documenting Device

| ✓ | type | criterion | expert | model |
|---|------|-----------|--------|-------|
| ✅ | inclusion | Patient and/or parents/guardian signed informed consent | included | included |
| ✅ | inclusion | Patients with cough or shortness of breath | included | included |
| ✅ | exclusion | Chest tubes | not excluded | not excluded |
| ✅ | exclusion | Skin lesions precluding attachment of sensors | not excluded | not excluded |
| ❌ | exclusion | Respiratory distress | excluded | not excluded |
| ✅ | exclusion | Pregnant women | not applicable | not applicable |

*Example 5: 5/6 correct.*

## Example 6: sigir-20142 / NCT02618655 — Clinical Research for the Diagnosis of Tick-borne Diseases in Patients With Unex…

| ✓ | type | criterion | expert | model |
|---|------|-----------|--------|-------|
| ❌ | inclusion | patients have fever more than one week | not included | not enough information |
| ✅ | inclusion | temperature is higher than 38℃ Celsius degree | included | included |
| ❌ | inclusion | full of physical examination and laboratory examination… | not applicable | not enough information |
| ✅ | exclusion | fever for non-infectious diseases such as rheumatic aut… | not excluded | not excluded |
| ❌ | exclusion | we find that the patient selected does not meet the sel… | not enough information | not excluded |
| ❌ | exclusion | patients leave with automatic discharge | not enough information | not excluded |

*Example 6: 2/6 correct.*

## Example 7: sigir-201420 / NCT01594385 — Seprafilm in Open Abdomens: a Study of Wound and Adhesion Characteristics in Tra…

| ✓ | type | criterion | expert | model |
|---|------|-----------|--------|-------|
| ✅ | inclusion | Trauma patients undergoing DC/OA management for traumat… | not enough information | not enough information |
| ✅ | inclusion | Age 18+ | included | included |
| ✅ | inclusion | Life expectancy longer than 48 hours | included | included |
| ✅ | exclusion | Prisoners | not excluded | not excluded |
| ✅ | exclusion | Pregnant patients | not enough information | not enough information |
| ✅ | exclusion | Younger than 18 years of age | not excluded | not excluded |

*Example 7: 6/6 correct.*

## Example 8: sigir-201420 / NCT02255487 — Irrisept Versus Standard of Care in the Prevention of Surgical Site Infections

| ✓ | type | criterion | expert | model |
|---|------|-----------|--------|-------|
| ✅ | inclusion | Is male or female, 18 years of age or older | included | included |
| ✅ | inclusion | Has provided written informed consent or has surrogate … | included | included |
| ❌ | inclusion | Has experienced abdominal trauma, blunt or penetrating,… | included | not enough information |
| ❌ | inclusion | Has experienced acute surgical abdomen requiring open a… | included | not enough information |
| ✅ | exclusion | Known allergy to Chlorhexidine Gluconate (CHG) | not excluded | not excluded |
| ❌ | exclusion | Estimated Abbreviated Injury Scale (AIS) score of six (… | not excluded | not enough information |
| ❌ | exclusion | American Society of Anesthesiologists Physical Status C… | not excluded | not enough information |
| ❌ | exclusion | Female volunteers who are pregnant and/or breast feedin… | not excluded | not enough information |
| ✅ | exclusion | Damage control laparotomy | not excluded | not excluded |
| ✅ | exclusion | Abdominal incision created prior to operating room (i.e… | not excluded | not excluded |
| ✅ | exclusion | Currently enrolled in an ongoing, interventional, rando… | not excluded | not excluded |

*Example 8: 6/11 correct.*
