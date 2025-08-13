import math



def ascvd_pce(age, sex, race, tc, hdl, sbp, treated, smoker, diabetic):
    sex_norm = sex.strip().title()
    race_input = race.strip().lower()
    if race_input in ("white", "other", "white/other"):
        race_norm = 'White/Other'
    elif race_input in ("black", "african american"):
        race_norm = 'African American'
    else:
        raise ValueError('Unsupported race : {race}')



    """
    Returns 10-year ASCVD risk (%) using the 2013 ACC/AHA Pooled-Cohort Equations.

    :param age: years 
    :param sex: 'Female' or 'Male'
    :param race: 'White/Other' or 'African American'
    :param tc: total cholesterol (mg/dL)
    :param hdl: HDL cholesterol (mg/dL)
    :param sbp: systolic blood pressure (mm Hg)
    :param treated: True if on antihypertensive therapy
    :param smoker: 1 if current smoker, else 0
    :param diabetic: 1 if diabetic, else 0
    :return: risk percentage (float)
    """

    print("=== ASCVD PCE Inputs ===")
    print(f"Age:                 {age} years")
    print(f"Sex:                 {sex_norm}")
    print(f"Race:                {race_norm}")
    print(f"Total cholesterol:   {tc} mg/dL")
    print(f"HDL cholesterol:     {hdl} mg/dL")
    print(f"Systolic BP:         {sbp} mm Hg")
    print(f"On BP therapy?       {'Yes' if treated else 'No'}")
    print(f"Current smoker?      {'Yes' if smoker else 'No'}")
    print(f"Diabetic?            {'Yes' if diabetic else 'No'}")
    print("========================\n")
    ln_age = math.log(age)
    ln_tc = math.log(tc)
    ln_hdl = math.log(hdl)
    ln_sbp = math.log(sbp)
    tr = 1 if treated else 0
    sm = smoker
    dm = diabetic

    if sex_norm == 'Female' and race_norm == 'White/Other':
        # Coefficients and baseline from Table A
        lp = (
                -29.799 * ln_age
                + 4.884 * ln_age ** 2
                + 13.540 * ln_tc
                - 3.114 * ln_age * ln_tc
                - 13.578 * ln_hdl
                + 3.149 * ln_age * ln_hdl
                + (2.019 * ln_sbp) * tr
                + (1.957 * ln_sbp) * (1 - tr)
                + 7.574 * sm
                - 1.665 * ln_age * sm
                + 0.661 * dm
        )
        s0, mean = 0.9665, -29.18

    elif sex_norm == 'Female' and race_norm == 'African American':
        lp = (
                17.114 * ln_age
                + 0.940 * ln_tc
                - 18.920 * ln_hdl
                + 4.475 * ln_age * ln_hdl
                + (29.291 * ln_sbp) * tr
                - 6.432 * ln_age * ln_sbp * tr
                + (27.820 * ln_sbp) * (1 - tr)
                - 6.087 * ln_age * ln_sbp * (1 - tr)
                + 0.691 * sm
                + 0.874 * dm
        )
        s0, mean = 0.9533, 86.61

    elif sex_norm == 'Male' and race_norm == 'White/Other':
        lp = (
                12.344 * ln_age
                + 11.853 * ln_tc
                - 2.664 * ln_age * ln_tc
                - 7.990 * ln_hdl
                + 1.769 * ln_age * ln_hdl
                + (1.797 * ln_sbp) * tr
                + (1.764 * ln_sbp) * (1 - tr)
                + 7.837 * sm
                - 1.795 * ln_age * sm
                + 0.658 * dm
        )
        s0, mean = 0.9144, 61.18

    elif sex_norm == 'Male' and race_norm == 'African American':
        lp = (
                2.469 * ln_age
                + 0.302 * ln_tc
                - 0.307 * ln_hdl
                + (1.916 * ln_sbp) * tr
                + (1.809 * ln_sbp) * (1 - tr)
                + 0.549 * sm
                + 0.645 * dm
        )
        s0, mean = 0.8954, 19.54

    else:
        raise ValueError(f"Unsupported grouping: {sex_norm}, {race_norm}")

    risk_percent = 1 - (s0 ** math.exp(lp - mean))
    print(f"Risk percentage: {(risk_percent * 100):.2f}%")
    return risk_percent * 100


