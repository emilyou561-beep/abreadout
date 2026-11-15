"""Reading the Cookie Cats test with abreadout.

Same data as github.com/emilyou561-beep/cookie-cats-ab-test: 90,189 new
players, first gate at level 30 (control) or level 40 (treatment).
"""

import numpy as np
import pandas as pd

import abreadout as ab

URL = "https://raw.githubusercontent.com/emilyou561-beep/cookie-cats-ab-test/main/data/cookie_cats.csv"

df = pd.read_csv(URL)

print(ab.readout(df, group="version", metrics=["retention_1", "retention_7", "sum_gamerounds"],
                 control="gate_30", treatment="gate_40"))
print()

# was the test big enough to see a drop this size?
print(f"power for a 0.8 pt drop: {ab.power(baseline=0.19, mde=-0.008, n_control=44_700, n_treatment=45_489):.0%}")
print(f"smallest drop it could reliably detect: {ab.minimum_detectable_effect(0.19, 44_700, 45_489) * 100:.2f} pts")
print()

# rounds played are very skewed, so check the median with a bootstrap as well
c = df.loc[df.version == "gate_30", "sum_gamerounds"]
t = df.loc[df.version == "gate_40", "sum_gamerounds"]
print(ab.bootstrap_diff(c, t, statistic=np.median, n_resamples=2_000, metric="median rounds"))
