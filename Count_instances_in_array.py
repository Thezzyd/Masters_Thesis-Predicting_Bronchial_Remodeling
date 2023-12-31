import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict

'''kolagen_I_pow = [
["['AR08_K']", "['AR11_K']", "['AR12_K']", "['AR16_K']", "['AR19_K']", "['AR23_K']", "['AR28_K']", "['AR30_K']", "['AR40_K']", "['AR46_K']", "['AR47_K']", "['AR48_K']", "['AR51_K']"],
["['AR04_K']", "['AR08_K']", "['AR11_K']", "['AR12_K']", "['AR16_K']", "['AR19_K']", "['AR23_K']", "['AR28_K']", "['AR30_K']", "['AR40_K']", "['AR47_K']", "['AR48_K']", "['AR51_K']", "['AR52_K']", "['AR57_K']"],
["['AR04_K']", "['AR05_K']", "['AR09_K']", "['AR12_K']", "['AR23_K']", "['AR25_K']", "['AR28_K']", "['AR30_K']", "['AR34_K']", "['AR46_K']", "['AR47_K']", "['AR48_K']", "['AR49_K']", "['AR51_K']", "['AR53_K']"],
["['AR04_K']", "['AR08_K']", "['AR11_K']", "['AR12_K']", "['AR15_K']", "['AR16_K']", "['AR19_K']", "['AR23_K']", "['AR30_K']", "['AR42_K']", "['AR47_K']", "['AR48_K']", "['AR51_K']", "['AR53_K']", "['AR57_K']"],
["['AR05_K']", "['AR08_K']", "['AR09_K']", "['AR23_K']", "['AR25_K']", "['AR28_K']", "['AR30_K']", "['AR32_K']", "['AR39_K']", "['AR41_K']", "['AR42_K']", "['AR44_K']", "['AR46_K']", "['AR47_K']", "['AR48_K']", "['AR49_K']", "['AR51_K']", "['AR52_K']", "['AR53_K']", "['AR57_K']"],
["['AR12_K']", "['AR15_K']", "['AR16_K']", "['AR23_K']", "['AR30_K']", "['AR40_K']", "['AR46_K']", "['AR47_K']", "['AR51_K']", "['AR53_K']"],
["['AR04_K']", "['AR12_K']", "['AR15_K']", "['AR16_K']", "['AR23_K']", "['AR25_K']", "['AR30_K']", "['AR40_K']", "['AR43_K']", "['AR44_K']", "['AR46_K']", "['AR51_K']", "['AR53_K']"],
["['AR04_K']", "['AR08_K']", "['AR12_K']", "['AR23_K']", "['AR28_K']", "['AR30_K']", "['AR43_K']", "['AR47_K']", "['AR48_K']", "['AR51_K']", "['AR57_K']"],
["['AR04_K']", "['AR08_K']", "['AR12_K']", "['AR15_K']", "['AR16_K']", "['AR19_K']", "['AR23_K']", "['AR30_K']", "['AR44_K']", "['AR46_K']", "['AR47_K']", "['AR48_K']", "['AR51_K']", "['AR53_K']", "['AR57_K']"],
["['AR08_K']", "['AR11_K']", "['AR12_K']", "['AR16_K']", "['AR19_K']", "['AR23_K']", "['AR28_K']", "['AR30_K']", "['AR40_K']", "['AR46_K']", "['AR47_K']", "['AR48_K']", "['AR51_K']"]
]

rb1 = [
["['AR05_K']", "['AR08_K']", "['AR09_K']", "['AR11_K']", "['AR17_K']", "['AR25_K']", "['AR30_K']", "['AR34_K']", "['AR41_K']", "['AR42_K']", "['AR44_K']", "['AR51_K']"],
["['AR08_K']", "['AR09_K']", "['AR10_K']", "['AR19_K']", "['AR25_K']", "['AR30_K']", "['AR41_K']", "['AR42_K']", "['AR44_K']", "['AR51_K']"],
["['AR08_K']", "['AR09_K']", "['AR11_K']", "['AR19_K']", "['AR20_K']", "['AR22_K']", "['AR41_K']", "['AR42_K']", "['AR44_K']", "['AR48_K']", "['AR52_K']", "['AR53_K']"],
["['AR05_K']", "['AR08_K']", "['AR09_K']", "['AR11_K']", "['AR17_K']", "['AR25_K']", "['AR30_K']", "['AR34_K']", "['AR41_K']", "['AR44_K']", "['AR51_K']"],
["['AR08_K']", "['AR09_K']", "['AR25_K']", "['AR30_K']", "['AR34_K']", "['AR41_K']", "['AR42_K']", "['AR44_K']", "['AR51_K']"],
["['AR08_K']", "['AR09_K']", "['AR25_K']", "['AR30_K']", "['AR41_K']", "['AR42_K']", "['AR44_K']", "['AR51_K']"],
["['AR05_K']", "['AR08_K']", "['AR09_K']", "['AR10_K']", "['AR11_K']", "['AR17_K']", "['AR19_K']", "['AR21_K']", "['AR24_K']", "['AR25_K']", "['AR34_K']", "['AR41_K']", "['AR42_K']", "['AR44_K']", "['AR51_K']", "['AR53_K']"],
["['AR08_K']", "['AR09_K']", "['AR11_K']", "['AR20_K']", "['AR22_K']", "['AR24_K']", "['AR41_K']", "['AR42_K']", "['AR44_K']", "['AR48_K']", "['AR52_K']", "['AR53_K']"],
["['AR08_K']", "['AR09_K']", "['AR25_K']", "['AR30_K']", "['AR34_K']", "['AR41_K']", "['AR42_K']", "['AR44_K']", "['AR51_K']", "['AR52_K']"],
["['AR08_K']", "['AR09_K']", "['AR25_K']", "['AR30_K']", "['AR34_K']", "['AR41_K']", "['AR42_K']", "['AR44_K']", "['AR51_K']"]
]

rb10 = [
["['AR04_K']", "['AR07_K']", "['AR08_K']", "['AR09_K']", "['AR10_K']", "['AR11_K']", "['AR12_K']", "['AR13_K']", "['AR20_K']", "['AR23_K']", "['AR28_K']", "['AR34_K']", "['AR41_K']", "['AR42_K']", "['AR45_K']", "['AR47_K']", "['AR52_K']", "['AR53_K']", "['AR56_K']", "['AR57_K']"],
["['AR11_K']", "['AR13_K']", "['AR17_K']", "['AR23_K']", "['AR24_K']", "['AR25_K']", "['AR27_K']", "['AR39_K']", "['AR42_K']", "['AR44_K']", "['AR47_K']", "['AR48_K']", "['AR50_K']", "['AR51_K']", "['AR52_K']", "['AR53_K']", "['AR56_K']", "['AR57_K']"],
["['AR08_K']", "['AR10_K']", "['AR12_K']", "['AR13_K']", "['AR17_K']", "['AR20_K']", "['AR23_K']", "['AR24_K']", "['AR32_K']", "['AR34_K']", "['AR42_K']", "['AR45_K']", "['AR47_K']", "['AR48_K']", "['AR51_K']", "['AR52_K']", "['AR53_K']", "['AR57_K']"],
["['AR04_K']", "['AR08_K']", "['AR10_K']", "['AR12_K']", "['AR13_K']", "['AR17_K']", "['AR20_K']", "['AR23_K']", "['AR24_K']", "['AR34_K']", "['AR39_K']", "['AR41_K']", "['AR44_K']", "['AR45_K']", "['AR47_K']", "['AR52_K']", "['AR57_K']"],
["['AR09_K']", "['AR11_K']", "['AR17_K']", "['AR22_K']", "['AR24_K']", "['AR25_K']", "['AR28_K']", "['AR34_K']", "['AR41_K']", "['AR45_K']", "['AR48_K']", "['AR50_K']", "['AR52_K']", "['AR53_K']", "['AR57_K']"],
["['AR08_K']", "['AR11_K']", "['AR13_K']", "['AR17_K']", "['AR22_K']", "['AR23_K']", "['AR24_K']", "['AR27_K']", "['AR32_K']", "['AR34_K']", "['AR41_K']", "['AR44_K']", "['AR48_K']", "['AR50_K']", "['AR51_K']", "['AR52_K']", "['AR53_K']"],
["['AR04_K']", "['AR08_K']", "['AR10_K']", "['AR12_K']", "['AR13_K']", "['AR20_K']", "['AR23_K']", "['AR24_K']", "['AR34_K']", "['AR41_K']", "['AR42_K']", "['AR47_K']", "['AR52_K']", "['AR53_K']", "['AR56_K']", "['AR57_K']"],
["['AR10_K']", "['AR17_K']", "['AR20_K']", "['AR22_K']", "['AR27_K']", "['AR28_K']", "['AR30_K']", "['AR34_K']", "['AR47_K']", "['AR48_K']", "['AR51_K']"],
["['AR07_K']", "['AR11_K']", "['AR13_K']", "['AR17_K']", "['AR23_K']", "['AR24_K']", "['AR27_K']", "['AR28_K']", "['AR34_K']", "['AR42_K']", "['AR45_K']", "['AR47_K']", "['AR50_K']", "['AR51_K']", "['AR52_K']", "['AR53_K']", "['AR57_K']"],
["['AR04_K']", "['AR07_K']", "['AR08_K']", "['AR09_K']", "['AR10_K']", "['AR11_K']", "['AR12_K']", "['AR13_K']", "['AR20_K']", "['AR23_K']", "['AR24_K']", "['AR34_K']", "['AR39_K']", "['AR41_K']", "['AR45_K']", "['AR47_K']", "['AR51_K']", "['AR52_K']", "['AR53_K']", "['AR57_K']"]
]

wall_tch = [
["['AR04_K']", "['AR05_K']", "['AR08_K']", "['AR09_K']", "['AR11_K']", "['AR28_K']", "['AR41_K']", "['AR43_K']", "['AR50_K']", "['AR51_K']", "['AR52_K']", "['AR53_K']"],
["['AR04_K']", "['AR05_K']", "['AR08_K']", "['AR09_K']", "['AR11_K']", "['AR28_K']", "['AR41_K']", "['AR43_K']", "['AR48_K']", "['AR50_K']", "['AR51_K']", "['AR52_K']", "['AR53_K']", "['AR57_K']"],
["['AR05_K']", "['AR11_K']", "['AR13_K']", "['AR18_K']", "['AR24_K']", "['AR27_K']", "['AR30_K']", "['AR41_K']", "['AR42_K']", "['AR48_K']", "['AR50_K']"],
["['AR04_K']", "['AR05_K']", "['AR08_K']", "['AR09_K']", "['AR19_K']", "['AR24_K']", "['AR28_K']", "['AR41_K']", "['AR43_K']", "['AR48_K']", "['AR50_K']", "['AR52_K']", "['AR53_K']", "['AR57_K']"],
["['AR05_K']", "['AR11_K']", "['AR13_K']", "['AR18_K']", "['AR27_K']", "['AR41_K']", "['AR42_K']", "['AR48_K']", "['AR50_K']", "['AR53_K']"],
["['AR05_K']", "['AR09_K']", "['AR11_K']", "['AR18_K']", "['AR27_K']", "['AR41_K']", "['AR42_K']", "['AR48_K']", "['AR53_K']", "['AR56_K']"],
["['AR05_K']", "['AR11_K']", "['AR12_K']", "['AR18_K']", "['AR20_K']", "['AR42_K']", "['AR48_K']", "['AR50_K']", "['AR52_K']", "['AR53_K']", "['AR56_K']"],
["['AR05_K']", "['AR07_K']", "['AR08_K']", "['AR13_K']", "['AR18_K']", "['AR21_K']", "['AR25_K']", "['AR30_K']", "['AR39_K']", "['AR41_K']", "['AR48_K']", "['AR50_K']"],
["['AR05_K']", "['AR09_K']", "['AR17_K']", "['AR18_K']", "['AR27_K']", "['AR42_K']", "['AR48_K']", "['AR50_K']", "['AR53_K']"],
["['AR05_K']", "['AR08_K']", "['AR09_K']", "['AR11_K']", "['AR13_K']", "['AR18_K']", "['AR20_K']", "['AR42_K']", "['AR48_K']", "['AR50_K']", "['AR51_K']", "['AR53_K']", "['AR56_K']"]
]

sr_har = [
["['AR07_K']", "['AR11_K']", "['AR16_K']", "['AR20_K']", "['AR21_K']", "['AR22_K']", "['AR26_K']", "['AR27_K']", "['AR32_K']", "['AR39_K']", "['AR48_K']", "['AR52_K']", "['AR54_K']", "['AR56_K']", "['AR57_K']"],
["['AR07_K']", "['AR08_K']", "['AR11_K']", "['AR16_K']", "['AR18_K']", "['AR20_K']", "['AR21_K']", "['AR22_K']", "['AR26_K']", "['AR27_K']", "['AR32_K']", "['AR39_K']", "['AR52_K']", "['AR54_K']", "['AR56_K']", "['AR57_K']"],
["['AR07_K']", "['AR11_K']", "['AR15_K']", "['AR16_K']", "['AR20_K']", "['AR21_K']", "['AR22_K']", "['AR27_K']", "['AR32_K']", "['AR45_K']", "['AR52_K']", "['AR54_K']", "['AR57_K']"],
["['AR04_K']", "['AR05_K']", "['AR07_K']", "['AR11_K']", "['AR16_K']", "['AR23_K']", "['AR30_K']", "['AR39_K']", "['AR45_K']", "['AR50_K']", "['AR54_K']", "['AR56_K']"],
["['AR11_K']", "['AR16_K']", "['AR20_K']", "['AR21_K']", "['AR22_K']", "['AR23_K']", "['AR27_K']", "['AR32_K']", "['AR39_K']", "['AR48_K']", "['AR52_K']", "['AR57_K']"],
["['AR08_K']", "['AR11_K']", "['AR16_K']", "['AR20_K']", "['AR21_K']", "['AR22_K']", "['AR26_K']", "['AR27_K']", "['AR31_K']", "['AR32_K']", "['AR48_K']", "['AR52_K']", "['AR54_K']", "['AR56_K']", "['AR57_K']"],
["['AR11_K']", "['AR16_K']", "['AR20_K']", "['AR21_K']", "['AR22_K']", "['AR26_K']", "['AR27_K']", "['AR31_K']", "['AR32_K']", "['AR39_K']", "['AR45_K']", "['AR56_K']", "['AR57_K']"],
["['AR05_K']", "['AR07_K']", "['AR08_K']", "['AR11_K']", "['AR16_K']", "['AR20_K']", "['AR21_K']", "['AR23_K']", "['AR30_K']", "['AR31_K']", "['AR39_K']", "['AR50_K']", "['AR52_K']", "['AR53_K']", "['AR54_K']", "['AR57_K']"],
["['AR11_K']", "['AR16_K']", "['AR20_K']", "['AR21_K']", "['AR22_K']", "['AR27_K']", "['AR32_K']", "['AR39_K']", "['AR48_K']", "['AR52_K']", "['AR56_K']", "['AR57_K']"],
["['AR07_K']", "['AR11_K']", "['AR16_K']", "['AR18_K']", "['AR20_K']", "['AR21_K']", "['AR22_K']", "['AR26_K']", "['AR27_K']", "['AR32_K']", "['AR39_K']", "['AR52_K']", "['AR54_K']", "['AR57_K']"]
]

kolagen_sila = [
["['AR05_K']", "['AR07_K']", "['AR08_K']", "['AR09_K']", "['AR13_K']", "['AR18_K']", "['AR19_K']", "['AR21_K']", "['AR22_K']", "['AR23_K']", "['AR25_K']", "['AR28_K']", "['AR32_K']", "['AR45_K']", "['AR47_K']", "['AR48_K']", "['AR49_K']", "['AR50_K']", "['AR52_K']", "['AR54_K']", "['AR56_K']", "['AR57_K']"],
["['AR07_K']", "['AR09_K']", "['AR15_K']", "['AR17_K']", "['AR19_K']", "['AR21_K']", "['AR22_K']", "['AR23_K']", "['AR24_K']", "['AR25_K']", "['AR26_K']", "['AR27_K']", "['AR28_K']", "['AR30_K']", "['AR31_K']", "['AR32_K']", "['AR45_K']", "['AR47_K']", "['AR49_K']", "['AR50_K']", "['AR51_K']", "['AR52_K']", "['AR53_K']", "['AR54_K']", "['AR56_K']"],
["['AR15_K']", "['AR19_K']", "['AR20_K']", "['AR21_K']", "['AR22_K']", "['AR23_K']", "['AR24_K']", "['AR28_K']", "['AR32_K']", "['AR34_K']", "['AR41_K']", "['AR42_K']", "['AR44_K']", "['AR45_K']", "['AR46_K']", "['AR47_K']", "['AR48_K']", "['AR49_K']", "['AR50_K']", "['AR52_K']", "['AR53_K']", "['AR56_K']"],
["['AR05_K']", "['AR07_K']", "['AR08_K']", "['AR09_K']", "['AR10_K']", "['AR13_K']", "['AR15_K']", "['AR17_K']", "['AR19_K']", "['AR21_K']", "['AR22_K']", "['AR23_K']", "['AR24_K']", "['AR25_K']", "['AR26_K']", "['AR27_K']", "['AR28_K']", "['AR31_K']", "['AR34_K']", "['AR39_K']", "['AR44_K']", "['AR45_K']", "['AR47_K']", "['AR49_K']", "['AR50_K']", "['AR53_K']", "['AR54_K']", "['AR56_K']"],
["['AR05_K']", "['AR13_K']", "['AR19_K']", "['AR21_K']", "['AR22_K']", "['AR27_K']", "['AR28_K']", "['AR29_K']", "['AR30_K']", "['AR32_K']", "['AR41_K']", "['AR42_K']", "['AR44_K']", "['AR45_K']", "['AR46_K']", "['AR47_K']", "['AR48_K']", "['AR49_K']", "['AR50_K']"],
["['AR05_K']", "['AR07_K']", "['AR10_K']", "['AR17_K']", "['AR21_K']", "['AR22_K']", "['AR23_K']", "['AR25_K']", "['AR26_K']", "['AR27_K']", "['AR28_K']", "['AR31_K']", "['AR32_K']", "['AR34_K']", "['AR39_K']", "['AR40_K']", "['AR45_K']", "['AR47_K']", "['AR49_K']", "['AR50_K']", "['AR51_K']", "['AR52_K']", "['AR53_K']", "['AR54_K']", "['AR56_K']", "['AR57_K']"],
["['AR05_K']", "['AR10_K']", "['AR11_K']", "['AR13_K']", "['AR15_K']", "['AR17_K']", "['AR19_K']", "['AR20_K']", "['AR21_K']", "['AR22_K']", "['AR23_K']", "['AR24_K']", "['AR27_K']", "['AR28_K']", "['AR32_K']", "['AR40_K']", "['AR41_K']", "['AR42_K']", "['AR44_K']", "['AR46_K']", "['AR47_K']", "['AR49_K']", "['AR53_K']", "['AR54_K']", "['AR56_K']"],
["['AR09_K']", "['AR11_K']", "['AR12_K']", "['AR13_K']", "['AR16_K']", "['AR17_K']", "['AR20_K']", "['AR21_K']", "['AR22_K']", "['AR23_K']", "['AR24_K']", "['AR25_K']", "['AR26_K']", "['AR27_K']", "['AR28_K']", "['AR32_K']", "['AR34_K']", "['AR40_K']", "['AR42_K']", "['AR43_K']", "['AR44_K']", "['AR45_K']", "['AR46_K']", "['AR47_K']", "['AR49_K']", "['AR50_K']", "['AR51_K']", "['AR52_K']", "['AR53_K']", "['AR54_K']", "['AR56_K']", "['AR57_K']"],
["['AR08_K']", "['AR10_K']", "['AR12_K']", "['AR13_K']", "['AR19_K']", "['AR21_K']", "['AR23_K']", "['AR25_K']", "['AR27_K']", "['AR28_K']", "['AR29_K']", "['AR30_K']", "['AR31_K']", "['AR32_K']", "['AR41_K']", "['AR42_K']", "['AR44_K']", "['AR45_K']", "['AR47_K']", "['AR48_K']", "['AR49_K']", "['AR50_K']", "['AR52_K']"],
["['AR05_K']", "['AR11_K']", "['AR12_K']", "['AR19_K']", "['AR20_K']", "['AR21_K']", "['AR22_K']", "['AR24_K']", "['AR27_K']", "['AR28_K']", "['AR31_K']", "['AR32_K']", "['AR41_K']", "['AR42_K']", "['AR43_K']", "['AR44_K']", "['AR45_K']", "['AR46_K']", "['AR47_K']", "['AR49_K']", "['AR50_K']", "['AR52_K']", "['AR53_K']", "['AR54_K']"]
]'''


s_kol_pow = [
["['AR10_S']", "['AR12_S']", "['AR13_S']", "['AR23_S']", "['AR24_S']", "['AR25_S']", "['AR26_S']", "['AR33_S']", "['AR36_S']", "['AR39_S']", "['AR43_S']", "['AR47_S']", "['AR50_S']", "['AR54_S']", "['AR57_S']"],
["['AR04_S']", "['AR09_S']", "['AR12_S']", "['AR16_S']", "['AR23_S']", "['AR24_S']", "['AR25_S']", "['AR26_S']", "['AR27_S']", "['AR36_S']", "['AR43_S']", "['AR47_S']", "['AR49_S']", "['AR52_S']", "['AR55_S']", "['AR57_S']"],
["['AR12_S']", "['AR23_S']", "['AR24_S']", "['AR26_S']", "['AR28_S']", "['AR33_S']", "['AR35_S']", "['AR36_S']", "['AR39_S']", "['AR43_S']", "['AR47_S']", "['AR48_S']", "['AR49_S']", "['AR50_S']"],
["['AR10_S']", "['AR13_S']", "['AR25_S']", "['AR26_S']", "['AR33_S']", "['AR36_S']", "['AR39_S']", "['AR43_S']", "['AR47_S']", "['AR48_S']", "['AR50_S']", "['AR57_S']"],
["['AR12_S']", "['AR23_S']", "['AR24_S']", "['AR26_S']", "['AR28_S']", "['AR33_S']", "['AR35_S']", "['AR36_S']", "['AR39_S']", "['AR43_S']", "['AR47_S']", "['AR48_S']", "['AR49_S']", "['AR50_S']"],
["['AR10_S']", "['AR11_S']", "['AR12_S']", "['AR23_S']", "['AR26_S']", "['AR27_S']", "['AR33_S']", "['AR36_S']", "['AR43_S']", "['AR45_S']", "['AR47_S']", "['AR52_S']", "['AR57_S']"],
["['AR04_S']", "['AR09_S']", "['AR12_S']", "['AR15_S']", "['AR23_S']", "['AR26_S']", "['AR27_S']", "['AR33_S']", "['AR36_S']", "['AR39_S']", "['AR43_S']", "['AR47_S']", "['AR49_S']", "['AR50_S']", "['AR52_S']", "['AR55_S']"],
["['AR15_S']", "['AR23_S']", "['AR24_S']", "['AR25_S']", "['AR26_S']", "['AR36_S']", "['AR43_S']", "['AR47_S']", "['AR52_S']", "['AR55_S']", "['AR57_S']"],
["['AR08_S']", "['AR11_S']", "['AR13_S']", "['AR23_S']", "['AR25_S']", "['AR26_S']", "['AR30_S']", "['AR33_S']", "['AR36_S']", "['AR39_S']", "['AR43_S']", "['AR47_S']", "['AR49_S']", "['AR50_S']", "['AR57_S']"],
["['AR23_S']", "['AR25_S']", "['AR36_S']", "['AR43_S']", "['AR49_S']", "['AR52_S']", "['AR55_S']", "['AR57_S']"]
]

s_kol_sila = [
["['AR12_S']", "['AR23_S']", "['AR24_S']", "['AR26_S']", "['AR28_S']", "['AR33_S']", "['AR35_S']", "['AR36_S']", "['AR39_S']", "['AR43_S']", "['AR47_S']", "['AR48_S']", "['AR49_S']", "['AR50_S']"],
["['AR13_S']", "['AR23_S']", "['AR25_S']", "['AR26_S']", "['AR33_S']", "['AR36_S']", "['AR39_S']", "['AR47_S']", "['AR50_S']", "['AR52_S']", "['AR57_S']"],
["['AR09_S']", "['AR11_S']", "['AR12_S']", "['AR18_S']", "['AR19_S']", "['AR24_S']", "['AR28_S']", "['AR29_S']", "['AR33_S']", "['AR41_S']", "['AR42_S']", "['AR45_S']", "['AR47_S']", "['AR50_S']", "['AR55_S']", "['AR57_S']"],
["['AR04_S']", "['AR09_S']", "['AR11_S']", "['AR12_S']", "['AR13_S']", "['AR19_S']", "['AR27_S']", "['AR28_S']", "['AR41_S']", "['AR42_S']", "['AR43_S']", "['AR45_S']", "['AR47_S']", "['AR49_S']", "['AR50_S']", "['AR55_S']", "['AR56_S']"],
["['AR04_S']", "['AR12_S']", "['AR13_S']", "['AR16_S']", "['AR18_S']", "['AR19_S']", "['AR28_S']", "['AR33_S']", "['AR35_S']", "['AR41_S']", "['AR42_S']", "['AR43_S']", "['AR45_S']", "['AR47_S']", "['AR49_S']", "['AR50_S']", "['AR55_S']", "['AR56_S']", "['AR57_S']"],
["['AR09_S']", "['AR11_S']", "['AR12_S']", "['AR13_S']", "['AR19_S']", "['AR27_S']", "['AR28_S']", "['AR33_S']", "['AR35_S']", "['AR41_S']", "['AR42_S']", "['AR45_S']", "['AR47_S']", "['AR49_S']", "['AR50_S']", "['AR52_S']", "['AR56_S']", "['AR57_S']"],
["['AR04_S']", "['AR09_S']", "['AR11_S']", "['AR12_S']", "['AR19_S']", "['AR24_S']", "['AR27_S']", "['AR28_S']", "['AR29_S']", "['AR30_S']", "['AR35_S']", "['AR39_S']", "['AR41_S']", "['AR42_S']", "['AR43_S']", "['AR45_S']", "['AR47_S']", "['AR49_S']", "['AR50_S']", "['AR56_S']"],
["['AR04_S']", "['AR07_S']", "['AR11_S']", "['AR12_S']", "['AR15_S']", "['AR17_S']", "['AR18_S']", "['AR19_S']", "['AR31_S']", "['AR33_S']", "['AR35_S']", "['AR41_S']", "['AR42_S']", "['AR45_S']", "['AR49_S']", "['AR50_S']", "['AR53_S']", "['AR54_S']", "['AR55_S']", "['AR57_S']"],
["['AR09_S']", "['AR11_S']", "['AR12_S']", "['AR27_S']", "['AR39_S']", "['AR41_S']", "['AR42_S']", "['AR45_S']", "['AR47_S']", "['AR49_S']", "['AR50_S']", "['AR55_S']", "['AR56_S']"],
["['AR04_S']", "['AR07_S']", "['AR11_S']", "['AR12_S']", "['AR13_S']", "['AR15_S']", "['AR18_S']", "['AR22_S']", "['AR23_S']", "['AR27_S']", "['AR28_S']", "['AR33_S']", "['AR35_S']", "['AR41_S']", "['AR42_S']", "['AR43_S']", "['AR45_S']", "['AR47_S']", "['AR49_S']", "['AR50_S']", "['AR54_S']", "['AR55_S']", "['AR57_S']"]
]

s_RB1 = [
["['AR04_S']", "['AR07_S']", "['AR08_S']", "['AR09_S']", "['AR10_S']", "['AR11_S']", "['AR12_S']", "['AR13_S']", "['AR17_S']", "['AR19_S']", "['AR22_S']", "['AR23_S']", "['AR24_S']", "['AR25_S']", "['AR30_S']", "['AR33_S']", "['AR35_S']", "['AR41_S']", "['AR42_S']", "['AR44_S']", "['AR47_S']", "['AR50_S']", "['AR51_S']", "['AR53_S']", "['AR55_S']", "['AR56_S']", "['AR57_S']"],
["['AR08_S']", "['AR09_S']", "['AR12_S']", "['AR23_S']", "['AR30_S']", "['AR41_S']", "['AR42_S']", "['AR44_S']", "['AR48_S']", "['AR51_S']", "['AR52_S']"],
["['AR07_S']", "['AR08_S']", "['AR10_S']", "['AR11_S']", "['AR18_S']", "['AR19_S']", "['AR21_S']", "['AR22_S']", "['AR23_S']", "['AR24_S']", "['AR25_S']", "['AR30_S']", "['AR35_S']", "['AR42_S']", "['AR44_S']", "['AR47_S']", "['AR50_S']", "['AR51_S']", "['AR52_S']", "['AR53_S']", "['AR55_S']", "['AR56_S']", "['AR57_S']"],
["['AR07_S']", "['AR11_S']", "['AR12_S']", "['AR13_S']", "['AR17_S']", "['AR18_S']", "['AR19_S']", "['AR21_S']", "['AR22_S']", "['AR23_S']", "['AR24_S']", "['AR25_S']", "['AR27_S']", "['AR28_S']", "['AR33_S']", "['AR41_S']", "['AR42_S']", "['AR43_S']", "['AR44_S']", "['AR45_S']", "['AR47_S']", "['AR48_S']", "['AR50_S']", "['AR51_S']", "['AR52_S']", "['AR55_S']", "['AR57_S']"],
["['AR07_S']", "['AR08_S']", "['AR09_S']", "['AR11_S']", "['AR13_S']", "['AR17_S']", "['AR21_S']", "['AR22_S']", "['AR23_S']", "['AR24_S']", "['AR25_S']", "['AR28_S']", "['AR42_S']", "['AR47_S']", "['AR50_S']", "['AR51_S']", "['AR53_S']", "['AR55_S']", "['AR56_S']", "['AR57_S']"],
["['AR08_S']", "['AR12_S']", "['AR13_S']", "['AR17_S']", "['AR19_S']", "['AR22_S']", "['AR23_S']", "['AR24_S']", "['AR27_S']", "['AR35_S']", "['AR39_S']", "['AR42_S']", "['AR43_S']", "['AR44_S']", "['AR47_S']", "['AR48_S']", "['AR51_S']", "['AR53_S']", "['AR55_S']", "['AR57_S']"],
["['AR04_S']", "['AR11_S']", "['AR12_S']", "['AR13_S']", "['AR17_S']", "['AR18_S']", "['AR19_S']", "['AR21_S']", "['AR23_S']", "['AR24_S']", "['AR25_S']", "['AR27_S']", "['AR28_S']", "['AR30_S']", "['AR33_S']", "['AR35_S']", "['AR39_S']", "['AR41_S']", "['AR42_S']", "['AR43_S']", "['AR44_S']", "['AR45_S']", "['AR47_S']", "['AR48_S']", "['AR50_S']", "['AR51_S']", "['AR52_S']", "['AR53_S']", "['AR55_S']", "['AR57_S']"],
["['AR07_S']", "['AR10_S']", "['AR17_S']", "['AR19_S']", "['AR23_S']", "['AR24_S']", "['AR25_S']", "['AR30_S']", "['AR33_S']", "['AR35_S']", "['AR39_S']", "['AR41_S']", "['AR42_S']", "['AR45_S']", "['AR48_S']", "['AR50_S']", "['AR51_S']", "['AR52_S']", "['AR53_S']", "['AR55_S']", "['AR56_S']"],
["['AR08_S']", "['AR09_S']", "['AR11_S']", "['AR13_S']", "['AR17_S']", "['AR19_S']", "['AR22_S']", "['AR23_S']", "['AR24_S']", "['AR25_S']", "['AR28_S']", "['AR30_S']", "['AR35_S']", "['AR41_S']", "['AR44_S']", "['AR47_S']", "['AR50_S']", "['AR51_S']", "['AR53_S']", "['AR55_S']", "['AR56_S']", "['AR57_S']"],
["['AR07_S']", "['AR10_S']", "['AR11_S']", "['AR17_S']", "['AR19_S']", "['AR21_S']", "['AR22_S']", "['AR23_S']", "['AR24_S']", "['AR25_S']", "['AR30_S']", "['AR33_S']", "['AR35_S']", "['AR39_S']", "['AR41_S']", "['AR42_S']", "['AR43_S']", "['AR45_S']", "['AR47_S']", "['AR48_S']", "['AR51_S']", "['AR52_S']", "['AR53_S']", "['AR55_S']", "['AR56_S']", "['AR57_S']"]
]

s_RB10 = [
["['AR08_S']", "['AR09_S']", "['AR30_S']", "['AR41_S']", "['AR42_S']", "['AR44_S']", "['AR48_S']", "['AR49_S']", "['AR52_S']"],
["['AR08_S']", "['AR09_S']", "['AR12_S']", "['AR30_S']", "['AR41_S']", "['AR42_S']", "['AR44_S']", "['AR48_S']", "['AR49_S']", "['AR52_S']"],
["['AR04_S']", "['AR08_S']", "['AR09_S']", "['AR30_S']", "['AR41_S']", "['AR44_S']"],
["['AR08_S']", "['AR09_S']", "['AR12_S']", "['AR33_S']", "['AR41_S']", "['AR42_S']", "['AR44_S']", "['AR48_S']", "['AR49_S']", "['AR51_S']", "['AR56_S']"],
["['AR08_S']", "['AR09_S']", "['AR30_S']", "['AR41_S']", "['AR42_S']", "['AR44_S']", "['AR48_S']", "['AR49_S']", "['AR52_S']"],
["['AR08_S']", "['AR09_S']", "['AR30_S']", "['AR41_S']", "['AR42_S']", "['AR44_S']", "['AR48_S']", "['AR49_S']"],
["['AR08_S']", "['AR09_S']", "['AR30_S']", "['AR41_S']", "['AR44_S']", "['AR49_S']"],
["['AR04_S']", "['AR08_S']", "['AR09_S']", "['AR30_S']", "['AR41_S']", "['AR42_S']", "['AR44_S']", "['AR49_S']"],
["['AR08_S']", "['AR09_S']", "['AR30_S']", "['AR41_S']", "['AR42_S']", "['AR44_S']", "['AR48_S']", "['AR49_S']", "['AR52_S']"],
["['AR04_S']", "['AR08_S']", "['AR09_S']", "['AR30_S']", "['AR41_S']", "['AR42_S']", "['AR44_S']", "['AR49_S']", "['AR50_S']"]
]

s_sr_arm = [
["['AR16_S']", "['AR21_S']", "['AR26_S']", "['AR27_S']", "['AR30_S']", "['AR48_S']", "['AR50_S']", "['AR52_S']", "['AR53_S']"],
["['AR11_S']", "['AR16_S']", "['AR21_S']", "['AR26_S']", "['AR30_S']", "['AR35_S']", "['AR48_S']", "['AR50_S']", "['AR52_S']", "['AR53_S']"],
["['AR15_S']", "['AR21_S']", "['AR23_S']", "['AR30_S']", "['AR31_S']", "['AR36_S']", "['AR48_S']", "['AR49_S']", "['AR50_S']", "['AR53_S']", "['AR55_S']", "['AR56_S']", "['AR57_S']"],
["['AR11_S']", "['AR16_S']", "['AR21_S']", "['AR26_S']", "['AR27_S']", "['AR30_S']", "['AR48_S']", "['AR50_S']", "['AR52_S']", "['AR53_S']"],
["['AR04_S']", "['AR11_S']", "['AR16_S']", "['AR18_S']", "['AR21_S']", "['AR23_S']", "['AR26_S']", "['AR30_S']", "['AR35_S']", "['AR48_S']", "['AR50_S']", "['AR52_S']", "['AR53_S']"],
["['AR16_S']", "['AR18_S']", "['AR21_S']", "['AR26_S']", "['AR27_S']", "['AR30_S']", "['AR31_S']", "['AR48_S']", "['AR50_S']", "['AR52_S']", "['AR53_S']"],
["['AR04_S']", "['AR07_S']", "['AR16_S']", "['AR21_S']", "['AR26_S']", "['AR27_S']", "['AR30_S']", "['AR39_S']", "['AR48_S']", "['AR50_S']", "['AR52_S']", "['AR53_S']", "['AR54_S']", "['AR55_S']"],
["['AR21_S']", "['AR31_S']", "['AR36_S']", "['AR39_S']", "['AR49_S']", "['AR50_S']", "['AR53_S']", "['AR56_S']"],
["['AR04_S']", "['AR07_S']", "['AR11_S']", "['AR15_S']", "['AR16_S']", "['AR23_S']", "['AR26_S']", "['AR30_S']", "['AR31_S']", "['AR36_S']", "['AR39_S']", "['AR45_S']", "['AR49_S']", "['AR50_S']", "['AR52_S']", "['AR53_S']", "['AR54_S']"],
["['AR11_S']", "['AR22_S']", "['AR23_S']", "['AR26_S']", "['AR27_S']", "['AR30_S']", "['AR31_S']", "['AR36_S']", "['AR39_S']", "['AR48_S']", "['AR49_S']", "['AR50_S']", "['AR52_S']", "['AR53_S']", "['AR57_S']"]
]

disct_1 = {}
disct_2 = {}

def count_instances(x, y):
    return {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}

def sort_results(myDict):
    myKeys = list(myDict.keys())
    myKeys.sort()
    return {i: myDict[i] for i in myKeys}

for arr in s_sr_arm:
    counter = Counter(arr)
    disct_1 = count_instances(disct_1, counter)

print(sort_results(disct_1)) 

