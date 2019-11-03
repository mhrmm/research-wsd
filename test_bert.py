import unittest
from wordsense import SenseInstance
from bert import vectorize_instance
import torch
from torch import tensor

class TestBert(unittest.TestCase):
    def test_bert(self):
        instance = SenseInstance(["I", "swung", "the", "bat"], 3, "bat_like_a_fruit_bat")
        result = vectorize_instance(instance)
        expected = tensor(
            [-1.4630e-01,  2.4450e-01, -6.3510e-01, -3.9560e-01, -2.1650e-01,
            -5.0140e-01,  3.0900e-01,  5.0450e-01, -4.6510e-01, -3.0520e-01,
            -2.8000e-03, -2.9200e-02, -5.2180e-01,  3.6780e-01, -7.6220e-01,
             3.5340e-01,  6.3300e-01, -2.0990e-01,  1.1400e-01, -9.3600e-02,
             1.7400e-02, -2.9000e-01, -6.2380e-01,  6.8520e-01,  1.9710e-01,
             4.9820e-01,  3.3090e-01,  5.8900e-01, -1.8800e-02, -1.4805e+00,
             1.3540e-01,  3.9380e-01, -8.5670e-01, -7.6160e-01, -1.8070e-01,
             2.8000e-02, -2.3940e-01, -4.5450e-01, -6.8300e-01,  3.3110e-01,
             2.5510e-01,  1.7710e-01,  4.4190e-01, -3.5880e-01, -6.5480e-01,
            -2.0530e-01,  1.3050e-01, -4.5720e-01, -3.9570e-01, -1.4110e-01,
             2.2470e-01, -7.8200e-02, -2.9920e-01, -1.2480e-01, -4.7900e-01,
             1.1255e+00,  4.9040e-01, -6.1250e-01, -5.5520e-01,  4.0360e-01,
            -5.8460e-01, -1.7970e-01,  8.7220e-01, -5.5090e-01, -2.4340e-01,
            -2.8960e-01,  9.9900e-02,  1.1554e+00, -5.9920e-01,  3.3910e-01,
            -7.5830e-01, -5.1540e-01,  1.7030e-01, -1.2510e-01, -7.7680e-01,
            -4.1240e-01, -8.2180e-01,  6.0860e-01, -1.0220e-01,  2.1130e-01,
             3.5380e-01,  3.2800e-02, -9.1200e-01,  6.4070e-01,  5.6100e-01,
             3.3910e-01, -4.2050e-01, -7.2870e-01,  4.4900e-01,  7.5990e-01,
            -5.2400e-02, -3.8510e-01, -8.2580e-01,  7.2410e-01,  4.7730e-01,
            -5.0730e-01, -7.1600e-02, -7.1320e-01, -1.9660e-01,  1.3166e+00,
             7.2010e-01, -6.5810e-01,  4.3250e-01,  1.4290e-01,  2.9930e-01,
             1.2670e-01,  9.0250e-01,  2.8050e-01, -3.1240e-01,  4.0000e-04,
            -2.8370e-01, -4.4300e-02, -5.0240e-01, -6.1630e-01, -3.2400e-01,
            -5.0980e-01,  1.4667e+00, -8.2760e-01, -2.8190e-01,  3.0590e-01,
             7.1830e-01,  2.4870e-01,  2.1180e-01,  1.2193e+00, -8.3900e-02,
             4.2520e-01, -4.4000e-02,  3.0790e-01,  1.1930e-01, -1.3736e+00,
             4.1870e-01,  3.0960e-01,  1.6680e-01, -1.4263e+00,  3.2190e-01,
             1.0051e+00,  5.1100e-01, -4.7100e-02, -7.6870e-01,  6.2950e-01,
             5.9450e-01, -5.6120e-01,  6.6100e-02, -5.9530e-01,  8.1130e-01,
            -4.9990e-01, -3.5000e-02, -7.6350e-01,  1.2432e+00,  7.7020e-01,
             1.2787e+00, -1.1970e-01, -4.0690e-01, -1.7890e-01, -9.6500e-02,
             1.3730e-01, -3.9890e-01,  6.6770e-01,  1.8020e-01,  6.2240e-01,
             9.9090e-01, -8.9420e-01, -6.2600e-02,  3.0880e-01, -4.3950e-01,
            -4.1620e-01,  2.1970e-01,  1.2287e+00,  8.1510e-01, -2.0190e-01,
            -8.8590e-01, -1.8700e-01,  6.4800e-01, -3.5630e-01, -4.2800e-01,
            -1.0502e+00,  1.9170e-01, -6.9300e-02, -8.2900e-02, -7.5500e-02,
            -2.7240e-01, -2.8000e-02,  3.2650e-01,  5.8590e-01,  1.1810e-01,
            -1.8040e-01,  1.5010e+00, -4.9470e-01, -9.5600e-02,  5.0260e-01,
             7.7370e-01, -5.5030e-01, -4.5600e-01, -3.6430e-01,  2.1120e-01,
            -1.1527e+00, -4.6980e-01, -1.7028e+00,  3.1100e-02,  6.1100e-01,
             3.5980e-01,  4.0970e-01,  1.0623e+00, -2.5100e-02, -1.0282e+00,
            -2.6020e-01, -4.3300e-02,  3.9280e-01, -2.3600e-01,  1.0316e+00,
            -1.3450e-01,  1.2412e+00,  3.2100e-01, -4.9270e-01, -2.4120e-01,
             1.1690e-01, -5.7770e-01,  6.8540e-01, -1.6310e-01,  3.9980e-01,
             6.7640e-01,  5.8090e-01, -1.0646e+00,  2.6470e-01, -2.7200e-02,
             1.9808e+00, -2.5310e-01, -3.2830e-01, -8.1260e-01,  1.2640e+00,
            -4.8710e-01,  8.3100e-02,  7.9260e-01,  7.8190e-01, -7.4600e-02,
            -8.3600e-01, -1.2313e+00,  3.3800e-01, -4.7640e-01, -1.2260e-01,
            -1.3580e-01, -8.0900e-01,  6.4800e-02,  8.6550e-01, -4.1700e-02,
             3.3700e-01, -4.8270e-01,  1.6050e-01,  1.8450e-01, -5.2320e-01,
            -1.6850e-01, -1.1827e+00, -2.1530e-01, -9.8970e-01, -1.0710e+00,
            -1.7010e-01, -2.5990e-01,  7.0620e-01, -1.0840e-01,  7.1670e-01,
             4.3410e-01,  5.6700e-02,  1.1213e+00,  3.3820e-01, -5.2150e-01,
             2.9120e-01, -4.8300e-01,  7.1600e-02, -3.4040e-01,  2.5940e-01,
            -5.7020e-01, -9.0630e-01,  9.5210e-01,  6.1950e-01, -9.7430e-01,
            -2.5200e-01,  5.5950e-01,  2.0000e-01, -6.0860e-01, -9.4800e-01,
             6.6200e-02,  3.3830e-01, -1.0664e+00,  2.3800e-02, -3.7970e-01,
            -9.8890e-01,  2.7570e-01,  1.1230e-01, -3.6250e-01, -4.9690e-01,
             1.5230e-01,  5.6560e-01,  1.2440e-01, -2.8700e-01, -4.3300e-01,
            -8.2170e-01,  5.8050e-01,  5.4580e-01,  1.8590e-01,  8.0600e-02,
            -7.8600e-02, -5.5230e-01,  1.6700e-01,  2.7700e-02, -3.5080e-01,
             4.6100e-01, -8.3650e-01, -1.1220e-01, -3.3713e+00, -1.6400e-02,
            -5.7540e-01, -5.0600e-01,  5.0680e-01, -2.5460e-01, -4.8960e-01,
            -4.3600e-01, -4.4680e-01, -4.0970e-01, -1.9470e-01, -6.0730e-01,
            -9.9000e-03, -4.5300e-01, -3.5990e-01, -1.7250e-01, -1.9640e-01,
            -1.8350e-01,  1.7170e-01,  9.3260e-01,  4.2460e-01, -1.0720e-01,
            -6.4300e-02, -7.7380e-01,  3.1050e-01,  7.5110e-01,  2.9950e-01,
             1.6420e-01, -1.0887e+00, -2.8500e-02, -4.8390e-01,  2.9810e-01,
             1.9340e-01,  5.9600e-02,  2.8140e-01, -3.5880e-01,  6.7260e-01,
             5.0000e-02, -8.7740e-01, -9.2570e-01, -6.3280e-01, -4.0770e-01,
             4.4540e-01,  4.5780e-01,  1.4294e+00,  1.9850e-01,  2.4850e-01,
            -6.2590e-01, -6.0210e-01, -9.8000e-02,  3.9800e-02, -8.3530e-01,
            -4.3530e-01, -6.2390e-01,  6.0500e-02, -4.9400e-01,  9.5380e-01,
            -1.8540e-01,  2.6700e-02, -6.1840e-01,  2.4850e-01,  2.0720e-01,
            -5.8660e-01,  6.5210e-01,  3.7900e-01, -4.6850e-01,  2.6200e-02,
             1.1720e-01,  5.3760e-01,  9.1000e-01,  7.2100e-02, -6.5890e-01,
            -7.6770e-01, -1.4047e+00,  1.8000e-03, -1.7640e-01,  7.6330e-01,
            -4.5280e-01,  5.9940e-01, -8.4700e-02, -9.2970e-01, -5.1390e-01,
             4.1200e-02, -2.6630e-01,  3.6060e-01, -4.4080e-01,  4.0420e-01,
            -7.5810e-01,  3.2030e-01, -6.2240e-01, -1.7930e-01,  1.8200e-02,
             2.1500e-01,  8.6890e-01,  1.3720e-01,  3.6980e-01,  2.1440e-01,
            -4.8120e-01, -3.9210e-01,  4.7680e-01,  3.6100e-01,  2.3100e-02,
             5.7800e-02,  3.2310e-01,  4.4300e-02,  3.5810e-01, -2.2380e-01,
             6.4240e-01, -6.5880e-01, -4.8110e-01,  2.0840e-01, -1.0120e+00,
             7.8870e-01, -3.8930e-01, -8.3780e-01,  2.3190e-01,  1.3100e-02,
             4.4990e-01, -6.3560e-01,  3.5570e-01, -2.7370e-01,  2.5590e-01,
            -1.1083e+00, -4.8240e-01, -7.2260e-01, -2.6150e-01,  2.5800e-01,
            -7.7830e-01, -9.2940e-01, -2.3410e-01, -3.2490e-01, -3.0510e-01,
            -1.2610e-01,  9.1840e-01, -2.5830e-01,  4.1020e-01, -8.9130e-01,
            -4.0020e-01,  4.7120e-01,  3.4380e-01,  1.5120e-01,  7.5970e-01,
             8.6490e-01, -3.9270e-01, -3.3510e-01,  2.6370e-01, -2.7460e-01,
            -5.8590e-01, -1.0750e+00, -8.2200e-02, -5.8700e-02, -8.4190e-01,
             4.0200e-01, -4.8700e-02,  8.3940e-01,  1.0127e+00,  4.0000e-04,
             1.4460e-01, -1.1870e-01,  9.8100e-02,  4.7800e-01, -3.0570e-01,
             5.9880e-01,  5.8850e-01,  1.4710e-01,  4.0570e-01,  2.5200e-02,
            -2.5100e-02, -1.9940e-01, -1.9110e-01, -1.2030e-01, -9.5090e-01,
            -5.6820e-01, -3.3940e-01, -7.4150e-01,  7.9250e-01, -7.6100e-02,
            -4.8110e-01, -1.1135e+00, -5.4350e-01, -1.9630e-01,  1.8840e-01,
            -3.1980e-01, -5.7790e-01, -4.2500e-02, -5.2900e-01, -5.2180e-01,
            -1.1026e+00,  6.3780e-01,  3.7520e-01,  3.7600e-01,  3.1360e-01,
            -3.7920e-01, -1.3900e-01, -5.0380e-01, -1.0912e+00,  1.4400e-01,
             1.0140e+00, -1.7200e-01,  4.1310e-01,  8.6370e-01,  5.6000e-03,
             2.5840e-01, -5.6170e-01,  9.2300e-02, -1.8990e-01, -1.4880e-01,
            -8.7390e-01,  4.1600e-01,  7.1270e-01,  3.3810e-01, -1.9070e-01,
             2.1910e-01, -2.4190e-01,  6.7400e-01, -2.3120e-01, -5.1940e-01,
            -2.3990e-01, -2.6970e-01, -3.1340e-01,  5.5950e-01, -1.1570e-01,
            -2.4600e-02,  2.0430e-01, -1.6950e-01,  3.9610e-01, -3.1500e-01,
             1.2300e-01,  1.9400e-01, -3.6390e-01, -5.4250e-01, -8.8350e-01,
             3.0450e-01,  2.1850e-01,  7.3500e-02,  6.5970e-01, -4.1000e-02,
            -6.7100e-01, -2.3840e-01,  5.0300e-02, -5.7200e-02, -5.3200e-02,
            -3.8100e-02,  5.5760e-01, -6.9400e-02, -4.4300e-01,  6.0000e-03,
             9.9580e-01,  1.6560e-01,  2.6960e-01, -1.6540e-01,  3.1080e-01,
             9.7500e-02, -8.9400e-02,  1.5600e-01, -6.1300e-02, -2.3850e-01,
            -3.6980e-01,  7.5240e-01,  7.0140e-01, -5.4000e-03, -4.6890e-01,
            -1.3530e-01,  7.4240e-01, -8.0000e-03,  2.8250e-01,  5.2170e-01,
            -1.5010e-01,  3.3470e-01,  5.9200e-01,  1.3331e+00, -3.4600e-02,
            -7.5290e-01, -3.7450e-01, -5.7080e-01,  4.8480e-01, -2.4370e-01,
            -3.6410e-01,  5.2370e-01, -4.8320e-01, -4.4020e-01,  4.6230e-01,
             6.4880e-01,  2.5000e-01, -5.3340e-01,  7.0920e-01,  7.6900e-02,
             6.8800e-02, -2.9450e-01, -4.9120e-01,  1.9050e-01,  2.4790e-01,
            -1.3490e-01,  5.6320e-01, -2.0860e-01,  1.1069e+00,  8.2000e-02,
             1.3600e-01,  7.8400e-01,  1.1320e-01,  3.5500e-01, -2.1920e-01,
             5.4930e-01,  8.9380e-01, -3.8060e-01,  3.4000e-03,  1.7730e-01,
            -1.7200e-01,  4.2380e-01,  8.6130e-01, -3.4680e-01, -9.6530e-01,
            -3.2900e-02, -8.5710e-01, -4.1250e-01, -3.4600e-02, -3.0420e-01,
            -6.8800e-02,  1.1410e-01, -1.6320e-01, -5.5560e-01,  6.9290e-01,
             4.6800e-02, -2.2100e-01, -4.3260e-01,  4.5050e-01, -1.1525e+00,
            -4.6870e-01, -1.8340e-01,  1.5100e-01, -7.0400e-02, -6.9020e-01,
            -2.7730e-01,  5.8700e-01, -5.0640e-01,  5.3950e-01, -3.8660e-01,
             1.8950e-01,  1.3962e+00,  5.4600e-01,  4.4660e-01, -2.3060e-01,
            -1.7180e-01,  9.2530e-01,  3.5360e-01,  4.7000e-02,  7.3180e-01,
            -7.9900e-02,  1.1664e+00,  6.4120e-01, -5.9100e-02,  3.3900e-01,
             1.9700e-01, -9.8370e-01,  1.2435e+00,  1.6930e-01,  2.4690e-01,
             2.7720e-01,  3.1910e-01,  1.7800e-01, -1.3090e-01,  9.5100e-02,
             1.7770e-01, -1.2365e+00, -5.1270e-01,  4.4220e-01,  6.2960e-01,
             5.6400e-02, -3.2110e-01,  1.3620e-01, -7.6300e-02,  4.2040e-01,
             6.4720e-01, -3.3980e-01, -4.2830e-01,  6.8880e-01,  4.5060e-01,
            -2.2380e-01,  1.1648e+00, -3.9300e-02, -2.0340e-01,  7.0440e-01,
            -1.1873e+00,  3.8160e-01,  5.5160e-01, -3.3100e-01,  4.0440e-01,
            -2.9400e-02,  7.1920e-01, -3.6700e-01,  1.3203e+00, -3.1770e-01,
             9.2200e-01, -1.0000e-02, -7.1620e-01, -1.0239e+00,  4.6400e-02,
            -5.4730e-01,  1.3830e-01,  5.1540e-01, -1.9700e-02,  1.2030e+00,
            -2.7280e-01,  9.6590e-01,  1.0430e-01,  1.8040e-01, -3.9400e-01,
            -5.6530e-01,  1.2139e+00, -2.2600e-01,  1.3440e-01,  6.6720e-01,
            -4.8110e-01, -2.5200e-02, -6.7160e-01, -4.0650e-01,  4.4410e-01,
             2.6770e-01, -8.0580e-01,  1.4153e+00, -3.6510e-01,  2.5830e-01,
            -6.5000e-01,  6.7650e-01, -1.2115e+00, -2.5650e-01,  1.6610e-01,
             1.0894e+00,  4.1250e-01, -2.9840e-01, -5.5020e-01,  7.0980e-01,
             1.1451e+00, -7.4670e-01,  5.6190e-01, -2.1620e-01, -6.5710e-01,
             3.2280e-01, -4.1130e-01,  4.7090e-01,  4.0410e-01,  3.3860e-01,
             1.8100e-01, -1.8730e-01, -6.0800e-02, -5.0420e-01, -2.5010e-01,
            -1.0339e+00,  8.2530e-01,  3.1960e-01,  1.1000e-03,  8.4850e-01,
            -1.1800e-02, -4.5650e-01, -8.6920e-01, -1.4810e-01,  3.7370e-01,
            -3.1990e-01,  5.5160e-01, -3.2160e-01])
        n_digits = 4
        rounded = torch.round(result * 10**n_digits) / (10**n_digits)
        assert torch.all(torch.eq(rounded, expected))

    

if __name__ == "__main__":
	unittest.main()
