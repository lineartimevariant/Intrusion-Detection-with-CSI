import re
import numpy as np

TARGET_SUBCARRIERS = 64

def parse_csi_line(line):
    if "CSI_DATA" not in line:
        return None

    try:
        matches = re.findall(r"\[(.*?)\]", line)
        if not matches:
            return None

        values = list(map(int, matches[-1].split()))

        if len(values) < 2:
            return None

        if len(values) % 2 != 0:
            values = values[:-1]

        iq = np.array(values, dtype=np.float32)

        real = iq[::2]
        imag = iq[1::2]

        amp = np.sqrt(real**2 + imag**2)
        phase = np.arctan2(imag, real)

        n = min(len(amp), TARGET_SUBCARRIERS)

        amp_fixed = np.zeros(TARGET_SUBCARRIERS, dtype=np.float32)
        phase_fixed = np.zeros(TARGET_SUBCARRIERS, dtype=np.float32)

        amp_fixed[:n] = amp[:n]
        phase_fixed[:n] = phase[:n]

        amp_fixed = (amp_fixed - np.mean(amp_fixed)) / (np.std(amp_fixed) + 1e-6)
        phase_fixed = (phase_fixed - np.mean(phase_fixed)) / (np.std(phase_fixed) + 1e-6)

        features = np.concatenate([amp_fixed, phase_fixed])

        return features

    except Exception:
        return None