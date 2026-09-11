"""固定物理坐标的 STFT、能量标注和预览输出。"""

import io

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.signal import ShortTimeFFT, windows


class Spectrogram:
    def __init__(self, cfg):
        self.cfg = cfg
        s, t = cfg["signal"], cfg["stft"]
        self.size = t["image_size"]
        self.transform = ShortTimeFFT(windows.hann(t["window"], sym=False), t["hop"], s["fs_hz"],
                                     fft_mode="centered", mfft=t["nfft"], scale_to="magnitude")
        self.n = round(s["fs_hz"]*s["duration_s"])
        self.p1 = int(np.ceil(self.n/t["hop"]))+1
        # 输出像素中心按物理坐标采样；频率轴上高下低。
        times = (np.arange(self.size)+0.5)*s["duration_s"]/self.size
        freq = s["fs_hz"]/2-(np.arange(self.size)+0.5)*s["fs_hz"]/self.size
        fi = (freq+s["fs_hz"]/2)/(s["fs_hz"]/t["nfft"])
        ti = times/(t["hop"]/s["fs_hz"])
        self.coordinates = np.array(np.meshgrid(fi, ti, indexing="ij"))

    def power(self, x, agc=False):
        if agc:
            rms = np.sqrt(np.mean(np.abs(x)**2))
            if not np.isfinite(rms) or rms <= 0:
                raise ValueError("接收信号功率无效。")
            x = x/rms
        z = self.transform.stft(x, p0=0, p1=self.p1, padding="zeros")
        p = np.abs(z)**2
        # FFT 频轴周期延拓一行，保证 Nyquist 邻域的像素插值不引入常量边框。
        p = np.concatenate((p, p[:1]), axis=0)
        result = map_coordinates(p, self.coordinates, order=1, mode="nearest", prefilter=False)
        if not np.isfinite(result).all() or result.sum() <= 0:
            raise ValueError("STFT 能量无效。")
        return result

    def render(self, power):
        lo, hi = self.cfg["stft"]["db_range"]
        db = 10*np.log10(np.maximum(power, np.finfo(np.float64).tiny))
        gray = np.rint(np.clip((db-lo)/(hi-lo), 0, 1)*255).astype(np.uint8)
        stats = {"black_fraction": float(np.mean(gray == 0)), "white_fraction": float(np.mean(gray == 255))}
        return np.repeat(gray[:, :, None], 3, axis=2), stats


def png_bytes(rgb):
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG", compress_level=6)
    return stream.getvalue()


def energy_bbox(power, cfg):
    low, high = cfg["annotation"]["quantiles"]
    pad = cfg["annotation"]["padding_px"]
    intervals = []
    for marginal in (power.sum(axis=0), power.sum(axis=1)):
        cum = np.cumsum(marginal)/marginal.sum()
        start = max(0, int(np.searchsorted(cum, low))-pad)
        end = min(len(marginal), int(np.searchsorted(cum, high))+1+pad)
        if end <= start:
            raise ValueError("能量框退化。")
        intervals.append((start, end))
    (x0, x1), (y0, y1) = intervals
    return [x0, y0, x1-x0, y1-y0]


def box_overlap(a, b):
    x = max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]))
    y = max(0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    return x*y/min(a[2]*a[3], b[2]*b[3])


def support_mask(power, fraction):
    values = np.sort(power.ravel())[::-1]
    idx = min(len(values)-1, np.searchsorted(np.cumsum(values), values.sum()*fraction))
    return power >= values[idx]


def support_overlap(powers, fraction):
    masks = [support_mask(p, fraction) for p in powers]
    return [{"pair": [i, j], "intersection_over_smaller": float(np.count_nonzero(a & b)/min(a.sum(), b.sum()))}
            for i, a in enumerate(masks) for j, b in enumerate(masks) if i < j]


def yolo_text(annotations, size):
    lines = []
    for a in annotations:
        x, y, w, h = a["bbox"]
        lines.append(f"{a['category_id']-1} {(x+w/2)/size:.10f} {(y+h/2)/size:.10f} {w/size:.10f} {h/size:.10f}\n")
    return "".join(lines)
