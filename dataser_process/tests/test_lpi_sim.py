"""物理不变量、全量配额及可恢复生成的回归测试。"""

import copy
import json
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lpi_sim.common import file_hash, load_config, read_json, seed_for, validate_config
from lpi_sim.generation import generate_job
from lpi_sim.planning import plan_jobs, quotas
from lpi_sim.signals import chip_indices, noisy_scene, parameters, waveform
from lpi_sim.time_frequency import Spectrogram, energy_bbox
from lpi_sim.workflow import execute, verify_existing


def default_config():
    return load_config("dataser_process/configs/lpi5_v1.yaml")


def small_config(output):
    cfg = default_config()
    cfg["output"] = str(output)
    cfg["snr_db"] = [-10, 10]
    cfg["splits"] = {s: {"scenes": 30, "negatives": 3} for s in ("train", "val", "test")}
    cfg["signal"].update(fs_hz=64000000, duration_s=0.000064, carrier_hz=[-10000000, 10000000],
                         pulse_s=[0.000020, 0.000040], bandwidth_hz=[4000000, 12000000], bfsk_spacing_hz=[4000000, 10000000])
    cfg["stft"].update(window=64, hop=32, nfft=256, image_size=96)
    return cfg


class PlanningTests(unittest.TestCase):
    def test_full_quotas_combinations_and_discrete_parameters(self):
        cfg = default_config()
        jobs = plan_jobs(cfg)
        self.assertEqual(sum(v["images"] for v in quotas(cfg).values()), 99000)
        for split in cfg["splits"]:
            sjobs = [j for j in jobs if j["split"] == split and j["classes"]]
            n = len(sjobs)
            per_class = Counter(c for j in sjobs for c in j["classes"])
            self.assertEqual(len(set(per_class.values())), 1)
            self.assertEqual(sum(j["force_overlap"] for j in sjobs), n//3)
            for k in range(1, 6):
                counts = Counter(tuple(j["classes"]) for j in sjobs if len(j["classes"]) == k)
                self.assertLessEqual(max(counts.values())-min(counts.values()), 1)
                self.assertEqual(sum(counts.values()), n//(3 if k == 1 else 6))
        self.assertEqual(quotas(cfg)["test"]["boxes_per_class"], 23760)
        ids = [i for j in jobs for i in range(j["image_start"], j["image_start"]+(11 if j["classes"] else 1))]
        self.assertEqual(ids, list(range(1, 99001)))

    def test_bad_configs(self):
        for change in (lambda c: c["splits"]["train"].update(scenes=31),
                       lambda c: c.update(snr_db=[2, 2]),
                       lambda c: c["signal"].update(carrier_hz=[0, 300000000]),
                       lambda c: c["stft"].update(hop=99999),
                       lambda c: c["annotation"].update(quantiles=[-.1, 1])):
            cfg = default_config()
            change(cfg)
            with self.assertRaises(ValueError):
                validate_config(cfg)


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = default_config()
        self.job = {"scene_id": "physics", "classes": list(range(5)), "variants": [0]*5}
        self.params = parameters(self.cfg, self.job, 0)

    def test_pulse_support_and_unit_power(self):
        for p in self.params:
            w = waveform(p, self.cfg)
            self.assertEqual(np.count_nonzero(w), p["length"])
            self.assertTrue(np.all(w[:p["start"]] == 0))
            self.assertTrue(np.all(w[p["start"]+p["length"]:] == 0))
            self.assertAlmostEqual(np.mean(abs(w[p["start"]:p["start"]+p["length"]])**2), 1, places=12)

    def test_lfm_nlfm_instantaneous_frequency(self):
        fs = self.cfg["signal"]["fs_hz"]
        for p in self.params[:2]:
            w = waveform(p, self.cfg)[p["start"]:p["start"]+p["length"]]
            # 离散相位差等于每个采样区间瞬时频率的平均值。
            measured = np.angle(w[1:]*w[:-1].conj())*fs/(2*np.pi)
            u = (np.arange(len(w)-1)+.5)/len(w)
            a = p.get("curvature", 0)
            expected = p["fc_hz"]+p["direction"]*p["bandwidth_hz"]*(u+a*(u-u*u)-.5)
            np.testing.assert_allclose(measured, expected, atol=1)

    def test_coded_phases_and_bfsk_frequencies(self):
        fs = self.cfg["signal"]["fs_hz"]
        for p in self.params[2:]:
            w = waveform(p, self.cfg)[p["start"]:p["start"]+p["length"]]
            t = np.arange(len(w))/fs
            if p["class_name"] in ("BPSK", "BFSK"):
                code = np.asarray(p["code"])
                self.assertEqual(code.sum()*2, len(code))
                transitions = np.count_nonzero(np.diff(code))
                self.assertTrue(len(code)/4 <= transitions <= len(code)*3/4)
                bits = code[chip_indices(len(w), len(code))]
                if p["class_name"] == "BFSK":
                    frequency = np.angle(w[1:]*w[:-1].conj())*fs/(2*np.pi)
                    np.testing.assert_allclose(frequency, (p["fc_hz"]+(bits-.5)*p["spacing_hz"])[:-1], atol=.01)
                else:
                    demod = w*np.exp(-1j*(2*np.pi*p["fc_hz"]*t+p["phase_rad"]))
                    np.testing.assert_allclose(demod, 1-2*bits, atol=1e-10)
            else:
                m = p["order"]
                phase = np.outer(np.arange(m), np.arange(m)).reshape(-1)*2*np.pi/m
                demod = w*np.exp(-1j*(2*np.pi*p["fc_hz"]*t+p["phase_rad"]))
                np.testing.assert_allclose(demod, np.exp(1j*phase[chip_indices(len(w), m*m)]), atol=1e-10)

    def test_chip_sampling_boundaries(self):
        counts = np.bincount(chip_indices(12017, 64))
        self.assertEqual(counts.sum(), 12017)
        self.assertLessEqual(counts.max()-counts.min(), 1)

    def test_actual_snr_all_levels_and_noise_only(self):
        waves = [waveform(p, self.cfg) for p in self.params]
        for snr in self.cfg["snr_db"]:
            x, m = noisy_scene(waves, self.params, self.cfg, seed_for(5, snr), snr)
            self.assertTrue(np.isfinite(x).all())
            for component in m["components"]:
                self.assertLess(abs(component["snr_db"]-snr), 1e-10)
        noise, m = noisy_scene([], [], self.cfg, 17, None)
        self.assertEqual(m["components"], [])
        self.assertEqual(m["mixture_snr_db"], {})
        self.assertAlmostEqual(np.mean(abs(noise)**2), 1, delta=.02)

    def test_stft_orientation_and_physical_box(self):
        cfg = self.cfg
        fs = cfg["signal"]["fs_hz"]
        tf = Spectrogram(cfg)
        p = self.params[0].copy()
        p.update(class_name="BPSK", code=[0]*16, fc_hz=60000000, start=12000, length=18000)
        power = tf.power(waveform(p, cfg))
        y = np.argmax(power.sum(axis=1))
        expected_y = tf.size*(.5-p["fc_hz"]/fs)-.5
        self.assertLess(abs(y-expected_y), 2)
        x, y, w, h = energy_bbox(power, cfg)
        expected_start = p["start"]/tf.n*tf.size
        expected_end = (p["start"]+p["length"])/tf.n*tf.size
        self.assertLess(abs(x-expected_start), 4)
        self.assertLess(abs(x+w-expected_end), 4)
        p["fc_hz"] = -60000000
        low = np.argmax(tf.power(waveform(p, cfg)).sum(axis=1))
        self.assertGreater(low, tf.size/2)

    def test_fixed_render_and_agc(self):
        tf = Spectrogram(self.cfg)
        w = waveform(self.params[0], self.cfg)
        np.testing.assert_allclose(tf.power(w, True), tf.power(w*13, True), atol=1e-12)
        a, _ = tf.render(np.full((2, 2), 1e-3))
        b, _ = tf.render(np.array([[1e-3, 1], [1e-9, 1e-3]]))
        self.assertEqual(a[0, 0, 0], b[0, 0, 0])


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.a = Path(cls.tmp.name)/"serial"
        cls.b = Path(cls.tmp.name)/"parallel"
        cls.cfg = small_config(cls.a)
        execute(cls.cfg, workers=1)
        cfg_b = small_config(cls.b)
        execute(cfg_b, workers=2)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_serial_parallel_and_coco(self):
        for folder in ("images", "labels", "annotations", "metadata"):
            files = list((self.a/folder).rglob("*"))
            for p in files:
                if p.is_file():
                    self.assertEqual(file_hash(p), file_hash(self.b/p.relative_to(self.a)))
        from pycocotools.coco import COCO
        coco = COCO(str(self.a/"annotations/instances_test.json"))
        self.assertEqual(len(coco.getImgIds()), 63)
        self.assertEqual(len(coco.getCatIds()), 5)
        self.assertEqual(len(coco.getAnnIds()), 160)

    def test_verify_is_read_only_and_completed_resume(self):
        before = {str(p): p.stat().st_mtime_ns for p in self.a.rglob("*") if p.is_file()}
        verify_existing(self.cfg)
        after = {str(p): p.stat().st_mtime_ns for p in self.a.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(execute(self.cfg, workers=2, resume=True)["status"], "passed")

    def test_reject_overwrite_and_configuration_drift(self):
        with self.assertRaisesRegex(ValueError, "非空"):
            execute(self.cfg)
        changed = copy.deepcopy(self.cfg)
        changed["seed"] += 1
        with self.assertRaisesRegex(ValueError, "改变"):
            execute(changed, resume=True)

    def test_interrupted_resume_and_missing_file(self):
        dest = Path(self.tmp.name)/"interrupted"
        cfg = small_config(dest)
        calls = 0
        def interrupted(args):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError("simulated interruption")
            return generate_job(args)
        with patch("lpi_sim.generation.generate_job", interrupted):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                execute(cfg, workers=1)
        self.assertEqual(read_json(dest/"dataset_manifest.json")["status"], "generating")
        (dest/"images"/"train"/".partial-abandoned").write_bytes(b"incomplete atomic write")
        # 已完成场景缺少文件时也必须可重建；内容已变更则不能覆盖。
        next((dest/"images").rglob("*.png")).unlink()
        execute(cfg, workers=2, resume=True)
        self.assertFalse((dest/"images"/"train"/".partial-abandoned").exists())
        for p in (self.a/"images").rglob("*.png"):
            self.assertEqual(file_hash(p), file_hash(dest/p.relative_to(self.a)))

    def test_detect_corruption_and_extra_files(self):
        dest = Path(self.tmp.name)/"corrupt"
        shutil.copytree(self.a, dest)
        cfg = small_config(dest)
        # 直接调用校验器，以避免输出根路径变化先触发配置哈希保护。
        from lpi_sim.validation import verify_dataset
        label = next((dest/"labels").rglob("*.txt"))
        original = label.read_bytes()
        label.write_text("0 0.5 0.5 0.2 0.2\n")
        with self.assertRaisesRegex(ValueError, "哈希"):
            verify_dataset(cfg, dest)
        label.write_bytes(original)
        (dest/"labels"/"train"/"extra.txt").write_text("")
        with self.assertRaisesRegex(ValueError, "额外"):
            verify_dataset(cfg, dest)

    def test_preview_artifacts_are_deterministic(self):
        left = Path(self.tmp.name)/"preview-serial"
        right = Path(self.tmp.name)/"preview-parallel"
        for root, workers in ((left, 1), (right, 2)):
            cfg = small_config(root)
            cfg["splits"]["val"] = {"scenes": 0, "negatives": 0}
            cfg["splits"]["test"] = {"scenes": 0, "negatives": 0}
            execute(cfg, workers=workers, preview=True)
        for p in (left/"quality_report").rglob("*"):
            if p.suffix in (".npz", ".png"):
                self.assertEqual(file_hash(p), file_hash(right/p.relative_to(left)))


if __name__ == "__main__":
    unittest.main()
