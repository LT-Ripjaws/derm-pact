import tempfile
import unittest
import importlib.util
from pathlib import Path

from PIL import Image

from medguard.data.audit_derm_near_duplicates import perceptual_hash, secondary_similarity


class PerceptualHashTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("scipy") is None or importlib.util.find_spec("skimage") is None:
            raise unittest.SkipTest("scipy/scikit-image are absent from the bundled runtime")

    def test_hash_is_stable_to_lossless_resize_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            resized_path = Path(directory) / "resized.png"
            image = Image.new("L", (64, 64))
            for y in range(64):
                for x in range(64):
                    image.putpixel((x, y), (x * 3 + y * 5) % 256)
            image.save(path)
            image.resize((128, 128), Image.Resampling.LANCZOS).save(resized_path)
            distance = (perceptual_hash(path) ^ perceptual_hash(resized_path)).bit_count()
            self.assertLessEqual(distance, 4)

    def test_secondary_similarity_recognizes_reencoded_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            copy = Path(directory) / "copy.jpg"
            image = Image.new("RGB", (180, 120))
            for y in range(120):
                for x in range(180):
                    image.putpixel((x, y), ((x * 3) % 256, (y * 5) % 256, (x + y) % 256))
            image.save(source)
            image.resize((360, 240), Image.Resampling.LANCZOS).save(copy, quality=85)
            result = secondary_similarity(source, copy, size=96)
            self.assertGreater(float(result["ssim"]), 0.90)
            self.assertGreater(float(result["normalized_correlation"]), 0.95)

    def test_secondary_similarity_rejects_different_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            horizontal = Path(directory) / "horizontal.png"
            checker = Path(directory) / "checker.png"
            first = Image.new("RGB", (128, 128))
            second = Image.new("RGB", (128, 128))
            for y in range(128):
                for x in range(128):
                    first.putpixel((x, y), (x * 2, x * 2, x * 2))
                    value = 255 if ((x // 8) + (y // 8)) % 2 else 0
                    second.putpixel((x, y), (value, value, value))
            first.save(horizontal)
            second.save(checker)
            result = secondary_similarity(horizontal, checker, size=96)
            self.assertLess(float(result["ssim"]), 0.50)
