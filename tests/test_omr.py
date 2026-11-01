import unittest

import cv2
import numpy as np

import omr
from app import _safe_excel_text, _valid_key


class OMRCoreTests(unittest.TestCase):
    def test_answer_key_validation_accepts_multiple_correct_options(self):
        self.assertTrue(_valid_key([[0], [1, 2], [3]], 3))
        self.assertFalse(_valid_key([[0], [], [3]], 3))
        self.assertFalse(_valid_key([[0], [4], [3]], 3))
        self.assertFalse(_valid_key([[0, 0], [1], [3]], 3))

    def test_excel_text_is_not_interpreted_as_formula(self):
        self.assertEqual(_safe_excel_text('=1+1'), "'=1+1")
        self.assertEqual(_safe_excel_text('Munna'), 'Munna')

    def test_answer_scoring_handles_correct_wrong_blank_and_double(self):
        gray = np.full((1400, 1500), 245, dtype=np.uint8)
        points = []
        for q in range(50):
            block = q // 25
            row = q % 25
            for option in range(4):
                x = 100 + block * 320 + option * 45
                y = 100 + row * 45
                points.append((x, y))

        # Q1: one correct mark; Q2: one accepted alternative;
        # Q3: two marks; Q4: blank.
        for q, options in {0: [1], 1: [2], 2: [0, 1]}.items():
            for option in options:
                x, y = points[q * 4 + option]
                cv2.circle(gray, (int(x), int(y)), 8, 20, -1)

        key = [[1], [1, 2], [0], [3]] + [[0] for _ in range(46)]
        result = omr._scan_answers(gray, points, 4, key, 1500)
        _, _, answers, status, correct, wrong, blank, double, selected = result

        self.assertEqual(status, ['c', 'c', 'd', 'b'])
        self.assertEqual((correct, wrong, blank, double), (2, 2, 1, 1))
        self.assertEqual(selected[:4], [[1], [2], [0, 1], []])
        self.assertEqual(answers[:4], ['খ', 'গ', '*', '-'])


if __name__ == '__main__':
    unittest.main()
