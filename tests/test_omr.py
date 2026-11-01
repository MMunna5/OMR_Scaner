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

    def test_portrait_detector_ignores_outline_but_reads_filled_bubble(self):
        blank = np.full((25, 25), 185, dtype=np.uint8)
        cv2.circle(blank, (12, 12), 8, 35, 1)
        self.assertFalse(omr._portrait_marked(blank, 12, 12, 180))

        filled = blank.copy()
        cv2.circle(filled, (12, 12), 5, 25, -1)
        self.assertTrue(omr._portrait_marked(filled, 12, 12, 180))

    def test_portrait_detector_reads_mark_on_tinted_paper(self):
        tinted = np.full((25, 25), 145, dtype=np.uint8)
        cv2.circle(tinted, (12, 12), 8, 65, 1)
        cv2.circle(tinted, (12, 12), 4, 55, -1)
        self.assertTrue(omr._portrait_marked(tinted, 12, 12, 140))

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


    def test_rectify_recovers_sideways_standard_sheet(self):
        sheet = np.full((1400, 1500, 3), 245, dtype=np.uint8)
        for i in range(25):
            x = 200 + i * 40
            cv2.rectangle(sheet, (x - 12, 190), (x + 12, 210), (0, 0, 255), -1)
            cv2.rectangle(sheet, (x - 2, 1185), (x + 2, 1215), (0, 0, 255), -1)

        sideways = cv2.rotate(sheet, cv2.ROTATE_90_CLOCKWISE)
        aligned, profile = omr.rectify(sideways)
        self.assertEqual(profile, 'standard')
        self.assertEqual(aligned.shape[:2], (omr.FH, omr.FW))


if __name__ == '__main__':
    unittest.main()
