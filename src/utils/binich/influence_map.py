import numpy as np
import matplotlib.colors as mcolors
import seaborn as sns
from PIL import Image
import random
import cv2

import unit_utils

class InfluenceMap:
    def __init__(self, unit_scale: int):
        # the number of units of one player in the map
        self.unit_scale = unit_scale
        # the resolution of the map, default is 128
        self.map_resolution = 128
        # the boundary width of the map
        self.map_boundary_with = 2
        # the influence list of player1 and player2
        self.player1_influence_list = [25]
        self.player2_influence_list = [-16, -9, -4, -1]
        # the max/min influence value
        self.max_influence = 25 * unit_scale
        self.min_influence = -16 * unit_scale
        # the influence map
        self.influence_map = np.zeros((int((pow(self.map_resolution, 1))), int((pow(self.map_resolution, 1)))))


    def make_regalur_image(self, img, size=(256, 256)):
        return img.resize(size).convert('RGB')

    def hashing(self, img):
        open_cv_image = np.array(img)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        chans = cv2.split(open_cv_image)
        colors = ("b", "g", "r")
        for (chans, color) in zip(chans, colors):
            hist = cv2.calcHist([chans], [0], None, [8], [0, 256])
        (b, g, r) = cv2.split(open_cv_image)
        bh = cv2.equalizeHist(b)
        gh = cv2.equalizeHist(g)
        rh = cv2.equalizeHist(r)
        equ2 = cv2.merge((bh, gh, rh))
        chans2 = cv2.split(equ2)
        r_list = []
        g_list = []
        b_list = []
        for (chans2, color) in zip(chans2, colors):
            hist = cv2.calcHist([chans2], [0], None, [8], [0, 256])
            if color == 'r':
                r_list = hist.T[0]
            if color == 'g':
                g_list = hist.T[0]
            if color == 'b':
                b_list = hist.T[0]
        hash_string = ''
        r_max = max(r_list, key=abs)
        g_max = max(g_list, key=abs)
        b_max = max(b_list, key=abs)
        for i in range(8):
            hash_string += '{:01X}'.format(int(r_list[i] / r_max * 15.9))
        for i in range(8):
            hash_string += '{:01X}'.format(int(g_list[i] / g_max * 15.9))
        for i in range(8):
            hash_string += '{:01X}'.format(int(b_list[i] / b_max * 15.9))
        return hash_string

    def map_to_grid(self, x, y):
        i, j = int(x), int(self.map_resolution - y)
        return i, j

    def grid_to_map(self, i, j):
        x, y = random.uniform(i, i + 1), \
               random.uniform(int(self.map_resolution - j) - 1, int(self.map_resolution - j))
        return x, y

    def ripple(self, ripple_level, ripple_center, alliance):
        max_x = self.map_resolution - 1
        max_y = self.map_resolution - 1
        if alliance == 'Self':
            if ripple_level == 0:
                self.influence_map[int(ripple_center[1])][int(ripple_center[2])] += self.player1_influence_list[ripple_level]
            else:
                for dx in range(-abs(ripple_level) if int(ripple_center[1]) > 0 else 0,
                                abs(ripple_level) + 1 if int(ripple_center[1]) < max_x else 0):
                    for dy in range(-abs(ripple_level) if int(ripple_center[2]) > 0 else 0,
                                    abs(ripple_level) + 1 if int(ripple_center[2]) < max_y else 0):
                        if dx != 0 or dy != 0:
                            self.influence_map[int(ripple_center[1]) + dx][int(ripple_center[2]) + dy] \
                                += self.player1_influence_list[ripple_level]
        elif alliance == 'Enemy':
            if ripple_level == 0:
                self.influence_map[int(ripple_center[1])][int(ripple_center[2])] += self.player2_influence_list[ripple_level]
            else:
                for dx in range(-abs(ripple_level) if int(ripple_center[1]) > 0 else 0,
                                abs(ripple_level) + 1 if int(ripple_center[1]) < max_x else 0):
                    for dy in range(-abs(ripple_level) if int(ripple_center[2]) > 0 else 0,
                                    abs(ripple_level) + 1 if int(ripple_center[2]) < max_y else 0):
                        if dx != 0 or dy != 0:
                            self.influence_map[int(ripple_center[1]) + dx][int(ripple_center[2]) + dy] \
                                += self.player2_influence_list[ripple_level]

    def transfer_array_to_img(self, arr: np.ndarray):
        norm = mcolors.TwoSlopeNorm(vmin=-self.min_influence, vcenter=0.0, vmax=self.max_influence)
        p1 = sns.heatmap(arr, cmap='RdBu', norm=norm, annot=False, cbar=False, square=True, xticklabels=False, yticklabels=False)
        s1 = p1.get_figure()
        img = Image.frombytes('RGB', s1.canvas.get_width_height(), s1.canvas.buffer_rgba())
        return self.make_regalur_image(img)

    def calculate_influence_map(self, player1_unit_list, player2_unit_list):
        for player1_unit in player1_unit_list:
            for index in range(len(self.player1_influence_list)):
                self.ripple(index, player1_unit, 'Self')
        for player2_unit in player2_unit_list:
            for index in range(len(self.player2_influence_list)):
                self.ripple(index, player2_unit, 'Enemy')
        return self.influence_map

    def get_map_boundary(self, width):
        rows, cols = len(self.influence_map), len(self.influence_map[0])
        non_zero = [(i, j) for i in range(rows) for j in range(cols) if self.influence_map[i][j] != 0]
        if not non_zero:
            return 40, 44, 40, 44
        min_row, min_col = non_zero[0]
        max_row, max_col = non_zero[0]
        for i, j in non_zero:
            if i < min_row:
                min_row = i
            if i > max_row:
                max_row = i
            if j < min_col:
                min_col = j
            if j > max_col:
                max_col = j
        # 上，下，左，右
        top = max(min_row - width, 0)
        bottom = min(max_row + 1 + width, self.map_resolution)
        left = max(min_col - width, 0)
        right = min(max_col + 1 + width, self.map_resolution)
        return top, bottom, left, right

    def get_im_window(self):
        u_util = unit_utils.UnitUtils()
        player1_units = u_util.get_units(u_util.player1_unit_type, 'SELF')
        player2_units = u_util.get_units(u_util.player1_unit_type, 'ENEMY')
        player1_units_features = sorted([(item['tag'], item['x'], item['y']) for item in player1_units], key=lambda x: x[0])
        player2_units_features = sorted([(item['tag'], item['x'], item['y']) for item in player2_units], key=lambda x: x[0])
        im = self.calculate_influence_map(player1_units_features, player2_units_features)
        top, bottom, left, right = self.get_map_boundary(self.map_boundary_with)
        # window of influence map
        wim = im.T[left:right, top:bottom]
        return wim

    def get_im_hash(self):
        wim = self.get_im_window()
        im_hash = self.hashing(self.transfer_array_to_img(wim))
        return im_hash