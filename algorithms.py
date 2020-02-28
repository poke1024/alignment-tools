import collections

import numpy as np

from itertools import chain
from functools import lru_cache


Cell = collections.namedtuple('Cell', ['x', 'trace'])


def path_deltas(path):
	path = np.array(path)
	s = path[0]
	deltas = []
	for q, p in zip(path, path[1:]):
		deltas.append(tuple(p - q))
	return tuple(s), deltas


def pick(*cells, f):
	best = f(*[cell.x for cell in cells])
	return Cell(best, chain(
		*[cell.trace for cell in cells if cell.x == best]))


def needleman_wunsch_matrix(sim_matrix, gap_cost):
	sim_matrix = np.array(sim_matrix)

	def S(i, j):
		return sim_matrix[i - 1, j - 1]

	@lru_cache(maxsize=None)
	def M(i, j):
		if i == 0 and j == 0:
			return Cell(
				0, 
				trace=[])

		if j == 0:  # i.e. M[i, 0]
			return Cell(
				M(i - 1, 0).x - gap_cost,
				trace=[(i - 1, 0)])

		if i == 0:  # i.e. M[0, j]
			return Cell(
				M(0, j - 1).x - gap_cost,
				trace=[(0, j - 1)])

		return pick(
			Cell(
				M(i - 1, j - 1).x + S(i, j),
				trace=[(i - 1, j - 1)]),
			Cell(
				M(i - 1, j).x - gap_cost,
				trace=[(i - 1, j)]),
			Cell(
				M(i, j - 1).x - gap_cost,
				trace=[(i, j - 1)]),
			f=max)

	n, m = sim_matrix.shape
	rows = []
	for i in range(n + 1):
		row = []
		for j in range(m + 1):
			row.append(M(i, j))
		rows.append(row)

	return rows


def smith_waterman_matrix(sim_matrix, gap_cost):
	sim_matrix = np.array(sim_matrix)

	def S(i, j):
		return sim_matrix[i - 1, j - 1]

	@lru_cache(maxsize=None)
	def M(i, j):
		if i == 0 and j == 0:
			return Cell(
				0,
				trace=[])

		if j == 0:  # i.e. M[i, 0]
			return Cell(
				0,
				trace=[(i - 1, 0)])

		if i == 0:  # i.e. M[0, j]
			return Cell(
				0,
				trace=[(0, j - 1)])

		return pick(
			Cell(
				M(i - 1, j - 1).x + S(i, j),
				trace=[(i - 1, j - 1)]),
			Cell(
				M(i - 1, j).x - gap_cost,
				trace=[(i - 1, j)]),
			Cell(
				M(i, j - 1).x - gap_cost,
				trace=[(i, j - 1)]),
			Cell(
				0,
				trace=[]),
			f=max)

	n, m = sim_matrix.shape
	rows = []
	for i in range(n + 1):
		row = []
		for j in range(m + 1):
			row.append(M(i, j))
		rows.append(row)

	return rows


def dynamic_time_warping_matrix(sim_matrix):
	sim_matrix = np.array(sim_matrix)

	def S(i, j):
		return sim_matrix[i - 1, j - 1]

	@lru_cache(maxsize=None)
	def M(i, j):
		if i == 0 and j == 0:
			return Cell(0, [])

		if j == 0:  # i.e. M[i, 0]
			return Cell(0, [(i - 1, 0)])

		if i == 0:  # i.e. M[0, j]
			return Cell(0, [(0, j - 1)])

		x, trace = pick(
			Cell(
				M(i - 1, j - 1).x,
				trace=[(i - 1, j - 1)]),
			Cell(
				M(i - 1, j).x,
				trace=[(i - 1, j)]),
			Cell(
				M(i, j - 1).x,
				trace=[(i, j - 1)]),
			f=max)

		return Cell(x + S(i, j), trace=trace)

	n, m = sim_matrix.shape
	rows = []
	for i in range(n + 1):
		row = []
		for j in range(m + 1):
			row.append(M(i, j))
		rows.append(row)

	return rows


def global_alignment_paths(M):
	paths = []

	def build_path(path, i, j):
		path.append((i, j))

		if i == 0 and j == 0:
			paths.append(path)
		else:
			c = Cell(*M[i][j])
			for t in c.trace:
				build_path(path.copy(), *t)

	i = len(M) - 1
	j = len(M[0]) - 1
	build_path([], i, j)

	return paths


def local_alignment_paths(M):
	paths = []

	def build_path(path, i, j):
		path.append((i, j))

		if i == 0 and j == 0:
			paths.append(path)
		else:
			c = Cell(*M[i][j])
			if c.x <= 0:
				paths.append(path)
			else:
				for t in c.trace:
					build_path(path.copy(), *t)

	highest_score = -np.inf
	start = None

	for i in range(len(M)):
		for j in range(len(M[i])):
			x = Cell(*M[i][j]).x
			if x > highest_score:
				highest_score = x
				start = (i, j)

	build_path([], *start)

	return paths


def path_to_alignment(path, a, b):
	path = list(reversed(path))
	s, deltas = path_deltas(path)

	ra = []
	rb = []

	i, j = s
	for d in deltas:
		if d == (1, 1):
			ra.append(a[i])
			rb.append(b[j])
		elif d == (1, 0):
			ra.append(a[i])
			rb.append("-")
		elif d == (0, 1):
			ra.append("-")
			rb.append(b[j])
		else:
			raise ValueError(
				"illegal path delta %s" % str(d))

		di, dj = d
		i += di
		j += dj

	return ra, rb


def path_to_dtw_pairs(path, a, b):
	path = list(reversed(path))
	s, deltas = path_deltas(path)

	ra = []
	rb = []

	i, j = s
	for d in deltas:
		ra.append(a[i])
		rb.append(b[j])
		di, dj = d
		i += di
		j += dj

	return ra, rb
