def frequency(tol_str, tar_str):
	"""Generate the frequency of tar_str in tol_str.

	:param tol_str: mother string.
	:param tar_str: substring.
	"""
	i, j, tar_count = 0, 0, 0
	len_tol_str = len(tol_str)
	len_tar_str = len(tar_str)
	while i < len_tol_str and j < len_tar_str:
		if tol_str[i] == tar_str[j]:
			i += 1
			j += 1
			if j >= len_tar_str:
				tar_count += 1
				i = i - j + 1
				j = 0
		else:
			i = i - j + 1
			j = 0

	return tar_count

#todo: the following two defs are not being uesd in the project and the functions waited to be implemented
def extra_aaindex(filename):
	pass


def norm_index_vals(index_dict):
	pass
