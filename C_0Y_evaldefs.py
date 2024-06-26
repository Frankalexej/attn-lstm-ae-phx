import numpy as np

def filter_data_by_tags(data, tags, select):
    # Convert tag_list to a set for faster membership testing
    tag_set = set(select)

    # Create a boolean mask where True indicates the tag is in tag_list
    mask = np.isin(tags, list(tag_set))

    # Use the mask to filter the data and tags arrays
    filtered_data = data[mask]
    filtered_tags = tags[mask]

    return filtered_data, filtered_tags

def filter_data_by_tags_to_list(data, tags, tag_list):
    result = []

    for tag in tag_list:
        # Create a boolean mask where True indicates the tag matches the current tag
        mask = (tags == tag)
        # Use the mask to filter the data array for the current tag
        filtered_data = data[mask]
        # Append the filtered data to the result list
        result.append(filtered_data)

    return result

def postproc_standardize(data, tags, outlier_ratio=0): 
    if outlier_ratio > 0: 
        # Step 1: Remove outliers
        low_percentile = np.percentile(data, outlier_ratio, axis=0)
        high_percentile = np.percentile(data, 100-outlier_ratio, axis=0)

        # Keep rows where all elements are within the percentiles
        mask = (data > low_percentile) & (data < high_percentile)
        filtered_data = data[mask.all(axis=1)]
        filtered_tags = tags[mask.all(axis=1)]
    else: 
        filtered_data = data
        filtered_tags = tags

    # Step 2: Standardization
    # Calculate mean and std only from the filtered data
    mean = np.mean(filtered_data, axis=0)
    std = np.std(filtered_data, axis=0)

    # Z-score standardization
    standardized_data = (filtered_data - mean) / std
    return standardized_data, filtered_tags

def indicator_function(condition):
    return 1 if condition else 0

def unsym_abx_error(cap_delta, cap_ksi, distance):
    n_delta = cap_delta.shape[0]
    n_ksi = cap_ksi.shape[0]
    sum_value = 0

    for a in range(n_delta):
        for b in range(n_ksi):
            for x in range(n_delta):
                if x != a:
                    sum_value += (indicator_function(
                        distance(cap_delta[x], cap_ksi[b]) < distance(cap_delta[a], cap_delta[x])
                    ) + 0.5 * indicator_function(
                        distance(cap_delta[x], cap_ksi[b]) == distance(cap_delta[a], cap_delta[x])
                    ))

    return 1 / (n_delta * (n_delta - 1) * n_ksi) * (sum_value)

def sym_abx_error(cap_delta, cap_ksi, distance):
    return 0.5 * (unsym_abx_error(cap_delta, cap_ksi, distance) + unsym_abx_error(cap_ksi, cap_delta, distance))

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))